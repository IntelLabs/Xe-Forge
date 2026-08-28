"""
SYCL language backend (plan §11.4 - §11.7).

On Intel this is where the time actually is. In a real vLLM-XPU or SGLang-XPU decode
run a large share of GPU time sits in SYCL C++ — ATen XPU operators, extension ops and
templated sycl-tla GEMM/attention — with oneDNN and oneMKL taking most of the rest. A
Triton-only pipeline optimizes the tail and reports it as a win.

Three things make SYCL different from Triton in *mechanism*, not in status:

* **Identity** comes from a mangled symbol, not a Python function. Demangle it, map
  the functor/lambda type back to a translation unit, and record whether the kernel
  was AOT-compiled or JIT-compiled from SPIR-V — a bundle that AOT-builds a kernel the
  workload JITs is not the same kernel (§11.4).
* **Closure** comes from the build graph, not an AST walk. `compile_commands.json`
  gives the exact compile line: every include path, every define, every flag. This is
  more reliable than the Triton path, not less, because the build system is
  authoritative where AST resolution is inferential (§11.5).
* **Compiler options are a first-class action space** (§11.7). GRF mode, sub-group
  size, AOT target and the floating-point contract are cheap, deterministic wins that
  should be swept *before* any agent is invoked. An agent asked to rewrite a kernel
  that is simply running in the wrong GRF mode will produce an expensive, complicated,
  worse answer.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from xe_forge.orbit.languages.base import (
    BaseLanguageBackend,
    BuildResult,
    CompilerAxis,
    CostProfile,
)
from xe_forge.orbit.models import (
    BuildRecipe,
    ExtractionCheck,
    KernelBundle,
    KernelLanguage,
    ResolutionMethod,
    SourceLocation,
)

# Itanium-mangled SYCL kernel names embed the functor type after a typeinfo prefix.
_MANGLED_RE = re.compile(r"_ZTS\w+")
_TEMPLATE_ARGS_RE = re.compile(r"<(.+)>")
_SYCL_HINTS = ("sycl", "_ZTS", "cutlass", "xe_mma", "at::native::xpu", "torch_ipex")

# Device targets the SYCL AOT compiler accepts, keyed by how the device reports itself.
# Mirrors the mapping Xe-Forge's own sycl executor already performs.
DEVICE_TO_AOT_TARGET = {
    "b580": "bmg-g31",
    "b570": "bmg-g31",
    "battlemage": "bmg-g31",
    "bmg": "bmg-g31",
    "a770": "acm-g10",
    "a750": "acm-g10",
    "arc": "acm-g10",
    "pvc": "pvc",
    "max": "pvc",
    "lunar": "lnl-m",
    "lnl": "lnl-m",
}


@dataclass
class CompileCommand:
    """One entry from a compile-commands database."""

    file: Path
    directory: Path
    arguments: list[str] = field(default_factory=list)

    @property
    def includes(self) -> list[str]:
        out: list[str] = []
        for i, arg in enumerate(self.arguments):
            if arg.startswith("-I") and len(arg) > 2:
                out.append(arg[2:])
            elif arg == "-I" and i + 1 < len(self.arguments):
                out.append(self.arguments[i + 1])
        return out

    @property
    def defines(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for i, arg in enumerate(self.arguments):
            raw = None
            if arg.startswith("-D") and len(arg) > 2:
                raw = arg[2:]
            elif arg == "-D" and i + 1 < len(self.arguments):
                raw = self.arguments[i + 1]
            if raw:
                name, _, value = raw.partition("=")
                out[name] = value
        return out

    @property
    def compiler(self) -> str:
        return self.arguments[0] if self.arguments else "icpx"

    @property
    def is_sycl(self) -> bool:
        return any(a.startswith("-fsycl") for a in self.arguments)


class SyclBackend(BaseLanguageBackend):
    """Identity, build-graph closure, compiler axes and verification for SYCL."""

    name = "sycl"
    language = KernelLanguage.SYCL
    # A rebuild, not a JIT compile: minutes, not seconds. This ratio is why budget
    # accounting must price SYCL candidates separately (§11.6).
    cost_profile = CostProfile(
        build_seconds=120.0,
        iteration_seconds=120.0,
        notes=(
            "Single-TU rebuild against installed headers. Roughly an order of "
            "magnitude more expensive per candidate than Triton; warm ccache and "
            "isolate the translation unit or an eight-hour budget disappears into "
            "three trials."
        ),
    )

    def __init__(
        self,
        compile_commands: Path | None = None,
        source_tree: Path | None = None,
    ) -> None:
        self.compile_commands_path = Path(compile_commands) if compile_commands else None
        # Intel's kernel surface ships as open source but installs as a compiled wheel,
        # so a build database usually does not exist on a user's machine. Pointing at a
        # checked-out tree recovers the kernel-to-file mapping without a build.
        self.source_tree = Path(source_tree) if source_tree else None
        self._db: list[CompileCommand] | None = None
        self._source_index: dict[str, Path] | None = None

    def source_index(self) -> dict[str, Path]:
        if self._source_index is None:
            self._source_index = index_source_tree(self.source_tree) if self.source_tree else {}
        return self._source_index

    # -- identity ---------------------------------------------------------

    def identify(self, kernel_name: str) -> float:
        lowered = kernel_name.lower()
        if _MANGLED_RE.search(kernel_name):
            return 0.9
        if any(hint in lowered for hint in _SYCL_HINTS):
            return 0.75
        return 0.0

    def demangle(self, symbol: str) -> str:
        """Demangle a SYCL kernel symbol, falling back to the raw name.

        Uses `c++filt` when present. Its absence is not an error — it costs confidence,
        not correctness, and the resolver reports the reduced confidence honestly.
        """
        if not symbol.startswith("_Z"):
            return symbol
        filt = shutil.which("c++filt")
        if filt is None:
            return symbol
        try:
            result = subprocess.run(
                [filt, symbol], capture_output=True, text=True, timeout=10, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return symbol

    def resolve_source(self, kernel_name: str, launch=None) -> SourceLocation:
        """Map a mangled kernel name back to a translation unit.

        Confidence is graded, not binary (§11.4): a name that resolves to a unique
        translation unit is high confidence; a heavily templated lambda matching
        several instantiations is not, and we say so rather than picking one.
        """
        demangled = self.demangle(kernel_name)
        template_args = _TEMPLATE_ARGS_RE.findall(demangled)
        instantiations = _split_template_args(template_args[0]) if template_args else []

        # A source-tree index resolves the functor directly, which is both more exact
        # than filename matching and available without a build.
        index = self.source_index()
        if index:
            for symbol in identifying_symbols(demangled):
                hit = index.get(symbol)
                if hit is not None:
                    # An exact index hit carries no confidence figure: the symbol
                    # either matched or it did not. The open question is which
                    # instantiation ran, and that is `candidates`, not a probability
                    # that the file is wrong.
                    return SourceLocation(
                        file=str(hit),
                        symbol=demangled,
                        method=ResolutionMethod.SYMBOL_INDEX,
                        candidates=instantiations,
                    )

        db = self._load_db()
        matches: list[CompileCommand] = []
        if db:
            for symbol in identifying_symbols(demangled) or [_identifier_stem(demangled) or ""]:
                matches = [c for c in db if symbol.lower() in c.file.name.lower()]
                if matches:
                    break

        if len(matches) == 1:
            # The build system named this translation unit, which is authoritative
            # (§11.5) — again exact rather than estimated.
            return SourceLocation(
                file=str(matches[0].file),
                symbol=demangled,
                method=ResolutionMethod.BUILD_GRAPH,
                candidates=instantiations,
            )

        if len(matches) > 1:
            # Several translation units matched by name, so nothing was resolved:
            # the candidates are the finding, and a single file must not be invented
            # from them (§11.4).
            return SourceLocation(
                symbol=demangled,
                method=ResolutionMethod.UNRESOLVED,
                confidence=0.4,
                candidates=[str(m.file) for m in matches],
            )

        # No build database, or no match in it. Templated names without the concrete
        # instantiation are explicitly ambiguous: a sycl-tla kernel without its tile
        # shape and layout parameters is not a kernel (§11.5).
        confidence = 0.5 if not instantiations else 0.3
        return SourceLocation(
            symbol=demangled,
            method=ResolutionMethod.UNRESOLVED,
            confidence=confidence,
            candidates=instantiations,
        )

    # -- closure ----------------------------------------------------------

    def _load_db(self) -> list[CompileCommand]:
        if self._db is not None:
            return self._db
        self._db = []
        if self.compile_commands_path and self.compile_commands_path.is_file():
            try:
                raw = json.loads(self.compile_commands_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return self._db
            if isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict) or "file" not in entry:
                        continue
                    arguments = entry.get("arguments")
                    if arguments is None and isinstance(entry.get("command"), str):
                        arguments = entry["command"].split()
                    self._db.append(
                        CompileCommand(
                            file=Path(entry["file"]),
                            directory=Path(entry.get("directory", ".")),
                            arguments=list(arguments or []),
                        )
                    )
        return self._db

    def closure(self, source: SourceLocation) -> list[Path]:
        """Translation unit plus its header closure, resolved by the preprocessor."""
        if not source.file:
            return []
        tu = Path(source.file)
        if not tu.is_file():
            return []

        files = [tu]
        command = self._command_for(tu)
        headers = self._preprocessed_headers(tu, command)
        if headers:
            files.extend(headers)
        else:
            # Falling back to a textual include scan is weaker and must be visible as
            # such to the caller, which is why it is separated from the real answer.
            files.extend(_scan_local_includes(tu))
        # Deduplicate while preserving order.
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in files:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def _command_for(self, tu: Path) -> CompileCommand | None:
        for entry in self._load_db():
            if entry.file == tu or entry.file.name == tu.name:
                return entry
        return None

    def _preprocessed_headers(self, tu: Path, command: CompileCommand | None) -> list[Path]:
        """Ask the compiler for the real header closure, if a compiler is available."""
        compiler = command.compiler if command else _first_available_compiler()
        if compiler is None or shutil.which(compiler.split("/")[-1]) is None:
            return []
        argv = [compiler, "-M", "-MG", str(tu)]
        if command:
            argv.extend(f"-I{inc}" for inc in command.includes)
            argv.extend(f"-D{k}={v}" if v else f"-D{k}" for k, v in command.defines.items())
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0:
            return []
        return _parse_make_deps(result.stdout)

    def build_recipe(self, source: SourceLocation) -> BuildRecipe | None:
        """The exact compile line, verbatim, so a rebuild reproduces the same binary."""
        if not source.file:
            return None
        command = self._command_for(Path(source.file))
        if command is None:
            # No compile-commands entry, so this line is reconstructed rather than
            # verbatim. Two details matter enough to get right anyway: the compiler must
            # be found even when oneAPI is not on PATH, and the standard must be C++20 —
            # torch-xpu-ops uses `requires` clauses, and defaulting to C++17 produces a
            # wall of errors inside framework headers that reads like a broken closure.
            from xe_forge.orbit.patch.sycl_override import available_compiler

            return BuildRecipe(
                compiler=available_compiler() or _first_available_compiler() or "icpx",
                flags=["-fsycl", "-std=c++20"],
                entry_symbol=source.symbol or "",
            )
        return BuildRecipe(
            compiler=command.compiler,
            flags=[a for a in command.arguments[1:] if a.startswith("-")],
            includes=command.includes,
            defines=command.defines,
            entry_symbol=source.symbol or "",
            aot_target=_aot_target_from_flags(command.arguments),
        )

    # -- build and verify --------------------------------------------------

    def build(self, bundle: KernelBundle) -> BuildResult:
        """Single-TU isolation: build one translation unit, not the whole repository.

        That distinction is the difference between a two-minute loop and a forty-minute
        one (§11.6).
        """
        if bundle.build is None:
            return BuildResult(ok=False, reason="bundle has no BuildRecipe")
        if not bundle.primary_source:
            return BuildResult(ok=False, reason="bundle has no primary source")

        compiler = bundle.build.compiler
        if shutil.which(compiler) is None:
            return BuildResult(
                ok=False,
                reason=(
                    f"SYCL compiler {compiler!r} not on PATH; install oneAPI to build "
                    f"SYCL bundles. Identity, closure and option axes still work "
                    f"without it."
                ),
            )

        source = Path(bundle.primary_source)
        output = source.with_suffix(".so")
        argv = [compiler, *bundle.build.flags, "-shared", "-fPIC", str(source), "-o", str(output)]
        argv.extend(f"-I{inc}" for inc in bundle.build.includes)
        argv.extend(f"-D{k}={v}" if v else f"-D{k}" for k, v in bundle.build.defines.items())

        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return BuildResult(ok=False, reason=f"build failed to start: {exc}")

        diagnostics = _parse_device_diagnostics(result.stderr)
        return BuildResult(
            ok=result.returncode == 0,
            output=(result.stdout + result.stderr)[-4000:],
            artifact=output if result.returncode == 0 else None,
            diagnostics=diagnostics,
            reason="" if result.returncode == 0 else f"compiler exited {result.returncode}",
        )

    def verify(self, bundle: KernelBundle) -> ExtractionCheck:
        check = ExtractionCheck()
        failures: list[str] = []

        if bundle.build is None:
            failures.append("no BuildRecipe: a SYCL bundle cannot be rebuilt without one")
        elif bundle.launch is not None:
            recorded_grf = bundle.launch.compiled_metadata.get("grf_mode")
            recipe_grf = _grf_from_flags(bundle.build.flags)
            if recorded_grf and recipe_grf and str(recorded_grf) != str(recipe_grf):
                failures.append(
                    f"GRF mode differs: workload ran {recorded_grf}, recipe builds "
                    f"{recipe_grf}; these do not perform the same"
                )
            # AOT vs JIT is not a detail: they perform differently and rebuild
            # differently, so a mismatch means this is not the same kernel (§11.4).
            recorded_aot = bundle.launch.compiled_metadata.get("aot")
            if recorded_aot is not None:
                recipe_aot = bundle.build.aot_target is not None
                if bool(recorded_aot) != recipe_aot:
                    failures.append(
                        f"AOT/JIT mismatch: workload aot={recorded_aot}, recipe aot={recipe_aot}"
                    )

        if not bundle.closure:
            failures.append("empty closure: no translation unit or headers recorded")

        check.failures = failures
        check.verified = not failures
        return check

    # -- action space ------------------------------------------------------

    def option_axes(self) -> list[CompilerAxis]:
        """Cheap, deterministic wins that need no code change and no agent (§11.7)."""
        return [
            CompilerAxis(
                name="grf_mode",
                values=["default", "large", "auto"],
                flag_template="-Xs '-internal_options -cl-intel-<value>-grf'",
                description=(
                    "Large vs default register file — often the single largest lever "
                    "on Xe, and it interacts with occupancy"
                ),
            ),
            CompilerAxis(
                name="sub_group_size",
                values=[8, 16, 32],
                flag_template="-DSG_SIZE=<value>",
                description="Required sub-group size; the right value is generation-dependent",
            ),
            CompilerAxis(
                name="aot_target",
                values=sorted(set(DEVICE_TO_AOT_TARGET.values())),
                flag_template="-fsycl-targets=spir64_gen -Xs '-device <value>'",
                description="AOT device target versus SPIR-V JIT",
            ),
            CompilerAxis(
                name="opt_level",
                values=["-O2", "-O3"],
                flag_template="<value>",
                description="Optimization level",
            ),
            CompilerAxis(
                name="fp_contract",
                values=["fast", "on", "off"],
                flag_template="-ffp-contract=<value>",
                description="Floating-point contract",
                # fast-math changes numerics, so it is gated by the correctness rules
                # rather than being a free win (§11.7, §19).
                changes_numerics=True,
            ),
        ]


def aot_target_for_device(device_name: str) -> str | None:
    """Map a reported device name onto a SYCL AOT target."""
    lowered = (device_name or "").lower()
    for hint, target in DEVICE_TO_AOT_TARGET.items():
        if hint in lowered:
            return target
    return None


def _first_available_compiler() -> str | None:
    for candidate in ("icpx", "dpcpp", "clang++"):
        if shutil.which(candidate):
            return candidate
    return None


def _aot_target_from_flags(arguments: list[str]) -> str | None:
    for i, arg in enumerate(arguments):
        if arg == "-device" and i + 1 < len(arguments):
            return arguments[i + 1]
        if arg.startswith("-fsycl-targets="):
            return arg.split("=", 1)[1]
    return None


def _grf_from_flags(flags: list[str]) -> str | None:
    for flag in flags:
        if "grf" in flag.lower():
            if "large" in flag.lower():
                return "large"
            if "auto" in flag.lower():
                return "auto"
            return "default"
    return None


def _parse_device_diagnostics(stderr: str) -> dict[str, object]:
    """Capture the compiler's device-side diagnostics — the SYCL counterpart to
    Triton's compiled-kernel metadata, feeding the same verification (§11.4)."""
    diagnostics: dict[str, object] = {}
    spills = re.search(r"(\d+)\s+bytes? of spill", stderr, re.I)
    if spills:
        diagnostics["spill_bytes"] = int(spills.group(1))
    regs = re.search(r"GRF\s+usage[:\s]+(\d+)", stderr, re.I)
    if regs:
        diagnostics["grf_usage"] = int(regs.group(1))
    slm = re.search(r"SLM\s+(?:usage|size)[:\s]+(\d+)", stderr, re.I)
    if slm:
        diagnostics["slm_bytes"] = int(slm.group(1))
    return diagnostics


def _parse_make_deps(text: str) -> list[Path]:
    """Parse `-M` output (a make rule) into the header list."""
    body = text.split(":", 1)[1] if ":" in text else text
    body = body.replace("\\\n", " ")
    paths: list[Path] = []
    for token in body.split():
        token = token.strip()
        if not token or token.endswith(".o"):
            continue
        path = Path(token)
        if path.is_file():
            paths.append(path)
    return paths


def _scan_local_includes(tu: Path) -> list[Path]:
    """Textual fallback when no compiler is available: local quoted includes only."""
    try:
        text = tu.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[Path] = []
    for match in re.finditer(r'#\s*include\s+"([^"]+)"', text):
        candidate = (tu.parent / match.group(1)).resolve()
        if candidate.is_file():
            found.append(candidate)
    return found


def _split_template_args(blob: str) -> list[str]:
    """Split a template argument list at depth zero."""
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for char in blob:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return [a for a in args if a]


def _identifier_stem(demangled: str) -> str | None:
    """Best-effort kernel identifier from a demangled name, for file matching."""
    cleaned = re.sub(r"<.*>", "", demangled)
    parts = [p for p in re.split(r"[:\s(]+", cleaned) if p]
    return parts[-1] if parts else None


# Generic wrappers that appear in almost every ATen XPU kernel name. They identify the
# *launch mechanism*, not the kernel, so matching a source file on them is worthless.
_GENERIC_WRAPPERS = frozenset(
    {
        "VectorizedElementwiseKernel",
        "ElementwiseKernel",
        "UnrolledElementwiseKernel",
        "RoundedRangeKernel",
        "Array",
        "TrivialOffsetCalculator",
        "OffsetCalculator",
        "detail",
        "native",
        "xpu",
        "at",
        "sycl",
    }
)


def identifying_symbols(demangled: str) -> list[str]:
    """Distinctive identifiers in a demangled SYCL kernel name, most specific first.

    A real ATen XPU kernel name looks like:

        at::native::xpu::VectorizedElementwiseKernel<
            4, at::native::xpu::GeluErfFunctor<float>, ...>

    The outer template is a generic launch wrapper shared by hundreds of kernels; the
    thing that actually names the computation — and the file it lives in — is the
    functor *inside* the template arguments. An earlier version of this resolver
    stripped the template arguments before matching, which threw away the only useful
    token and left every elementwise kernel looking identical.
    """
    # An Itanium-mangled name carries no separators, so the generic identifier scan
    # below sees it as one token. Unitrace reports demangled names, but a raw Level Zero
    # trace does not.
    #
    # Itanium mangling is length-prefixed — `_ZTS13IgammaFunctor` says "13 characters" —
    # so the identifier can be read off exactly. An earlier version instead stripped a
    # trailing `I...E` template suffix with a greedy pattern, which recovered
    # `GeluErfFunctor` and silently reduced `IgammaFunctor` to an empty string because
    # the identifier itself contains the `I` the pattern keyed on. Counting characters is
    # both simpler and correct.
    candidates = [demangled]
    mangled = re.match(r"_ZTS(\d+)(.*)$", demangled)
    if mangled:
        length = int(mangled.group(1))
        body = mangled.group(2)
        if 0 < length <= len(body):
            candidates.append(body[:length])

    identifiers: list[str] = []
    for candidate in candidates:
        identifiers.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate))

    seen: list[str] = []
    for name in identifiers:
        if name in _GENERIC_WRAPPERS or len(name) < 4:
            continue
        if name not in seen:
            seen.append(name)

    # Functors and kernels first: those are what source files are named after.
    def rank(name: str) -> tuple[int, int]:
        specific = 0 if ("Functor" in name or "Kernel" in name) else 1
        return (specific, -len(name))

    return sorted(seen, key=rank)


def index_source_tree(root: Path, patterns: tuple[str, ...] = ("*.cpp", "*.h")) -> dict[str, Path]:
    """Map C++ symbol -> defining file across a SYCL source tree.

    This is the practical alternative to `compile_commands.json`. Intel's kernel
    surface — torch-xpu-ops, vllm-xpu-kernels, sgl-kernel-xpu — ships as open source,
    but a *wheel* contains only the compiled shared object, so on a normal install
    there is no build database to consult. Indexing the checked-out sources recovers
    the kernel-to-file mapping without requiring the user to build anything.
    """
    index: dict[str, Path] = {}
    root = Path(root)
    for pattern in patterns:
        for path in root.rglob(pattern):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in re.finditer(r"\b(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
                index.setdefault(match.group(1), path)
            for match in re.finditer(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*_kernel)\s*\(", text):
                index.setdefault(match.group(1), path)
    return index
