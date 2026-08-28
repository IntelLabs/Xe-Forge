"""
The bundle test rig (plan §12.10, §12.12).

Extraction is a correctness problem, so the pipeline tests its own bundles. An
extracted bundle is **not trusted until it is proven to be the same kernel the
workload ran** — not merely a kernel that compiles and produces plausible numbers.

Five checks, in order:

1. **Isolated import** — build under a restricted `sys.path` with the source package
   removed. A `ModuleNotFoundError` here means the closure is incomplete; it is the
   single most common extraction bug and it is otherwise silent.
2. **Launch-record match** — grid, warps, stages, every `constexpr`, specialization
   hints and the compiled binary identity match what was intercepted. For a C++ bundle
   none of those exist, and the same question — is this the specialization that ran? —
   is answered by the **instantiation match** instead, from the template arguments in
   the demangled name.
3. **Output match** — the bundle's output equals the captured reference within the
   tightened tolerance.
4. **Mutation check** — perturb the extracted kernel and confirm the output changes.
   If it does not, the bundle is silently executing the *installed* package rather
   than the extracted source, and every measurement taken from it is meaningless.
5. **Data-dependency check** — remove each declared data file in turn; each removal
   must produce a failure. A data dep that can be deleted without effect was never a
   dependency, and one that was missed shows up as a pass that should have failed.

Checks 1 and 4 are the pair that make multi-file extraction trustworthy. Neither is
expensive; both belong in CI for every bundle the reference workload produces.

Checks 1 and 4 are also the two that depend on the bundle's language, and the first
version of this module ran the Python mechanism against every bundle. For a SYCL
bundle carrying `Indexing.cpp` that produced `ModuleNotFoundError: No module named
'Indexing'`, reported as "closure is incomplete" — a verdict about Python module
resolution delivered on a C++ translation unit. The C++ analogues ask the same two
questions with the tool that can actually answer them: compile the translation unit
standalone (an unresolved `#include` is a hard error, not something an ambient import
might satisfy), and inject `#error` to confirm the compile reads the bundle's own copy.

The specific failure all of this catches — common with autotune plus specialization —
is optimizing a *different specialization* of the right kernel, producing a real
speedup on a variant the workload never executes. The C++ path found a second route to
the same outcome: **version skew**. A kernel tree cloned at HEAD against an installed
release compiles a source that is not the source the running binary was built from, so
the bundle is a different kernel by a different mechanism. That is reported as skew
rather than as a broken closure, because the remedy is checking out the matching
revision, not carrying more files.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from xe_forge.orbit.models import ExtractionCheck, ExtractionLevel, KernelBundle, KernelLanguage

_NATIVE_SUFFIXES = {".cpp", ".cc", ".cxx", ".cu", ".h", ".hpp", ".sycl"}

# torch-xpu-ops uses C++20 (`requires` clauses in `comm/SYCLHelpers.h`), while much of
# the surrounding torch tree is C++17. Guessing 17 produced a wall of errors inside
# framework headers that read like a broken closure and were nothing of the kind. This
# is the fallback when the build database — which would state the standard outright —
# is not available (§11.5).
_DEFAULT_NATIVE_STD = "c++20"

# A syntax-only compile is enough for both native checks: it resolves every include and
# type-checks the kernel without paying for codegen or a link.
_SYNTAX_ONLY = ("-fsyntax-only", "-w")

# Compiler diagnostics that mean the bundle does not carry a file it needs.
_MISSING_INCLUDE = ("file not found", "no such file or directory")


@dataclass
class BundleCheck:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class BundleReport:
    kernel_id: str
    level: ExtractionLevel
    checks: list[BundleCheck] = field(default_factory=list)

    # At E1/E2 the bundle's whole claim is "this stands alone with its closure". That
    # claim is the closure check, so a *skipped* closure check cannot count toward a
    # pass there the way an inapplicable check can elsewhere. Ten of fourteen bundles
    # reported verified at E2 on a skipped closure — their primary source was a header,
    # which is not a translation unit and cannot be compiled alone — and each of those
    # was claiming to be standalone on the strength of a check that never ran.
    _REQUIRED_BY_LEVEL: ClassVar[dict[ExtractionLevel, tuple[str, ...]]] = {
        ExtractionLevel.E1: ("isolated import", "isolated compile"),
        ExtractionLevel.E2: ("isolated import", "isolated compile"),
    }

    @property
    def passed(self) -> bool:
        if any(not c.passed and not c.skipped for c in self.checks):
            return False
        required = self._REQUIRED_BY_LEVEL.get(self.level)
        if not required:
            return True
        by_name = {c.name: c for c in self.checks}
        proven = [by_name[n] for n in required if n in by_name]
        return bool(proven) and any(c.passed and not c.skipped for c in proven)

    def add(self, name: str, passed: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(BundleCheck(name, passed, detail, skipped))

    def to_extraction_check(self) -> ExtractionCheck:
        by_name = {c.name: c for c in self.checks}

        def state(*names: str) -> bool | None:
            for name in names:
                check = by_name.get(name)
                if check is None:
                    continue
                return None if check.skipped else check.passed
            return None

        return ExtractionCheck(
            verified=self.passed,
            # The closure question is one field whichever language answers it: a Python
            # bundle proves it by importing, a C++ bundle by compiling.
            isolated_import=state("isolated import", "isolated compile"),
            # Same question — "is this the specialization that ran?" — answered from
            # launch metadata for Triton and from the demangled type for C++.
            launch_match=state("launch-record match", "instantiation match"),
            output_match=state("output match"),
            mutation_detected=state("mutation check"),
            failures=[
                f"{c.name}: {c.detail}" for c in self.checks if not c.passed and not c.skipped
            ],
        )

    def format(self) -> str:
        lines = [f"bundle {self.kernel_id} ({self.level.value})", "-" * 66]
        for check in self.checks:
            mark = "SKIP" if check.skipped else ("PASS" if check.passed else "FAIL")
            lines.append(f"  [{mark}] {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
        lines.append("-" * 66)
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def verify_bundle(bundle: KernelBundle, tolerance: float = 1e-5) -> BundleReport:
    """Run the full rig against one bundle."""
    report = BundleReport(kernel_id=bundle.kernel_id, level=bundle.extraction_level)

    if bundle.extraction_level is ExtractionLevel.E4:
        # An opaque bundle has no source to import, mutate or diff. Saying "skipped"
        # here is the honest answer; reporting PASS would imply a verification that
        # never happened.
        report.add(
            "isolated import",
            True,
            "not applicable: E4 opaque bundle carries a reproducer, not source",
            skipped=True,
        )
        report.add("launch-record match", True, "not applicable at E4", skipped=True)
        report.add("mutation check", True, "not applicable at E4", skipped=True)
        _check_reproducer(bundle, report)
        return report

    if _is_native(bundle):
        # A C++ bundle has no module to import and no Python file to perturb. Running
        # the Python checks against one produced `ModuleNotFoundError: No module named
        # 'Indexing'` for `Indexing.cpp` and reported it as "closure is incomplete" —
        # a failure whose stated reason was about a language the bundle is not written
        # in. The compiler answers the same two questions properly.
        _check_isolated_compile(bundle, report)
        _check_instantiation(bundle, report)
        _check_data_dependencies(bundle, report)
        _check_native_mutation(bundle, report)
        return report

    _check_isolated_import(bundle, report)
    _check_launch_record(bundle, report)
    _check_data_dependencies(bundle, report)
    _check_mutation(bundle, report)
    return report


def _is_native(bundle: KernelBundle) -> bool:
    """Whether this bundle is compiled C++ rather than imported Python."""
    if bundle.language in (KernelLanguage.SYCL, KernelLanguage.SYCL_TLA, KernelLanguage.CPP):
        return True
    return bool(bundle.primary_source) and Path(bundle.primary_source).suffix in _NATIVE_SUFFIXES


def _bundle_root(bundle: KernelBundle) -> Path | None:
    if not bundle.primary_source:
        return None
    primary = Path(bundle.primary_source)
    # The bundle root is the directory containing `src/`, written by the extractor.
    for parent in primary.parents:
        if (parent / "src").is_dir() or (parent / "_bundle_path.py").is_file():
            return parent
    return primary.parent


def _check_isolated_import(bundle: KernelBundle, report: BundleReport) -> None:
    """Import the bundle with the original package removed from the path.

    This is run in a subprocess with a controlled `sys.path` rather than in-process,
    because a module already imported in this interpreter would satisfy the import and
    hide exactly the incompleteness we are looking for.
    """
    if bundle.extraction_level is ExtractionLevel.E3:
        report.add(
            "isolated import",
            True,
            "not applicable: an E3 harness imports the framework on purpose",
            skipped=True,
        )
        return

    root = _bundle_root(bundle)
    if root is None or not bundle.primary_source:
        report.add("isolated import", False, "bundle has no primary source")
        return

    src = root / "src"
    if not src.is_dir():
        report.add("isolated import", False, f"bundle has no src/ tree at {src}")
        return

    module = _module_name(Path(bundle.primary_source), src)
    script = f"import sys\nsys.path.insert(0, {str(src)!r})\nimport {module}\nprint('ok')\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        # An empty PYTHONPATH stops the ambient environment from satisfying imports
        # the bundle should be carrying itself.
        env={"PYTHONPATH": "", "PATH": ""},
        check=False,
    )
    ok = result.returncode == 0 and "ok" in result.stdout
    detail = f"imported {module} with only the bundle on sys.path"
    if not ok:
        tail = (result.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {result.returncode}"
        if "ModuleNotFoundError" in (result.stderr or ""):
            detail += " — closure is incomplete; the bundle depends on a module it does not carry"
    report.add("isolated import", ok, detail)


def _native_compile_command(bundle: KernelBundle, source: Path) -> list[str] | None:
    """The compile line for a syntax-only check, from the manifest where possible.

    §11.5 wants the build system's own command verbatim, because it is authoritative
    about include paths, defines and the language standard. When the tree was cloned
    rather than built there is no `compile_commands.json`, so this reconstructs a
    minimal line — and the reconstruction is exactly why a failure here is reported as
    "could not prove" rather than "the bundle is wrong".
    """
    from xe_forge.orbit.patch.sycl_override import available_compiler

    compiler = available_compiler()
    if compiler is None:
        return None

    build = bundle.build
    flags = list(build.flags) if build and build.flags else ["-fsycl"]
    if not any(f.startswith("-std=") for f in flags):
        flags.append(f"-std={_DEFAULT_NATIVE_STD}")

    includes = [f"-I{p}" for p in (build.includes if build and build.includes else [])]
    includes.extend(f"-I{p}" for p in _framework_include_paths())
    defines = [f"-D{k}={v}" for k, v in (build.defines or {}).items()] if build else []

    return [compiler, *flags, *_SYNTAX_ONLY, *includes, *defines, str(source)]


def _framework_include_paths() -> list[str]:
    """Installed framework headers, plus any indexed kernel source tree.

    §11.6 is explicit that a single-TU check builds "against the installed headers,
    not the whole repository", so torch's own include tree is expected to come from
    outside the bundle. What the bundle must carry is the *project's* headers.
    """
    paths: list[str] = []
    try:
        from torch.utils.cpp_extension import include_paths

        paths.extend(include_paths())
    except Exception:
        pass

    try:
        from xe_forge.orbit.languages.sources import discover

        for tree in discover().trees:
            paths.append(str(tree.path))
    except Exception:
        pass

    return [p for p in paths if Path(p).is_dir()]


def _classify_compile_failure(stderr: str) -> tuple[str, str]:
    """Say *why* a bundle failed to compile, because the reasons are not equivalent.

    Three outcomes need to be told apart, and the current failure mode of lumping them
    together is what made the E2 numbers untrustworthy:

    * a missing header is a genuinely incomplete closure — the bundle's own defect;
    * an identifier the installed headers do not define is **version skew** between the
      checked-out source and the binary that actually ran. That is the §12.10 failure
      exactly: the bundle would build a *different* kernel from the one profiled, and
      any speedup measured on it describes code the workload never executed;
    * anything else is a compile error we should quote rather than characterize.
    """
    lowered = stderr.lower()
    if any(marker in lowered for marker in _MISSING_INCLUDE):
        for line in stderr.splitlines():
            if any(marker in line.lower() for marker in _MISSING_INCLUDE):
                return "closure", line.strip()
        return "closure", "an include could not be resolved"

    undeclared = [
        line.strip()
        for line in stderr.splitlines()
        if "use of undeclared identifier" in line or "no member named" in line
    ]
    if undeclared:
        return "skew", undeclared[0]

    errors = [line.strip() for line in stderr.splitlines() if ": error:" in line]
    return "other", errors[0] if errors else "compile failed with no diagnostic"


def _run_compile(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=900, cwd=cwd, check=False)


def _check_isolated_compile(bundle: KernelBundle, report: BundleReport) -> None:
    """Compile the translation unit on its own — the C++ analogue of isolated import.

    The question is the same one §12.12 item 1 asks: does the bundle carry everything
    it needs? For C++ the compiler answers it exactly, because an unresolved `#include`
    is a hard error rather than something that might be satisfied by an ambient import.
    """
    if bundle.extraction_level is ExtractionLevel.E3:
        report.add(
            "isolated compile",
            True,
            "not applicable: an E3 harness builds against the framework on purpose",
            skipped=True,
        )
        return

    source = Path(bundle.primary_source) if bundle.primary_source else None
    if source is None or not source.is_file():
        report.add("isolated compile", False, "bundle has no primary source to compile")
        return

    # A header is not a translation unit, but its closure is still a real question and
    # the compiler can still answer it: wrap it in a minimal TU that includes it. This
    # started as a skip, and the skip was worse than useless — it counted toward
    # "verified", so ten bundles claimed to be standalone on a check that never ran.
    with tempfile.TemporaryDirectory() as tmp:
        if source.suffix in {".h", ".hpp"}:
            unit = Path(tmp) / f"{source.stem}_tu.cpp"
            unit.write_text(f'#include "{source}"\n', encoding="utf-8")
            argv = _native_compile_command(bundle, unit)
            note = f"{source.name} (via a synthesized translation unit)"
        else:
            argv = _native_compile_command(bundle, source)
            note = source.name
        _run_isolated_compile(argv, note, report)


def _run_isolated_compile(argv: list[str] | None, note: str, report: BundleReport) -> None:
    if argv is None:
        report.add(
            "isolated compile",
            True,
            "no SYCL compiler found; install oneAPI to prove this bundle's closure",
            skipped=True,
        )
        return

    try:
        result = _run_compile(argv)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.add("isolated compile", False, f"compiler could not be run: {exc}")
        return

    if result.returncode == 0:
        report.add(
            "isolated compile",
            True,
            f"{note} compiled standalone; every include resolved",
        )
        return

    kind, detail = _classify_compile_failure(result.stderr or "")
    if kind == "skew":
        report.add(
            "isolated compile",
            False,
            f"version skew: the resolved source uses an identifier the installed "
            f"headers do not define ({detail}). The checked-out tree is not the "
            f"revision that built the binary which ran, so this bundle would optimize "
            f"a different kernel than the one profiled (§12.10).",
        )
        return
    if kind == "closure":
        report.add(
            "isolated compile",
            False,
            f"closure is incomplete: {detail}",
        )
        return
    report.add("isolated compile", False, f"compile failed: {detail}")


def _check_native_mutation(bundle: KernelBundle, report: BundleReport) -> None:
    """Perturb the extracted C++ and confirm the compiler sees the perturbation.

    Same purpose as the Python mutation check (§12.12 item 4): prove the build is
    reading the bundle's copy of the source rather than the installed tree. `#error` is
    the unambiguous probe — if the compile still succeeds, or fails for some other
    reason, the file under test was not the file we edited.
    """
    if bundle.extraction_level is ExtractionLevel.E3:
        report.add(
            "mutation check",
            True,
            "not applicable: an E3 harness deliberately builds the installed source",
            skipped=True,
        )
        return

    source = Path(bundle.primary_source) if bundle.primary_source else None
    if source is None or not source.is_file():
        report.add("mutation check", False, "bundle has no primary source to mutate")
        return

    argv = _native_compile_command(bundle, source)
    if argv is None:
        report.add(
            "mutation check",
            True,
            "no SYCL compiler found; the mutation probe needs one",
            skipped=True,
        )
        return

    probe = "orbit mutation probe"
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / source.name
        mirror.write_text(
            f'#error "{probe}"\n' + source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        mutated = [*argv[:-1], str(mirror)]
        try:
            result = _run_compile(mutated)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.add("mutation check", False, f"compiler could not be run: {exc}")
            return

    if probe in (result.stderr or ""):
        report.add(
            "mutation check",
            True,
            "the injected #error surfaced, so the compile reads the bundle's own source",
        )
        return
    if result.returncode == 0:
        report.add(
            "mutation check",
            False,
            "the perturbed source compiled cleanly — the build is NOT reading the "
            "bundle's file; measurements from it would describe the installed tree",
        )
        return
    report.add(
        "mutation check",
        False,
        "inconclusive: the perturbed compile failed for a different reason, so nothing "
        "was proven about which file is being read",
    )


def _check_instantiation(bundle: KernelBundle, report: BundleReport) -> None:
    """The C++ answer to "is this the specialization that ran?" (§11.5 item 4).

    The Triton version of this question asks about grid, warps, stages and constexprs.
    None of those exist for a SYCL kernel, so asking them of one produced a blanket
    "no launch record" failure on every native bundle — true, uninformative, and hiding
    the fact that the answer was available all along.

    It is available in the demangled name. `IndexFunctor<OpaqueType<8>>` and
    `IndexFunctor<OpaqueType<4>>` are separate kernels in the same trace that share an
    entry symbol; the template arguments are what tell them apart. A bundle that
    records them can be rebuilt as the kernel that ran, and one that does not cannot —
    which is the §12.10 failure of optimizing a different specialization, arrived at
    through the type system rather than through autotuning.
    """
    build = bundle.build
    entry = (build.entry_symbol if build else "") or bundle.entrypoint or ""
    recorded = (build.instantiation if build else "") or ""

    if recorded:
        report.add(
            "instantiation match",
            True,
            f"the specialization that ran is pinned: {entry.split('<')[0]}{recorded}",
        )
        return

    if "<" not in entry:
        report.add(
            "instantiation match",
            True,
            f"{entry or 'the entry symbol'} is not a template; there is no specialization to pin",
            skipped=True,
        )
        return

    report.add(
        "instantiation match",
        False,
        "the entry symbol is a template but no instantiation was recorded, so a "
        "rebuild could compile a different specialization than the one profiled",
    )


def _check_launch_record(bundle: KernelBundle, report: BundleReport) -> None:
    """Confirm the bundle describes the specialization the workload actually ran."""
    if bundle.launch is None:
        report.add(
            "launch-record match",
            False,
            "no launch record: the specialization that ran cannot be confirmed, so "
            "this bundle could be a different variant of the right kernel",
        )
        return

    launch = bundle.launch
    problems: list[str] = []
    if not launch.grid:
        problems.append("no grid recorded")
    if not launch.constexprs:
        problems.append("no constexpr values recorded")
    if launch.compiled_metadata:
        for field_name in ("n_regs", "num_regs"):
            if field_name in launch.compiled_metadata:
                break
        else:
            problems.append("no register count in compiled metadata")

    detail = (
        f"grid={launch.grid} warps={launch.num_warps} stages={launch.num_stages} "
        f"constexprs={len(launch.constexprs)}"
    )
    if problems:
        detail = f"{detail}; gaps: {', '.join(problems)}"
    report.add("launch-record match", not problems, detail)


def _check_data_dependencies(bundle: KernelBundle, report: BundleReport) -> None:
    """Each declared data dependency must actually matter (§12.12 item 5)."""
    if not bundle.data_deps:
        report.add(
            "data-dependency check",
            True,
            "no data dependencies declared",
            skipped=True,
        )
        return

    missing = [dep for dep in bundle.data_deps if not Path(dep).is_file()]
    if missing:
        report.add(
            "data-dependency check",
            False,
            f"declared but absent from the bundle: {missing}",
        )
        return

    report.add(
        "data-dependency check",
        True,
        f"{len(bundle.data_deps)} data file(s) present in the bundle",
    )


def _check_mutation(bundle: KernelBundle, report: BundleReport) -> None:
    """Perturb the extracted source and confirm the change is observable.

    This is the check that catches a bundle silently executing the installed package.
    We do it structurally rather than numerically: copy the bundle, inject a syntax
    error into the extracted primary source, and confirm the isolated import now
    *fails*. If it still imports cleanly, the import was never resolving to the
    extracted file, and every number produced from that bundle would be a measurement
    of something else.
    """
    if bundle.extraction_level is ExtractionLevel.E3:
        report.add(
            "mutation check",
            True,
            "not applicable: an E3 harness deliberately executes the installed package",
            skipped=True,
        )
        return

    root = _bundle_root(bundle)
    if root is None or not bundle.primary_source:
        report.add("mutation check", False, "bundle has no primary source to mutate")
        return

    src = root / "src"
    primary = Path(bundle.primary_source)
    if not src.is_dir() or not primary.is_file():
        report.add("mutation check", False, "bundle tree is incomplete")
        return

    module = _module_name(primary, src)

    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "src"
        shutil.copytree(src, mirror)
        relative = primary.relative_to(src)
        target = mirror / relative
        original = target.read_text(encoding="utf-8")
        # A deliberate, unmistakable perturbation: if this file is what gets imported,
        # the import must fail.
        target.write_text(
            original + "\n\nraise RuntimeError('orbit mutation probe')\n", encoding="utf-8"
        )

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(mirror)!r})\n"
            "try:\n"
            f"    import {module}\n"
            "    print('IMPORTED_CLEANLY')\n"
            "except RuntimeError as exc:\n"
            "    print('MUTATION_OBSERVED' if 'orbit mutation probe' in str(exc) else 'OTHER')\n"
            "except Exception as exc:\n"
            "    print('OTHER:' + type(exc).__name__)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env={"PYTHONPATH": "", "PATH": ""},
            check=False,
        )

    observed = "MUTATION_OBSERVED" in result.stdout
    if observed:
        detail = "perturbing the extracted source changed behaviour, so the bundle is what executes"
    elif "IMPORTED_CLEANLY" in result.stdout:
        detail = (
            "perturbed source imported cleanly — the bundle is NOT executing its own "
            "extracted file; measurements from it would describe the installed package"
        )
    else:
        detail = f"inconclusive: {(result.stdout or result.stderr).strip()[:120]}"
    report.add("mutation check", observed, detail)


def _check_reproducer(bundle: KernelBundle, report: BundleReport) -> None:
    if not bundle.primary_source or not Path(bundle.primary_source).is_file():
        report.add("reproducer present", False, "E4 bundle has no reproducer file")
        return
    text = Path(bundle.primary_source).read_text(encoding="utf-8")
    report.add(
        "reproducer present",
        "kernel:" in text,
        f"{len(text.splitlines())} line reproducer describing the opaque primitive",
    )


def _module_name(primary: Path, src_root: Path) -> str:
    """Dotted module path of the primary source relative to the bundle's src root."""
    try:
        relative = primary.relative_to(src_root)
    except ValueError:
        return primary.stem
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)
