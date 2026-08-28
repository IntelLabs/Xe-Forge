"""
Triton language backend (plan §11.3, §12.6).

Closure for Triton is an AST walk: starting from the intercepted launch record, every
call that lands on a `@triton.jit` function is a device helper and joins the work
list — transitively, across modules, including helpers reached through a re-export or
an alias. Module-level constants used as `constexpr` values come along, as do the
autotune configuration list and any heuristics callable that closes over module state.

Two rules from §12.6 are enforced here because getting them wrong is silent:

* **Keep the package structure, add a path shim.** Flattening into one file breaks
  relative imports and re-export chains, and destroys the mapping back to original
  source that patch-back needs. A bundle is a small tree plus a manifest.
* **A partially resolved closure is worse than an honest in-situ harness**, because it
  looks standalone and is not. Any unresolved step downgrades to E3 and records which
  step failed.

Triton itself is not required to compute a closure: the AST walk runs on source text,
which is what lets closure resolution be tested on CPU-only CI with no GPU stack.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from xe_forge.orbit.languages.base import (
    BaseLanguageBackend,
    BuildResult,
    CompilerAxis,
    CostProfile,
)
from xe_forge.orbit.models import (
    ExtractionCheck,
    KernelBundle,
    KernelLanguage,
    ResolutionMethod,
    SourceLocation,
)

# Decorators that mark a function as device code or as carrying launch configuration.
_JIT_DECORATORS = {"jit", "triton.jit"}
_AUTOTUNE_DECORATORS = {"autotune", "triton.autotune"}
_HEURISTIC_DECORATORS = {"heuristics", "triton.heuristics"}


@dataclass
class ClosureResult:
    """What the AST walk could and could not resolve."""

    entry_file: Path | None = None
    files: list[Path] = field(default_factory=list)
    helpers: list[str] = field(default_factory=list)
    constants: dict[str, object] = field(default_factory=dict)
    autotune_configs: list[str] = field(default_factory=list)
    heuristics: list[str] = field(default_factory=list)
    data_deps: list[Path] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unresolved


class TritonBackend(BaseLanguageBackend):
    """Identity, closure and verification for Triton kernels."""

    name = "triton"
    language = KernelLanguage.TRITON
    # A JIT compile, not a rebuild: seconds, not minutes.
    cost_profile = CostProfile(
        build_seconds=3.0,
        iteration_seconds=2.0,
        notes="JIT compile per configuration; cheap enough to iterate freely",
    )

    def identify(self, kernel_name: str) -> float:
        lowered = kernel_name.lower()
        if lowered.startswith("triton_"):
            return 0.95
        if "triton" in lowered or lowered.endswith("_kernel"):
            return 0.7
        return 0.0

    def resolve_source(self, kernel_name: str, launch=None) -> SourceLocation:
        """Prefer the intercepted launch record over any static guess (§12.4)."""
        if launch is not None and getattr(launch, "source_file", None):
            return SourceLocation(
                file=launch.source_file,
                line=getattr(launch, "source_line", None),
                symbol=getattr(launch, "fq_name", kernel_name),
                confidence=0.95,
            )

        module_path, _, symbol = kernel_name.partition(":")
        if symbol and module_path:
            found = _module_file(module_path)
            if found is not None:
                return SourceLocation(
                    file=str(found),
                    symbol=symbol,
                    method=ResolutionMethod.NAME_MATCH,
                    confidence=0.7,
                )

        # A Triton kernel in an *installed* framework has no launch record and no cache
        # entry until it has been JIT-compiled in this process, but its definition is
        # ordinary Python sitting in site-packages. Searching the framework tree for the
        # `@triton.jit` function by name resolves it exactly and deterministically — no
        # agent needed (§3), and these are the most patchable kernels we have, so failing
        # to locate them costs the whole in-place path.
        found = _find_jit_definition(kernel_name)
        if found is not None:
            path, line = found
            return SourceLocation(
                file=str(path),
                line=line,
                symbol=kernel_name,
                method=ResolutionMethod.SYMBOL_INDEX,
                candidates=[],
            )

        return SourceLocation(
            symbol=kernel_name,
            method=ResolutionMethod.UNRESOLVED,
            confidence=0.0,
            candidates=[],
        )

    def closure(self, source: SourceLocation) -> list[Path]:
        result = self.resolve_closure(source)
        return result.files

    def resolve_closure(self, source: SourceLocation, max_depth: int = 12) -> ClosureResult:
        """Walk the `@triton.jit` call graph transitively across modules (§12.6)."""
        out = ClosureResult()
        if not source.file:
            out.unresolved.append("no source file resolved for entrypoint")
            return out

        entry = Path(source.file)
        if not entry.is_file():
            out.unresolved.append(f"source file does not exist: {entry}")
            return out

        out.entry_file = entry
        # Everything under the entrypoint's package root is a closure member; anything
        # outside it (torch, triton, json) is an environment dependency the bundle
        # declares rather than carries.
        package_root = _package_root(entry)

        seen_files: set[Path] = set()
        work: list[tuple[Path, str | None]] = [(entry, source.symbol)]
        visits = 0
        max_visits = max_depth * 16

        while work and visits < max_visits:
            visits += 1
            current_file, symbol = work.pop()
            resolved_file = current_file.resolve()
            if resolved_file in seen_files:
                continue
            seen_files.add(resolved_file)

            try:
                tree = ast.parse(current_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                out.unresolved.append(f"cannot parse {current_file}: {exc}")
                continue

            if current_file not in out.files:
                out.files.append(current_file)

            imports = _import_map(tree)
            out.constants.update(_module_constants(tree))

            # Follow *every* in-package import, not only the ones we can see being
            # called. A module cannot be imported at all unless its module-level
            # imports resolve, so a closure built from the call graph alone produces a
            # bundle that looks complete and fails on import — which is exactly what
            # the isolated-import check catches (§12.12).
            for name, (module_name, level) in imports.items():
                target = _resolve_import(module_name, level, current_file)
                if target is None:
                    if level > 0:
                        out.unresolved.append(
                            f"cannot locate relative import {'.' * level}{module_name or name!r} "
                            f"in {current_file.name}"
                        )
                    # An unresolvable absolute import is an external dependency
                    # (torch, triton, json); those are declared, not carried.
                    continue
                if not _within(target, package_root):
                    continue
                if target.resolve() not in seen_files:
                    work.append((target, None))

            for func in _functions(tree):
                decorators = _decorator_names(func)
                is_jit = bool(decorators & _JIT_DECORATORS)
                if symbol is not None and func.name != symbol and not is_jit:
                    continue

                if decorators & _AUTOTUNE_DECORATORS:
                    out.autotune_configs.append(func.name)
                if decorators & _HEURISTIC_DECORATORS:
                    out.heuristics.append(func.name)

                for callee in _called_names(func):
                    if callee in out.helpers:
                        continue
                    if callee in imports or _defined_in(tree, callee):
                        out.helpers.append(callee)

            for dynamic in _dynamic_imports(tree):
                out.unresolved.append(
                    f"dynamic import in {current_file.name}: {dynamic} — closure cannot "
                    f"be proven complete, downgrade to E3"
                )

        if work:
            out.unresolved.append(
                f"closure exceeded {max_visits} module visits; {len(work)} left unexplored"
            )

        return out

    def build(self, bundle: KernelBundle) -> BuildResult:
        """Triton compiles at launch, so 'building' is importing the bundle."""
        if not bundle.primary_source:
            return BuildResult(ok=False, reason="bundle has no primary source")
        source = Path(bundle.primary_source)
        if not source.is_file():
            return BuildResult(ok=False, reason=f"primary source missing: {source}")
        try:
            ast.parse(source.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return BuildResult(ok=False, reason=f"bundle source does not parse: {exc}")
        return BuildResult(ok=True, artifact=source, output="triton bundle parses; JIT at launch")

    def verify(self, bundle: KernelBundle) -> ExtractionCheck:
        """Check the bundle is the kernel that ran, not merely a kernel (§12.10)."""
        check = ExtractionCheck()
        failures: list[str] = []

        if bundle.launch is None:
            failures.append("no launch record to verify against")
        else:
            recorded = bundle.launch
            if not recorded.constexprs:
                failures.append(
                    "launch record has no constexpr values; the specialization the "
                    "workload actually ran cannot be confirmed"
                )
            if recorded.selected_autotune_config is None and bundle.data_deps:
                failures.append(
                    "bundle carries tuned-config data but no winning autotune config "
                    "was recorded; baseline and candidate would not be comparable"
                )
            check.launch_match = not failures

        if not bundle.closure:
            failures.append("empty closure: a multi-file kernel cannot have no closure")

        check.failures = failures
        check.verified = not failures
        return check

    def option_axes(self) -> list[CompilerAxis]:
        return [
            CompilerAxis(
                name="num_warps",
                values=[1, 2, 4, 8, 16, 32],
                description="Warps per block; interacts with occupancy and registers",
            ),
            CompilerAxis(
                name="num_stages",
                values=[1, 2, 3, 4, 5],
                description="Software pipelining depth",
            ),
        ]


def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _decorator_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in func.decorator_list:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        rendered = _render_attr(node)
        if rendered:
            names.add(rendered)
            names.add(rendered.rsplit(".", 1)[-1])
    return names


def _render_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _render_attr(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _called_names(func: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            rendered = _render_attr(node.func)
            if rendered:
                names.append(rendered.rsplit(".", 1)[-1])
    return names


def _import_map(tree: ast.Module) -> dict[str, tuple[str, int]]:
    """Map imported symbol -> (module, relative level), following aliases.

    The level matters: `from .helpers import f` and `from helpers import f` name
    different modules, and treating a relative import as absolute is how a closure
    walk over a real package silently fails to find helpers that are right next to the
    kernel. Real inference kernels are packaged, so relative imports are the norm.
    """
    mapping: dict[str, tuple[str, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            for alias in node.names:
                key = alias.asname or alias.name
                if not module and level > 0:
                    # `from . import sycl_op` — the imported *name* is the submodule,
                    # so the target is `sycl_op.py`, not the package's __init__.py.
                    mapping[key] = (alias.name, level)
                else:
                    mapping[key] = (module, level)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mapping[alias.asname or alias.name.split(".")[0]] = (alias.name, 0)
    return mapping


def _package_root(path: Path) -> Path:
    """Directory the entrypoint's package hangs off, found by walking past __init__.py.

    This bounds the closure: files under this root are the kernel's own package and are
    carried in the bundle; anything above it is an installed dependency.
    """
    current = path.parent
    while (current / "__init__.py").is_file() and current.parent != current:
        current = current.parent
    return current


def _within(path: Path, root: Path) -> bool:
    """True when `path` lives under `root` (i.e. is a closure member, not a dependency)."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_import(module: str, level: int, current_file: Path) -> Path | None:
    """Locate an imported module's source file, honouring relative imports.

    A relative import is resolved against the importing file's own directory, walking
    up one level for each extra leading dot. That is done on the filesystem rather
    than through `find_spec`, because the package may not be importable from wherever
    Orbit happens to be running — and requiring it to be would make closure resolution
    depend on the caller's working directory.
    """
    if level and level > 0:
        base = current_file.parent
        for _ in range(level - 1):
            base = base.parent
        if not module:
            # `from . import x` — the target is the package's __init__.
            candidate = base / "__init__.py"
            return candidate if candidate.is_file() else None
        parts = module.split(".")
        target = base.joinpath(*parts)
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    if not module:
        return None

    # Absolute import: try a sibling file first, so a package that is present on disk
    # but not installed still resolves, then fall back to the import system.
    sibling = current_file.parent / f"{module.split('.')[-1]}.py"
    if sibling.is_file():
        return sibling
    return _module_file(module)


def _module_constants(tree: ast.Module) -> dict[str, object]:
    """Module-level literal assignments, which `constexpr` values often reference."""
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
    return constants


def _defined_in(tree: ast.Module, name: str) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _dynamic_imports(tree: ast.Module) -> list[str]:
    """Find `importlib.import_module` / `__import__` calls that defeat static closure."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = _render_attr(node.func) or ""
            if rendered.endswith("import_module") or rendered == "__import__":
                found.append(rendered)
    return found


def _module_file(module_path: str) -> Path | None:
    """Locate a module's source file without importing it."""
    try:
        spec = importlib.util.find_spec(module_path)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin or spec.origin == "built-in":
        return None
    path = Path(spec.origin)
    return path if path.is_file() else None


# Framework trees worth searching for a `@triton.jit` definition, most specific first.
# An installed framework is the common case; the environment variable exists so a
# developer working from a checkout can point at it without reinstalling.
_FRAMEWORK_SOURCE_ENV = "ORBIT_FRAMEWORK_SOURCES"
_FRAMEWORK_PACKAGES = ("vllm", "sglang", "torch")


def _find_jit_definition(name: str) -> tuple[Path, int] | None:
    """Locate `def <name>(` inside an installed framework's Python tree.

    Only a definition counts, not a call site: a kernel is patched where it is written.
    The first match wins, and ambiguity is reported by returning nothing rather than by
    guessing, because two frameworks defining the same kernel name is exactly the
    collision that produced a wrong resolution once already (§11.4 item 6).
    """
    import os

    if not name or not name.isidentifier():
        return None

    roots: list[Path] = []
    override = os.environ.get(_FRAMEWORK_SOURCE_ENV, "")
    roots.extend(Path(p).expanduser() for p in override.split(os.pathsep) if p)
    for package in _FRAMEWORK_PACKAGES:
        found = _module_file(package)
        if found is not None:
            roots.append(found.parent)

    needle = f"def {name}("
    matches: list[tuple[Path, int]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            index = text.find(needle)
            if index != -1:
                matches.append((path, text[:index].count("\n") + 1))
                if len(matches) > 1:
                    return None
        if matches:
            return matches[0]
    return matches[0] if len(matches) == 1 else None
