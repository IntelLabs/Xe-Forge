"""Kernel extraction and bundle construction: a bundle is a small tree plus a
manifest with package structure preserved; any unresolved closure step downgrades
the level and records which step failed. Design rationale: docs/DESIGN.md."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from xe_forge.orbit.languages import get_backend
from xe_forge.orbit.languages.triton_backend import ClosureResult
from xe_forge.orbit.models import (
    CapturedInvocation,
    ExtractionCheck,
    ExtractionLevel,
    KernelBundle,
    KernelLanguage,
    KernelRecord,
    LaunchRecord,
    ResolutionMethod,
    SourceLocation,
)

# Written into every bundle so the extracted tree imports ahead of the installed
# package; without it a bundle silently executes the version already on sys.path.
_PATH_SHIM = '''"""
Path shim for an extracted kernel bundle.

Prepends this bundle's `src/` ahead of everything else so the extracted sources win
over the installed package. Import this before importing the kernel.
"""

import sys
from pathlib import Path

_SRC = str((Path(__file__).parent / "src").resolve())
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)
'''


@dataclass
class ExtractionResult:
    """What extraction achieved, and what it had to give up to get there."""

    bundle: KernelBundle
    level: ExtractionLevel
    closure: ClosureResult | None = None
    downgraded_from: ExtractionLevel | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def downgraded(self) -> bool:
        return self.downgraded_from is not None


class Extractor:
    """Builds a `KernelBundle` from a kernel record and its intercepted launches."""

    def __init__(self, output_root: Path, agent: object | None = None) -> None:
        self.output_root = Path(output_root)
        # Strictly opt-in: the agent is consulted only where the deterministic path
        # cannot answer. The library default is None so CI never calls an LLM.
        self.agent = agent

    # -- level selection ---------------------------------------------------

    def target_level(self, kernel: KernelRecord) -> ExtractionLevel:
        """The level to *attempt* — E2 unless the provider is opaque.

        Provenance's level hint is not treated as a ceiling; it only rules out E4
        providers, which have no source to close over at any level.
        """
        if kernel.language is KernelLanguage.OPAQUE or kernel.provider.value in (
            "onednn",
            "onemkl",
            "runtime",
        ):
            return ExtractionLevel.E4
        if kernel.extraction_level is ExtractionLevel.E4:
            return ExtractionLevel.E4
        return ExtractionLevel.E2

    def extract(
        self,
        kernel: KernelRecord,
        launch: LaunchRecord | None = None,
        inputs: CapturedInvocation | None = None,
        level: ExtractionLevel | str = "auto",
    ) -> ExtractionResult:
        """Extract one kernel, downgrading rather than emitting an unprovable bundle.

        `level="auto"` attempts the best level the kernel's provenance supports and
        falls back on failure. An explicit level is attempted as given, so a caller can
        force E3 when they already know the closure is hopeless.
        """
        requested = self.target_level(kernel) if level in ("auto", None) else ExtractionLevel(level)

        if requested is ExtractionLevel.E4:
            return self._extract_opaque(kernel, launch, inputs)
        if requested is ExtractionLevel.E3:
            return self._extract_in_situ(kernel, launch, inputs, reasons=["requested E3"])
        return self._extract_closure(kernel, launch, inputs, requested)

    # -- E1 / E2: real closure --------------------------------------------

    def _extract_closure(
        self,
        kernel: KernelRecord,
        launch: LaunchRecord | None,
        inputs: CapturedInvocation | None,
        requested: ExtractionLevel,
    ) -> ExtractionResult:
        backend_name = (kernel.language or KernelLanguage.TRITON).value
        if backend_name in ("sycl", "sycl_tla"):
            return self._extract_sycl(kernel, launch, inputs)

        backend = get_backend("triton")
        source = backend.resolve_source(kernel.runtime_name, launch)

        if not source.file:
            return self._extract_in_situ(
                kernel,
                launch,
                inputs,
                reasons=[
                    "could not resolve the entrypoint to a source file; without a "
                    "source file there is nothing to compute a closure over"
                ],
                downgraded_from=requested,
            )

        closure = backend.resolve_closure(source)
        if not closure.complete:
            return self._extract_in_situ(
                kernel,
                launch,
                inputs,
                reasons=closure.unresolved,
                downgraded_from=requested,
                closure=closure,
            )

        bundle_dir = self._bundle_dir(kernel.id)
        copied, roots = self._copy_preserving_packages(closure.files, bundle_dir / "src")
        data_deps = self._copy_data_deps(inputs, bundle_dir)
        self._write_shim(bundle_dir)

        # A single-file closure is E1; anything reaching across modules is E2.
        achieved = ExtractionLevel.E1 if len(copied) == 1 else ExtractionLevel.E2

        bundle = KernelBundle(
            kernel_id=kernel.id,
            extraction_level=achieved,
            language=kernel.language or KernelLanguage.TRITON,
            entrypoint=source.symbol or kernel.runtime_name,
            primary_source=str(bundle_dir / "src" / _relative_name(source.file, roots)),
            closure=[str(p) for p in copied],
            data_deps=data_deps,
            launch=launch,
            inputs=inputs,
            dispatch_chain=kernel.dispatch_chain,
            verification=ExtractionCheck(),
        )
        self._pin_autotune(bundle, launch, closure)

        return ExtractionResult(bundle=bundle, level=achieved, closure=closure)

    # -- E3: in-situ harness ----------------------------------------------

    def _extract_in_situ(
        self,
        kernel: KernelRecord,
        launch: LaunchRecord | None,
        inputs: CapturedInvocation | None,
        reasons: list[str],
        downgraded_from: ExtractionLevel | None = None,
        closure: ClosureResult | None = None,
    ) -> ExtractionResult:
        """E3 in-situ harness: heavier per iteration than a standalone bundle, but
        always available and always faithful to the real dispatch."""
        bundle_dir = self._bundle_dir(kernel.id)
        harness = bundle_dir / "harness.py"
        harness.parent.mkdir(parents=True, exist_ok=True)
        harness.write_text(_in_situ_harness(kernel, inputs), encoding="utf-8")

        bundle = KernelBundle(
            kernel_id=kernel.id,
            extraction_level=ExtractionLevel.E3,
            language=kernel.language or KernelLanguage.OPAQUE,
            entrypoint=kernel.framework_op or kernel.runtime_name,
            primary_source=str(harness),
            closure=[str(harness)],
            data_deps=self._copy_data_deps(inputs, bundle_dir),
            launch=launch,
            inputs=inputs,
            dispatch_chain=kernel.dispatch_chain,
            verification=ExtractionCheck(),
            downgrade_reason="; ".join(reasons) if reasons else None,
        )
        return ExtractionResult(
            bundle=bundle,
            level=ExtractionLevel.E3,
            closure=closure,
            downgraded_from=downgraded_from,
            reasons=reasons,
        )

    # -- E4: opaque reproducer --------------------------------------------

    def _extract_opaque(
        self,
        kernel: KernelRecord,
        launch: LaunchRecord | None,
        inputs: CapturedInvocation | None,
    ) -> ExtractionResult:
        """No source extraction is possible; capture a reproducer instead.

        For oneDNN the verbose problem string is the reproducer: it drives the
        library's own benchmark tool.
        """
        bundle_dir = self._bundle_dir(kernel.id)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        reproducer = bundle_dir / "reproducer.txt"
        reproducer.write_text(
            _opaque_reproducer(kernel),
            encoding="utf-8",
        )

        bundle = KernelBundle(
            kernel_id=kernel.id,
            extraction_level=ExtractionLevel.E4,
            language=KernelLanguage.OPAQUE,
            entrypoint=kernel.runtime_name,
            primary_source=str(reproducer),
            closure=[],
            data_deps=self._copy_data_deps(inputs, bundle_dir),
            launch=launch,
            inputs=inputs,
            dispatch_chain=kernel.dispatch_chain,
            verification=ExtractionCheck(
                verified=True,
                failures=[],
            ),
            downgrade_reason=(
                "runtime memory operation, not a kernel: there is no source at any "
                "level. Actions are host-side — pinned memory, fewer or larger "
                "transfers, overlap with compute."
                if kernel.provider.value == "runtime"
                else "opaque library primitive: no editable source. Actions are "
                "restricted to fusion, backend change, layout change and library config."
            ),
        )
        return ExtractionResult(bundle=bundle, level=ExtractionLevel.E4)

    # -- SYCL --------------------------------------------------------------

    def _extract_sycl(
        self,
        kernel: KernelRecord,
        launch: LaunchRecord | None,
        inputs: CapturedInvocation | None,
    ) -> ExtractionResult:
        """SYCL closure comes from the build graph, not an AST walk. These start at
        E3 and earn E2 only once a standalone build reproduces the same binary."""
        backend = get_backend("sycl")
        source = backend.resolve_source(kernel.runtime_name, launch)

        # The registry answers most real kernel names exactly; only the residue it
        # cannot unpick goes to the repository agent.
        if not source.file:
            from xe_forge.orbit.languages.sources import discover

            found = discover().resolve(kernel.runtime_name, agent=self.agent)
            if found is not None:
                path, _tree, symbol = found
                source = SourceLocation(
                    file=str(path),
                    # The full demangled name, not the bare identifier: two template
                    # instantiations share an identifier and are different code.
                    symbol=backend.demangle(kernel.runtime_name) or symbol,
                    # Estimated, unlike an exact index hit: the file is evidenced,
                    # which instantiation ran is still open.
                    confidence=0.75,
                    method=ResolutionMethod.AGENT
                    if self.agent is not None
                    else ResolutionMethod.SYMBOL_INDEX,
                    previous_file=source.file,
                    previous_method=source.method,
                    candidates=source.candidates,
                )
                kernel.source_file = str(path)

        recipe = backend.build_recipe(source)
        if recipe is not None:
            # The instantiation is recoverable only from the demangled name, so it is
            # recorded at extraction time.
            recipe.instantiation = _instantiation_of(backend.demangle(kernel.runtime_name))

        if not source.file:
            result = self._extract_in_situ(
                kernel,
                launch,
                inputs,
                reasons=[
                    "no compile-commands entry for this kernel; without the build "
                    "graph the translation unit and its exact compile line are unknown"
                ],
                downgraded_from=ExtractionLevel.E2,
            )
            result.bundle.build = recipe
            return result

        closure_files = backend.closure(source)
        bundle_dir = self._bundle_dir(kernel.id)
        copied, roots = self._copy_preserving_packages(closure_files, bundle_dir / "src")

        bundle = KernelBundle(
            kernel_id=kernel.id,
            extraction_level=ExtractionLevel.E2,
            language=kernel.language or KernelLanguage.SYCL,
            entrypoint=source.symbol or kernel.runtime_name,
            primary_source=str(bundle_dir / "src" / _relative_name(source.file, roots))
            if copied
            else None,
            closure=[str(p) for p in copied],
            data_deps=self._copy_data_deps(inputs, bundle_dir),
            launch=launch,
            build=recipe,
            inputs=inputs,
            dispatch_chain=kernel.dispatch_chain,
            verification=ExtractionCheck(),
        )
        return ExtractionResult(bundle=bundle, level=ExtractionLevel.E2)

    # -- packaging helpers -------------------------------------------------

    def _bundle_dir(self, kernel_id: str) -> Path:
        return self.output_root / kernel_id

    def _copy_preserving_packages(
        self, files: list[Path], dest_root: Path
    ) -> tuple[list[Path], list[Path]]:
        """Copy the closure keeping package structure intact.

        Returns (copied paths, source roots). Package roots are found by walking up
        while `__init__.py` exists, so relative imports still resolve in the bundle.
        """
        dest_root.mkdir(parents=True, exist_ok=True)
        roots: list[Path] = []
        for path in files:
            root = _package_root(path)
            if root not in roots:
                roots.append(root)

        resolved = {p.resolve() for p in files}
        copied: list[Path] = []
        for path in files:
            relative = _relative_name(str(path), roots)
            target = dest_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(target)
            # Carry the package markers so the tree is importable as a package.
            self._materialize_init_files(path, dest_root, relative, resolved)

        return copied, roots

    def _materialize_init_files(
        self, source: Path, dest_root: Path, relative: str, closure: set[Path]
    ) -> None:
        """Create the package markers the copied tree needs to be importable.

        An original `__init__.py` is copied only when the closure actually resolved
        through it; otherwise an empty marker is written, since a verbatim init could
        import siblings the bundle does not carry.
        """
        parts = Path(relative).parts[:-1]
        current_src = _package_root(source)
        current_dst = dest_root
        for part in parts:
            current_src = current_src / part
            current_dst = current_dst / part
            current_dst.mkdir(parents=True, exist_ok=True)
            init_src = current_src / "__init__.py"
            init_dst = current_dst / "__init__.py"
            if init_dst.exists():
                continue
            if init_src.is_file() and init_src.resolve() in closure:
                shutil.copy2(init_src, init_dst)
            else:
                init_dst.write_text(
                    "# Package marker written by xe-orbit. The original __init__.py was\n"
                    "# not part of this kernel's closure, so it is deliberately empty:\n"
                    "# carrying its imports would make the bundle depend on modules it\n"
                    "# does not ship.\n",
                    encoding="utf-8",
                )

    def _copy_data_deps(self, inputs: CapturedInvocation | None, bundle_dir: Path) -> list[str]:
        """Data dependencies are copied, never regenerated."""
        if inputs is None or not inputs.data_deps:
            return []
        target = bundle_dir / "data"
        target.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for dep in inputs.data_deps:
            source = Path(dep)
            if not source.is_file():
                continue
            destination = target / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))
        return copied

    def _write_shim(self, bundle_dir: Path) -> None:
        (bundle_dir / "_bundle_path.py").write_text(_PATH_SHIM, encoding="utf-8")

    def _pin_autotune(
        self, bundle: KernelBundle, launch: LaunchRecord | None, closure: ClosureResult
    ) -> None:
        """Pin the autotune configuration that actually won at runtime; a missing
        winner is recorded as a gap rather than papered over with a default."""
        if not closure.autotune_configs:
            return
        if launch is None or launch.selected_autotune_config is None:
            bundle.env_pins["ORBIT_AUTOTUNE_UNPINNED"] = (
                "kernel carries an autotune decorator but no winning config was "
                "intercepted; pinned-vs-pinned comparison is not possible"
            )
            return
        bundle.env_pins["ORBIT_AUTOTUNE_PINNED"] = repr(launch.selected_autotune_config)


def _package_root(path: Path) -> Path:
    """Walk up while `__init__.py` exists, so packages keep their import path."""
    current = path.parent
    while (current / "__init__.py").is_file() and current.parent != current:
        current = current.parent
    return current


def _relative_name(file: str, roots: list[Path]) -> str:
    path = Path(file)
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return path.name


def _in_situ_harness(kernel: KernelRecord, inputs: CapturedInvocation | None) -> str:
    """Generate an E3 harness whose `forward` drives the installed framework."""
    op = kernel.framework_op or kernel.runtime_name
    chain = " -> ".join(kernel.dispatch_chain) or "unknown"
    return f'''"""
In-situ harness for {kernel.id} ({kernel.runtime_name}).

Extraction level E3: this kernel could not be isolated into a standalone closure, so
`Model.forward` drives the op through the installed framework instead — heavier per
iteration, but always faithful to the real dispatch.

Dispatch chain: {chain}
"""

from pathlib import Path

import torch

_HERE = Path(__file__).parent


class Model(torch.nn.Module):
    """Drives the real op with captured inputs.

    Xe-Forge resolves `Model` by duck typing, so nothing here needs to import a base
    class. `get_example_inputs` loads the tensors captured from the running workload,
    with strides and non-contiguous layouts preserved — regenerating them from shape
    and dtype is what silently breaks masked attention and paged KV-cache layouts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.op_name = {op!r}

    def get_example_inputs(self, input_shapes=None, device="xpu"):
        from xe_forge.orbit.capture import load_invocation
        from xe_forge.orbit.models import CapturedInvocation

        manifest = _HERE / "inputs" / "invocation.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"no captured inputs at {{manifest}}; run `xe-orbit capture {kernel.id}` first"
            )
        invocation = CapturedInvocation.model_validate_json(
            manifest.read_text(encoding="utf-8")
        )
        return load_invocation(invocation, device=device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Bind this harness to the framework entry point for "
            f"{{self.op_name!r}}. The adapter's build_in_situ_harness() supplies the "
            "framework-specific call; the generic path cannot know it."
        )
'''


def _opaque_reproducer(kernel: KernelRecord) -> str:
    shapes = ""
    if kernel.shapes:
        first = kernel.shapes[0]
        dims = ", ".join(f"{k}={v}" for k, v in first.dims.items())
        dtypes = ", ".join(f"{k}:{v}" for k, v in first.dtypes.items())
        shapes = f"observed shape: {dims}\nobserved dtypes: {dtypes}\n"
    return (
        f"# Opaque reproducer for {kernel.id}\n"
        f"kernel: {kernel.runtime_name}\n"
        f"provider: {kernel.provider.value}\n"
        f"calls: {kernel.calls}\n"
        f"total_time_us: {kernel.total_time_us:.1f}\n"
        f"gpu_share: {kernel.gpu_time_share * 100:.2f}%\n"
        f"{shapes}"
        "\n"
        "No source extraction is possible for this provider. To obtain a standalone\n"
        "measurement, re-run the workload with the library's verbose mode enabled\n"
        "(DNNL_VERBOSE=1 for oneDNN) and capture the primitive descriptor line: that\n"
        "problem string drives the library's own benchmark tool.\n"
        "\n"
        "Available actions: region fusion, backend change, layout change, library config.\n"
    )


def _instantiation_of(demangled: str) -> str:
    """The outermost template argument list of a demangled kernel name, if any.

    `at::native::xpu::FillFunctor<int>` yields `<int>`; a non-template name yields "".
    """
    if not demangled:
        return ""
    start = demangled.find("<")
    if start == -1:
        return ""
    depth = 0
    for index in range(start, len(demangled)):
        char = demangled[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return demangled[start : index + 1]
    # An unbalanced name means the runtime truncated it.
    return ""
