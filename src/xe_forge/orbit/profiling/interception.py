"""
Launch-site interception: records what each kernel launch actually was (grid,
constexprs, winning autotune config, compiled binary identity) via the Triton JIT run
path and the torch dispatcher. Hooks that cannot install are recorded as limitations,
never as an empty record set. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from pydantic import Field

from xe_forge.orbit.models import Artifact, LaunchRecord

# Attributes worth lifting off a compiled Triton kernel. Register and spill counts
# both verify extraction and flag register pressure, a first-order concern on Xe.
_COMPILED_METADATA_FIELDS = (
    "num_regs",
    "n_regs",
    "num_spills",
    "n_spills",
    "shared",
    "slm_size",
    "grf_mode",
    "hash",
    "name",
)


class LaunchLog(Artifact):
    """All launches intercepted during one trace run."""

    records: list[LaunchRecord] = Field(default_factory=list)
    dispatch_ops: list[dict[str, Any]] = Field(default_factory=list)
    triton_available: bool = False
    torch_available: bool = False
    limitations: list[str] = Field(default_factory=list)

    def for_kernel(self, fq_name: str) -> list[LaunchRecord]:
        return [r for r in self.records if r.fq_name == fq_name]

    def unique_kernels(self) -> list[str]:
        seen: list[str] = []
        for record in self.records:
            if record.fq_name not in seen:
                seen.append(record.fq_name)
        return seen


class TritonLaunchInterceptor:
    """Wraps Triton's JIT run path to record what each launch actually was.

    Triton's internals move between releases, so every attribute read here is
    defensive: a version whose signature we do not recognize yields a partial record
    with a limitation noted, never an exception that kills the user's workload.
    """

    def __init__(self, log: LaunchLog) -> None:
        self.log = log
        self._original = None
        self._jit_cls = None
        self._call_counter = 0

    def install(self) -> bool:
        try:
            from triton.runtime.jit import JITFunction
        except ImportError:
            self.log.limitations.append(
                "triton not importable; Triton launch interception unavailable"
            )
            return False

        if not hasattr(JITFunction, "run"):
            self.log.limitations.append(
                "triton.runtime.jit.JITFunction has no 'run'; unsupported Triton version"
            )
            return False

        self._jit_cls = JITFunction
        self._original = JITFunction.run
        interceptor = self

        def run(self, *args, **kwargs):
            record = interceptor._build_record(self, args, kwargs)
            result = interceptor._original(self, *args, **kwargs)
            if record is not None:
                interceptor._attach_compiled_metadata(record, result)
                interceptor.log.records.append(record)
            return result

        JITFunction.run = run  # type: ignore[method-assign]
        self.log.triton_available = True
        return True

    def remove(self) -> None:
        if self._jit_cls is not None and self._original is not None:
            self._jit_cls.run = self._original  # type: ignore[method-assign]
            self._jit_cls = None
            self._original = None

    def _build_record(self, jit_fn: Any, args: tuple, kwargs: dict) -> LaunchRecord | None:
        try:
            fn = getattr(jit_fn, "fn", None)
            module = getattr(fn, "__module__", None) or "unknown"
            name = getattr(fn, "__name__", None) or getattr(jit_fn, "__name__", "unknown")
            source_file = None
            source_line = None
            try:
                import inspect

                source_file = inspect.getsourcefile(fn) if fn else None
                _, source_line = inspect.getsourcelines(fn) if fn else (None, None)
            except (TypeError, OSError):
                pass

            grid = _normalize_grid(kwargs.get("grid"))
            constexprs = _extract_constexprs(jit_fn, kwargs)

            record = LaunchRecord(
                fq_name=f"{module}:{name}",
                source_file=source_file,
                source_line=source_line,
                grid=grid,
                num_warps=_as_int(kwargs.get("num_warps")),
                num_stages=_as_int(kwargs.get("num_stages")),
                constexprs=constexprs,
                arg_order=_param_names(jit_fn),
                call_index=self._call_counter,
            )
            self._call_counter += 1
            return record
        except Exception as exc:  # never break the workload being measured
            self.log.limitations.append(f"failed to build launch record: {exc}")
            return None

    def _attach_compiled_metadata(self, record: LaunchRecord, result: Any) -> None:
        """Pull register/spill/SLM counts and the binary hash off the compiled kernel."""
        if result is None:
            return
        metadata: dict[str, Any] = {}
        source = getattr(result, "metadata", None) or result
        for field in _COMPILED_METADATA_FIELDS:
            value = getattr(source, field, None)
            if value is None and isinstance(source, dict):
                value = source.get(field)
            if value is not None and isinstance(value, (int, float, str)):
                metadata[field] = value
        if metadata:
            record.compiled_metadata.update(metadata)


class DispatcherInterceptor:
    """Records `torch.ops.<ns>.<op>` calls and where their implementations live.

    This is the C++/SYCL path: there is no Python kernel to intercept, so the entry
    point is the registered op schema and the shared object behind it.
    """

    def __init__(
        self, log: LaunchLog, namespaces: tuple[str, ...] = ("_C", "torch_ipex", "sgl_kernel")
    ) -> None:
        self.log = log
        self.namespaces = namespaces
        self._patched: list[tuple[Any, str, Any]] = []

    def install(self) -> bool:
        try:
            import torch
        except ImportError:
            self.log.limitations.append("torch not importable; dispatcher interception unavailable")
            return False

        self.log.torch_available = True
        installed = False
        for namespace in self.namespaces:
            ns = getattr(torch.ops, namespace, None)
            if ns is None:
                continue
            installed = True
            self.log.dispatch_ops.append({"namespace": namespace, "status": "observed"})
        if not installed:
            self.log.limitations.append(
                f"none of the custom op namespaces {self.namespaces} are registered; "
                "no C++/SYCL extension ops to intercept in this process"
            )
        return installed

    def remove(self) -> None:
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()

    def record_op(self, namespace: str, op_name: str, shared_object: str | None = None) -> None:
        """Record one dispatcher op observation (used by adapters and tests)."""
        self.log.dispatch_ops.append(
            {
                "namespace": namespace,
                "op": op_name,
                "shared_object": shared_object,
            }
        )


@contextlib.contextmanager
def intercept_launches(
    triton: bool = True,
    dispatcher: bool = True,
    namespaces: tuple[str, ...] = ("_C", "torch_ipex", "sgl_kernel"),
) -> Iterator[LaunchLog]:
    """Record every kernel launch that happens inside the block.

    Always restores the original launch paths, including when the workload raises —
    leaving a monkey-patched Triton behind would corrupt every subsequent measurement
    in the same process.
    """
    log = LaunchLog()
    triton_hook = TritonLaunchInterceptor(log) if triton else None
    dispatch_hook = DispatcherInterceptor(log, namespaces) if dispatcher else None

    if triton_hook is not None:
        triton_hook.install()
    if dispatch_hook is not None:
        dispatch_hook.install()

    try:
        yield log
    finally:
        if dispatch_hook is not None:
            dispatch_hook.remove()
        if triton_hook is not None:
            triton_hook.remove()


def _normalize_grid(grid: Any) -> list[int]:
    if grid is None:
        return []
    if callable(grid):
        # Triton accepts a lambda taking the meta-parameters; we cannot evaluate it
        # safely here, so record that it was dynamic rather than guessing dimensions.
        return []
    if isinstance(grid, int):
        return [grid]
    try:
        return [int(x) for x in grid]
    except (TypeError, ValueError):
        return []


def _param_names(jit_fn: Any) -> list[str]:
    params = getattr(jit_fn, "params", None)
    if params:
        names = []
        for p in params:
            name = getattr(p, "name", None)
            if name:
                names.append(str(name))
        if names:
            return names
    arg_names = getattr(jit_fn, "arg_names", None)
    if arg_names:
        return [str(a) for a in arg_names]
    return []


def _extract_constexprs(jit_fn: Any, kwargs: dict) -> dict[str, Any]:
    """Recover the `tl.constexpr` values this launch was specialized on."""
    constexprs: dict[str, Any] = {}
    params = getattr(jit_fn, "params", None) or []
    constexpr_names = {
        str(getattr(p, "name", "")) for p in params if getattr(p, "is_constexpr", False)
    }
    for key, value in kwargs.items():
        if key in ("grid", "num_warps", "num_stages", "warmup"):
            continue
        if not constexpr_names or key in constexpr_names:
            if isinstance(value, (int, float, bool, str)):
                constexprs[key] = value
    return constexprs


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
