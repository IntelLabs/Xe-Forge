"""Pluggable build backends for compiled-DSL kernels.

Xe-Forge's own SYCL path compiles through ``ai_bench.sycl.compiler.SYCLCompiler``,
which knows exactly one way to build: a fixed include list derived from
``SYCL_TLA_DIR``, no AOT device target, no SPIR-V extension control, no register
mode, and no way for a kernel to declare the libraries it links against. Each of
those is a property of *the part* or of *the kernel*, not of Xe-Forge, so growing
a branch here per toolchain permutation does not converge.

A host project that already derives them -- because it owns an accelerator
capability record, a dependency-to-linker-flag mapping, and a compiler it invokes
for its own kernels -- supplies a backend instead. Xe-Forge keeps what is good in
it: the trial tree, the branching, the knowledge base.

The default remains the ``ai_bench`` compiler, so a workspace that names no
backend behaves exactly as before.

Registering one:

* in-process, by a host that imports xe_forge::

      from xe_forge.core.build_backend import register_build_backend
      register_build_backend("my-toolchain", MyBackend)

* by reference, with no import on Xe-Forge's side -- ``--build-backend
  some.module:factory``. The attribute may be a ``BuildBackend`` or a zero-argument
  callable returning one.

* by entry point, under the ``xe_forge.build_backends`` group.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from xe_forge.models import ExecutionResult

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "ai_bench"
ENTRY_POINT_GROUP = "xe_forge.build_backends"

__all__ = [
    "DEFAULT_BACKEND",
    "ENTRY_POINT_GROUP",
    "BuildBackend",
    "BuildError",
    "BuildSpec",
    "BuiltKernel",
    "available_build_backends",
    "register_build_backend",
    "resolve_build_backend",
]


class BuildError(RuntimeError):
    """A kernel could not be built, or a backend could not be resolved."""


@dataclass(frozen=True)
class BuildSpec:
    """Everything a backend needs about a kernel that is not its source text.

    ``dependencies`` is the field that the ``ai_bench`` path has nowhere to put:
    the names of libraries the kernel calls into (``"onednn"``, ``"onemkl"``,
    ``"level_zero"``, ...). It is deliberately a list of *names*, not flags --
    resolving a name to ``-l``/``-L``/``-rpath`` requires knowing where that
    library is on this machine, which is the backend's job and not the caller's.
    """

    name: str = "kernel"
    language: str = "sycl"
    dependencies: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    extra_flags: tuple[str, ...] = ()
    workdir: str | None = None
    # Backend-specific values that do not generalise. A backend must tolerate
    # keys it does not recognise rather than fail on them.
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class BuiltKernel(Protocol):
    """A kernel that has been built and can now be run.

    Keyword arguments carry the run configuration -- ``dims`` (the spec
    variant's dimensions), ``iterations``, and, for backends that exchange
    tensors through files, ``input_dir`` / ``output_dir``. A backend ignores
    the ones that do not apply to it rather than rejecting them, because the
    caller does not know which execution model it got.
    """

    def __call__(self, **run_args: Any) -> ExecutionResult: ...


@runtime_checkable
class BuildBackend(Protocol):
    """Builds kernel source into something runnable."""

    name: str

    def build(self, source: str, spec: BuildSpec, device: str) -> BuiltKernel: ...


_REGISTRY: dict[str, Callable[[], BuildBackend]] = {}


def register_build_backend(
    name: str,
    factory: Callable[[], BuildBackend] | BuildBackend,
    *,
    replace: bool = False,
) -> None:
    """Register a backend under *name*.

    *factory* may be a backend instance or a zero-argument callable returning
    one. A callable is preferred: resolution then costs nothing until the
    backend is actually asked for, which matters because constructing one can
    probe the machine for a compiler.
    """
    if name in _REGISTRY and not replace:
        raise BuildError(
            f"build backend {name!r} is already registered; pass replace=True to override"
        )
    _REGISTRY[name] = factory if callable(factory) else (lambda: factory)  # type: ignore[return-value]


def available_build_backends() -> list[str]:
    """Names that :func:`resolve_build_backend` will accept without an import path."""
    _register_builtins()
    names = set(_REGISTRY)
    names.update(ep.name for ep in _entry_points())
    return sorted(names)


def _register_builtins() -> None:
    if DEFAULT_BACKEND in _REGISTRY:
        return

    def _ai_bench() -> BuildBackend:
        from xe_forge.core.build_backends.ai_bench_sycl import AiBenchSyclBackend

        return AiBenchSyclBackend()

    _REGISTRY[DEFAULT_BACKEND] = _ai_bench


def _entry_points():
    try:
        from importlib.metadata import entry_points

        return list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception as exc:  # pragma: no cover - depends on installed metadata
        logger.debug("Could not read %s entry points: %s", ENTRY_POINT_GROUP, exc)
        return []


def _load_reference(ref: str) -> BuildBackend:
    """Load ``module:attr``, calling *attr* if it is a factory."""
    module_name, _, attr = ref.partition(":")
    if not attr:
        raise BuildError(
            f"build backend {ref!r} is not a registered name and is not of the form 'module:attr'. "
            f"Registered: {', '.join(available_build_backends())}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BuildError(
            f"cannot import module {module_name!r} for build backend {ref!r}: {exc}"
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise BuildError(f"module {module_name!r} has no attribute {attr!r}") from None

    backend = obj() if callable(obj) and not isinstance(obj, BuildBackend) else obj
    if not isinstance(backend, BuildBackend):
        raise BuildError(
            f"{ref!r} resolved to {type(backend).__name__}, which does not implement "
            "BuildBackend (needs a 'name' attribute and a build(source, spec, device) method)"
        )
    return backend


def resolve_build_backend(ref: str | BuildBackend | None) -> BuildBackend:
    """Resolve a backend from a name, a ``module:attr`` reference, or an instance.

    ``None`` resolves to the default, which is today's ``ai_bench`` compiler.
    """
    if ref is None:
        ref = DEFAULT_BACKEND
    if not isinstance(ref, str):
        if not isinstance(ref, BuildBackend):
            raise BuildError(
                f"{type(ref).__name__} does not implement BuildBackend "
                "(needs a 'name' attribute and a build(source, spec, device) method)"
            )
        return ref

    _register_builtins()
    factory = _REGISTRY.get(ref)
    if factory is not None:
        return factory()

    for ep in _entry_points():
        if ep.name == ref:
            obj = ep.load()
            backend = obj() if callable(obj) and not isinstance(obj, BuildBackend) else obj
            if not isinstance(backend, BuildBackend):
                raise BuildError(
                    f"entry point {ref!r} resolved to {type(backend).__name__}, "
                    "which does not implement BuildBackend"
                )
            return backend

    return _load_reference(ref)
