"""Availability shim for Triton (plan §15.2, §11.2).

Xe-Orbit's extractor has to be able to *read* the Triton structure of this
workload — ``@triton.jit`` device helpers, an autotune config list, a heuristics
callable — on machines where Triton is not installed at all. The reference CI
tier T0 (§16.6) is CPU-only with no Triton and no GPU runtime, and a workload
that simply stops being Triton-shaped there would test nothing.

So this module never branches the *source*. The decorators and the ``tl``
namespace always exist:

* Triton present  → they are the real thing; the kernels compile and launch.
* Triton absent   → they are structural stand-ins that preserve every attribute
  the closure walk in §12.6 reads (``fn``, ``configs``, ``key``, ``values``,
  ``kwargs``, ``num_warps``, ``num_stages``) and refuse to *execute*.

Refusing to execute is safe because every launch wrapper in this package checks
:data:`HAS_TRITON` (and the device type) before taking the Triton path, and
falls back to plain torch otherwise.

Config objects are :class:`KernelConfig`, not ``triton.Config``, on purpose. The
autotune config list must be one statically inspectable module-level literal
regardless of whether Triton is importable; :meth:`KernelConfig.to_triton`
converts at decoration time when there is a real Triton to convert for.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - depends on the machine, not on the code path
    import triton as _triton
    import triton.language as _tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - the CPU-only CI path
    _triton = None
    _tl = None
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# `tl` stand-in
# ---------------------------------------------------------------------------


class _LanguageStub:
    """Minimal stand-in for ``triton.language``.

    Only ``constexpr`` has to resolve at *definition* time, because it appears in
    kernel parameter annotations (``BLOCK_N: tl.constexpr``) which Python
    evaluates when the ``def`` executes. Everything else is referenced only from
    inside kernel bodies, which never run without Triton, so an attribute error
    that carries an explanation is the most useful thing to hand back.

    Note that the kernel modules deliberately do *not* use
    ``from __future__ import annotations``: Triton reads real annotation objects
    off the signature, and stringised annotations break its constexpr detection.
    """

    #: `x: tl.constexpr` must evaluate to *something* at def time.
    constexpr = int

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"triton.language.{name} was evaluated but Triton is not installed. "
            "A launch wrapper took the Triton path without checking HAS_TRITON first."
        )


tl: Any = _tl if HAS_TRITON else _LanguageStub()
triton: Any = _triton


# ---------------------------------------------------------------------------
# Autotune configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelConfig:
    """One point in an autotune search space.

    Mirrors ``triton.Config``'s public surface so §12.7 config pinning can read
    either kind through the same attribute names.
    """

    kwargs: dict[str, int] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 2

    def to_triton(self) -> Any:
        """Convert to a real ``triton.Config``. Only valid when Triton is present."""
        if not HAS_TRITON:  # pragma: no cover - guarded by every caller
            raise RuntimeError("KernelConfig.to_triton() requires Triton")
        return _triton.Config(
            dict(self.kwargs), num_warps=self.num_warps, num_stages=self.num_stages
        )

    def describe(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in sorted(self.kwargs.items()))
        return f"{items}, num_warps={self.num_warps}, num_stages={self.num_stages}"


# ---------------------------------------------------------------------------
# Decorator stand-ins
# ---------------------------------------------------------------------------


class _StubJITFunction:
    """Structural stand-in for ``triton.JITFunction``.

    Keeps ``.fn`` (which is what an AST-based closure walk starts from) and the
    ``kernel[grid](...)`` subscript protocol, so extraction code written against
    real Triton does not need a second code path for the CPU-only case.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.is_stub = True
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self.fn.__name__} is a Triton kernel and Triton is not installed. "
            "Use the pure-torch fallback in this module."
        )

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        def _launch(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"{self.fn.__name__}[{grid!r}] launched without Triton installed.")

        return _launch

    def __repr__(self) -> str:
        return f"<stub triton.jit {self.fn.__name__}>"


def jit(fn: Callable[..., Any]) -> Any:
    """``@triton.jit`` when Triton exists, a structural stand-in when it does not."""
    if HAS_TRITON:
        return _triton.jit(fn)
    return _StubJITFunction(fn)


def autotune(
    configs: Sequence[KernelConfig], key: Sequence[str], **kwargs: Any
) -> Callable[[Any], Any]:
    """``@triton.autotune`` over a :class:`KernelConfig` list.

    The config list stays a module-level literal either way, which is the thing
    §12.7 has to capture: whichever config wins at runtime must be pinned into
    the bundle, or baseline and candidate are not being compared under the same
    configuration policy.
    """

    def _decorate(kernel: Any) -> Any:
        if HAS_TRITON:
            tuned = _triton.autotune(
                configs=[c.to_triton() for c in configs], key=list(key), **kwargs
            )(kernel)
            # Keep the neutral list reachable for the extractor even on the real path.
            tuned.orbit_configs = list(configs)
            return tuned
        kernel.configs = list(configs)
        kernel.key = list(key)
        kernel.orbit_configs = list(configs)
        return kernel

    return _decorate


def heuristics(values: dict[str, Callable[[dict[str, Any]], Any]]) -> Callable[[Any], Any]:
    """``@triton.heuristics``.

    §12.6 step 4 calls out heuristics callables specifically because they close
    over module state that has to travel with the bundle. The stand-in keeps
    ``.values`` so the closure walk finds the callables in both cases.
    """

    def _decorate(kernel: Any) -> Any:
        if HAS_TRITON:
            tuned = _triton.heuristics(values)(kernel)
            tuned.orbit_heuristics = dict(values)
            return tuned
        kernel.values = dict(values)
        kernel.orbit_heuristics = dict(values)
        return kernel

    return _decorate


def cdiv(a: int, b: int) -> int:
    """``triton.cdiv`` without needing Triton for grid arithmetic."""
    return -(-a // b)


def next_power_of_2(n: int) -> int:
    """``triton.next_power_of_2`` without needing Triton."""
    p = 1
    while p < n:
        p <<= 1
    return p
