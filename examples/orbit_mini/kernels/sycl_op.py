"""Python side of the hand-written SYCL dispatcher op (plan §11, §13, §15.2).

The Triton path is not the only kernel language Orbit has to handle, and §11 is
explicit that SYCL is not the exception. So ``orbit_mini`` ships one genuinely
hand-written SYCL kernel — ``examples/orbit_mini/sycl/orbit_mini_rmsnorm.cpp``,
with its own ``CMakeLists.txt`` — registered through ``TORCH_LIBRARY`` as
``orbit_mini::rmsnorm_xpu``.

What that gets exercised, per §15.2:

* build-graph closure — the extension has a real ``compile_commands.json``
  (§11.3: for SYCL, closure comes from the build system, not an AST walk),
* the compiler-option sweep and the ``icpx`` harness,
* the **P1 operator override** rung of §13 on a SYCL op rather than a Triton
  kernel, since the op is registered on the XPU dispatch key and an optimized
  build shadows it without touching PyTorch.

**The extension is not built by default and must never be required.** This
module probes for the op, and falls back to torch when it is absent — which is
always the case on the CPU-only CI tier. Set ``ORBIT_MINI_SYCL_LIB`` to a built
``liborbit_mini_sycl.so`` to exercise the real thing on an XPU machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

#: Where the SYCL source and build files live. Recorded as a path, not imported,
#: because the build graph is the closure for a compiled kernel (§11.3).
SYCL_SOURCE_DIR: Path = Path(__file__).resolve().parent.parent / "sycl"

#: Qualified op name registered by the extension. This is the §13 P1 handle.
SYCL_OP_NAME: str = "orbit_mini::rmsnorm_xpu"

#: Env var pointing at a built shared library, if one exists.
SYCL_LIB_ENV: str = "ORBIT_MINI_SYCL_LIB"

_probe_done = False
_op: Any = None
_load_error: str | None = None


def _probe() -> Any:
    """Load and resolve the dispatcher op once. Never raises."""
    global _probe_done, _op, _load_error
    if _probe_done:
        return _op
    _probe_done = True

    lib_path = os.environ.get(SYCL_LIB_ENV)
    if lib_path:
        try:
            torch.ops.load_library(lib_path)
        except OSError as exc:
            _load_error = f"failed to load {lib_path}: {exc}"
            return None

    try:
        _op = torch.ops.orbit_mini.rmsnorm_xpu
    except (AttributeError, RuntimeError) as exc:
        _op = None
        _load_error = f"{SYCL_OP_NAME} is not registered ({type(exc).__name__})"
    return _op


def is_available(device: torch.device | str | None = None) -> bool:
    """True only when the op is registered *and* the device can run it.

    Both halves matter. A registered XPU op called on a CPU tensor is a
    dispatch error, not a fallback, and the fallback is the whole point here.
    """
    op = _probe()
    if op is None:
        return False
    if device is None:
        return True
    return torch.device(device).type == "xpu"


def status() -> str:
    """One-line report for the run summary."""
    if _probe() is not None:
        return f"{SYCL_OP_NAME}: registered"
    reason = _load_error or "extension not built"
    return f"{SYCL_OP_NAME}: unavailable ({reason})"


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm through the SYCL dispatcher op, or torch if it is not there.

    Kept as a thin shim so the P1 override verification in §13 — "re-profile and
    confirm the new kernel appears in the trace *and the old one does not*" —
    has a single, unambiguous call site to assert against.
    """
    op = _probe()
    if op is not None and x.device.type == "xpu":
        return op(x, weight, float(eps))
    return _rmsnorm_torch_fallback(x, weight, eps)


def _rmsnorm_torch_fallback(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference for the SYCL kernel, numerically matched to the .cpp source."""
    xf = x.to(torch.float32)
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * weight.to(torch.float32)).to(x.dtype)
