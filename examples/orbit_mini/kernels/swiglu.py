"""SwiGLU projection + activation — third kernel of the fusable region.

The gate/up projection and the SiLU-gated multiply are one launch here, which
makes this the consumer end of the ``gemm -> rmsnorm -> swiglu`` chain that
``kernels/region.py`` declares for the Xe-Fuse path (§15.2, §12.11).

Its device helper is ``device_ops.swiglu_gate``, which is
``helpers_b._swiglu_activate`` under an alias, which calls
``helpers_c._clamp_to_finite``. So this kernel's closure *overlaps* the hot
kernel's without being identical — a bundle that shares helper files between
kernels has to get the sharing right in both directions (§12.6).

It also reads the same ``tuned_configs.json`` data dependency, so removing that
file breaks two kernels rather than one, and §12.12 step 5 sees a failure from
either entry point.
"""

import torch

from .device_ops import swiglu_gate as _swiglu_gate_alias
from .triton_compat import HAS_TRITON, KernelConfig, autotune, jit, tl
from .tuned import lookup

#: Search space for the activation kernel. Smaller than the RMSNorm one on
#: purpose: two kernels with *different* config lists means an extractor cannot
#: get away with pinning one global config (§12.7).
SWIGLU_CONFIGS: list[KernelConfig] = [
    KernelConfig({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    KernelConfig({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    KernelConfig({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
]

LAST_LAUNCH: dict = {}


@autotune(configs=SWIGLU_CONFIGS, key=["N_COLS"])
@jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    N_COLS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Elementwise ``silu(gate) * up`` over a flattened activation buffer."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_elements

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0)
    out = _swiglu_gate_alias(gate, up)
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def select_config(n_cols: int) -> KernelConfig:
    """Deterministic stand-in for Triton's autotuner (see rmsnorm.select_config)."""
    for config in SWIGLU_CONFIGS:
        if config.kwargs["BLOCK_N"] >= n_cols:
            return config
    return SWIGLU_CONFIGS[-1]


def swiglu_projection(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """Project ``x`` through the gate and up weights, then apply SwiGLU.

    Consumer end of the fusable region. The two intermediates it materialises
    (``gate`` and ``up``) are exactly what a fused replacement would eliminate,
    so ``region.py`` names them explicitly.
    """
    tuned = lookup(x.device)
    gate = torch.matmul(x, gate_weight.transpose(-2, -1))
    up = torch.matmul(x, up_weight.transpose(-2, -1))

    n_cols = gate.shape[-1]
    config = select_config(n_cols)
    LAST_LAUNCH.clear()
    LAST_LAUNCH.update(
        {
            "kernel": "orbit_mini::_swiglu_kernel",
            "n_elements": gate.numel(),
            "n_cols": n_cols,
            "config": config.describe(),
            "tuned_entry": tuned.device_key,
            "clamp_limit": tuned.clamp_limit,
        }
    )

    if HAS_TRITON and x.device.type in ("cuda", "xpu"):
        LAST_LAUNCH["backend"] = "triton"
        gate_flat = gate.contiguous().reshape(-1)
        up_flat = up.contiguous().reshape(-1)
        out = torch.empty_like(gate_flat)
        block_n = config.kwargs["BLOCK_N"]
        grid = ((gate_flat.numel() + block_n - 1) // block_n,)
        _swiglu_kernel[grid](gate_flat, up_flat, out, gate_flat.numel(), n_cols)
        return out.reshape(gate.shape)

    LAST_LAUNCH["backend"] = "torch"
    return _swiglu_torch(gate, up, tuned.clamp_limit)


def _swiglu_torch(gate: torch.Tensor, up: torch.Tensor, clamp: float) -> torch.Tensor:
    """Pure-torch reference, matched to ``helpers_b._swiglu_activate``.

    The clamp is not cosmetic: it comes from the tuned-config JSON and it is
    applied inside the device helper too. Drop either and the two paths diverge
    on large activations, which is the kind of silent mismatch §12.12 step 3
    exists to catch.
    """
    g = torch.clamp(gate.to(torch.float32), -clamp, clamp)
    return (g * torch.sigmoid(g) * up.to(torch.float32)).to(gate.dtype)
