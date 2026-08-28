"""
The optimized replacement kernel (plan §13, rung P1).

This is what Xe-Forge would produce and Orbit would patch back. It stands in for an
agent-generated kernel so the reinsertion path can be validated without spending tokens
or needing a GPU.

Two implementations, selected at import:

* **Triton** — a real fused RMSNorm kernel, used when Triton is importable and the
  tensor is on an accelerator. This is the path a real candidate takes on XPU.
* **Fused torch** — one pass with `rsqrt` and no intermediate allocations, used on CPU
  and whenever Triton is unavailable.

The fallback is not a token gesture. The baseline it replaces makes four passes over
the data and allocates three intermediates; this makes one pass and allocates one. The
speedup is real and measurable on CPU, which is what lets the whole loop — apply,
re-profile, dispatch assertion, statistics — be exercised end to end on a machine with
no accelerator at all.

Importing this module registers the override. Not importing it is the revert.
"""

from __future__ import annotations

import torch

from examples.kernel_replacement import dispatch_log

NAMESPACE = "orbit_demo"
OP_NAME = "rms_norm"

# The marker the dispatch assertion looks for. It must differ from the baseline's
# marker, or "did the override take effect" is unanswerable (§13).
OPTIMIZED_KERNEL_TRITON = "orbit_demo_rms_norm_triton_fused"
OPTIMIZED_KERNEL_TORCH = "orbit_demo_rms_norm_torch_fused"

try:  # pragma: no cover - depends on the installed stack
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:  # pragma: no cover - requires an accelerator to execute

    @triton.jit
    def _rms_norm_fused(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """One row per program: accumulate the sum of squares, then scale in place.

        Single pass over the row, no intermediate materialization — which is exactly
        the difference from the naive baseline this replaces.
        """
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_row
        out_row = out_ptr + row * stride_row

        accumulator = tl.zeros([BLOCK_N], dtype=tl.float32)
        for offset in range(0, n_cols, BLOCK_N):
            cols = offset + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            values = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            accumulator += values * values

        inv_rms = 1.0 / tl.sqrt(tl.sum(accumulator, axis=0) / n_cols + eps)

        for offset in range(0, n_cols, BLOCK_N):
            cols = offset + tl.arange(0, BLOCK_N)
            mask = cols < n_cols
            values = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            weights = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_row + cols, values * inv_rms * weights, mask=mask)


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Launch the Triton kernel over a flattened view of the leading dimensions."""
    dispatch_log.record(OPTIMIZED_KERNEL_TRITON)

    original_shape = x.shape
    flattened = x.reshape(-1, original_shape[-1]).contiguous()
    out = torch.empty_like(flattened)
    n_rows, n_cols = flattened.shape

    block = min(1024, triton.next_power_of_2(n_cols))
    _rms_norm_fused[(n_rows,)](
        flattened,
        weight,
        out,
        flattened.stride(0),
        n_cols,
        eps,
        BLOCK_N=block,
    )
    return out.reshape(original_shape)


def _torch_fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """One pass, one allocation. `rsqrt` avoids the divide the baseline pays for."""
    dispatch_log.record(OPTIMIZED_KERNEL_TORCH)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms * weight


def optimized_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The entry point the generated override module calls."""
    if (HAS_TRITON and x.is_cuda) or (HAS_TRITON and x.device.type == "xpu"):
        return _triton_rms_norm(x, weight, eps)
    return _torch_fused_rms_norm(x, weight, eps)


def active_kernel_name() -> str:
    """Which implementation this process would dispatch to."""
    return OPTIMIZED_KERNEL_TRITON if HAS_TRITON else OPTIMIZED_KERNEL_TORCH


def register() -> str:
    """Override the baseline on every device key the workload might run on.

    This is what rung P1 is: an implementation registered for an op that already
    exists, shadowing the default. The framework is untouched, and reverting is simply
    not importing this module.
    """
    impl = torch.library.Library(NAMESPACE, "IMPL")
    for key in ("CPU", "XPU"):
        try:
            impl.impl(OP_NAME, optimized_kernel, key)
        except RuntimeError:
            # A key with no baseline registered is not an error worth failing on; the
            # dispatch assertion is what decides whether the override took effect.
            continue
    globals()["_OVERRIDE"] = impl
    return active_kernel_name()


# Registering on import is the contract the generated override module relies on.
REGISTERED_AS = register()
