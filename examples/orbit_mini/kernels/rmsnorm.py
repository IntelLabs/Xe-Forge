"""The hot kernel: RMSNorm (plan §15.2, §12.4, §12.6, §12.7, §12.8).

This is the kernel the reference pipeline is meant to pick up first, and it is
built to be hard to extract correctly. Four separate traps live here:

1. **Split device-helper closure.** The kernel body calls three ``@triton.jit``
   helpers that live in three different modules, one of them reached only
   through an alias re-export (:mod:`device_ops`) and one only transitively
   (:mod:`helpers_c`). See §12.6 step 2.
2. **Autotune.** :data:`RMSNORM_CONFIGS` is a four-point search space. Whichever
   config wins has to be captured and pinned into the bundle (§12.7), or
   baseline and candidate are measured under different configurations. When
   Triton is absent, :func:`select_config` is the statically inspectable
   equivalent and it picks deterministically.
3. **Heuristics closing over module state.** :func:`_num_stages_hint` reads the
   module-level :data:`SPLIT_THRESHOLD_ELEMS`; §12.6 step 4 exists because this
   pattern is everywhere and the constant is easy to drop.
4. **A data dependency.** The launch wrapper reads ``tuned_configs.json``
   (§12.8) for block size and epsilon. There is no in-code default.

Three implementations sit behind one wrapper — a Triton kernel, a hand-written
SYCL dispatcher op, and plain torch — which is also the §11 language taxonomy in
miniature. Only the torch path runs on CPU-only CI; the other two are guarded by
availability checks and are the parts that need silicon.

No ``from __future__ import annotations``: Triton needs real annotation objects
to detect ``tl.constexpr`` parameters.
"""

import torch

from . import sycl_op
from .device_ops import weighted_scale as _weighted_scale_alias
from .helpers_a import RMS_EPS_DEFAULT, _rms_scale, _sum_of_squares
from .triton_compat import HAS_TRITON, KernelConfig, autotune, heuristics, jit, next_power_of_2, tl
from .tuned import TunedEntry, lookup

# ---------------------------------------------------------------------------
# Autotune search space and heuristics (§12.7)
# ---------------------------------------------------------------------------

#: Module-level constant that the heuristics callable below closes over. §12.6
#: step 4: "heuristics lambdas frequently close over module state, which must
#: come along". Drop this and the extracted kernel compiles with a different
#: pipeline depth than the one that was measured.
SPLIT_THRESHOLD_ELEMS: int = 4096

#: The autotune search space, as one statically inspectable module-level literal.
#: §12.7 requires the winner to be recorded and pinned; §12.12 step 2 requires
#: the pinned specialization to match the intercepted launch.
RMSNORM_CONFIGS: list[KernelConfig] = [
    KernelConfig({"BLOCK_N": 64}, num_warps=2, num_stages=2),
    KernelConfig({"BLOCK_N": 128}, num_warps=4, num_stages=2),
    KernelConfig({"BLOCK_N": 256}, num_warps=8, num_stages=3),
    KernelConfig({"BLOCK_N": 512}, num_warps=8, num_stages=4),
]


def _num_stages_hint(args: dict) -> int:
    """Pipeline depth hint.

    Closes over the module-level :data:`SPLIT_THRESHOLD_ELEMS`. Named rather
    than a lambda only so the failure is legible when the constant goes missing.
    """
    return 4 if args["N_COLS"] * args["BLOCK_N"] > SPLIT_THRESHOLD_ELEMS else 2


#: Heuristics table. The lambda closes over module state too, on purpose.
RMSNORM_HEURISTICS = {
    "EVEN_N": lambda args: args["N_COLS"] % args["BLOCK_N"] == 0,
    "NUM_STAGES_HINT": _num_stages_hint,
    "USE_SPLIT_REDUCE": lambda args: args["N_COLS"] > SPLIT_THRESHOLD_ELEMS,
}


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@autotune(configs=RMSNORM_CONFIGS, key=["N_COLS"])
@heuristics(RMSNORM_HEURISTICS)
@jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    stride_x_col,
    stride_out_row,
    N_COLS,
    eps,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    NUM_STAGES_HINT: tl.constexpr,
    USE_SPLIT_REDUCE: tl.constexpr,
):
    """Row-wise RMSNorm.

    Note ``stride_x_col``: this kernel does **not** assume a contiguous input,
    because ``orbit_mini`` deliberately feeds it a transposed view (§15.2). A
    synthetic-input reconstruction that allocates a fresh contiguous tensor
    produces a launch whose strides do not match the intercepted record, and
    §12.12 step 2 must catch that.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N_COLS

    x_ptrs = x_ptr + row * stride_x_row + cols * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    sum_sq = _sum_of_squares(x, axis=0)
    scale = _rms_scale(sum_sq, N_COLS, eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Alias hop: `_weighted_scale_alias` is `device_ops.weighted_scale`, which is
    # `helpers_b._weighted_scale`, which calls `helpers_c._clamp_to_finite`.
    out = _weighted_scale_alias(x, w, scale)

    tl.store(out_ptr + row * stride_out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Config selection when Triton's autotuner is not available
# ---------------------------------------------------------------------------


def select_config(n_cols: int, tuned: TunedEntry) -> KernelConfig:
    """Pick the winning config deterministically.

    Stands in for Triton's autotuner on machines without Triton, so the
    "which config actually ran?" question in §12.7 has an answer on CPU-only CI
    as well as on silicon. Reads ``block_n`` from the tuned-config JSON, so the
    data dependency participates in the choice — capture the code without the
    data and you pin a different config than the one that ran.
    """
    want_block = max(next_power_of_2(n_cols), tuned.block_n)
    for config in RMSNORM_CONFIGS:
        if config.kwargs["BLOCK_N"] >= want_block:
            return config
    return RMSNORM_CONFIGS[-1]


def evaluate_heuristics(n_cols: int, config: KernelConfig) -> dict:
    """Evaluate :data:`RMSNORM_HEURISTICS` for a given specialization.

    Exposed so the launch-record match in §12.12 step 2 can compare heuristic
    values, not just autotune keys.
    """
    args = {"N_COLS": n_cols, "BLOCK_N": config.kwargs["BLOCK_N"]}
    return {name: fn(args) for name, fn in RMSNORM_HEURISTICS.items()}


# ---------------------------------------------------------------------------
# Launch wrapper
# ---------------------------------------------------------------------------

#: Last launch record (§12.4). The pipeline intercepts this for real; keeping a
#: copy here lets the CPU-only rig assert on grid, constexprs and — importantly —
#: input strides without a profiler.
LAST_LAUNCH: dict = {}


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float | None = None,
    allow_sycl: bool = True,
) -> torch.Tensor:
    """RMSNorm over the last dimension of ``x``.

    Dispatch order — Triton, then the SYCL dispatcher op, then torch — mirrors
    the §11 provider taxonomy. Every step is guarded by an availability check so
    the CPU-only path is always reachable.

    ``x`` is *not* made contiguous. That is deliberate: ``orbit_mini`` feeds a
    transposed view in, and forcing contiguity here would hide the very stride
    mismatch the capture stage (§16.4 row 6) has to notice.
    """
    tuned = lookup(x.device)
    if eps is None:
        eps = tuned.rms_eps if tuned.rms_eps > 0 else RMS_EPS_DEFAULT

    n_cols = x.shape[-1]
    rows = x.numel() // n_cols
    config = select_config(n_cols, tuned)

    LAST_LAUNCH.clear()
    LAST_LAUNCH.update(
        {
            "kernel": "orbit_mini::_rmsnorm_kernel",
            "grid": (rows,),
            "n_cols": n_cols,
            "eps": eps,
            "config": config.describe(),
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "constexprs": evaluate_heuristics(n_cols, config),
            "tuned_entry": tuned.device_key,
            # Recorded before any reshape. A synthetic-input reconstruction that
            # allocates a fresh contiguous tensor will not reproduce these, and
            # the launch-record match in §12.12 step 2 has to notice.
            "input_contiguous": bool(x.is_contiguous()),
            "input_strides": tuple(x.stride()),
            "input_shape": tuple(x.shape),
        }
    )

    if HAS_TRITON and x.device.type in ("cuda", "xpu"):
        # The Triton path needs a 2-D view; a non-contiguous input forces a
        # restride here, which is itself part of the kernel's real behaviour and
        # therefore part of what extraction has to reproduce (§12.6 step 5).
        x2d = x.reshape(rows, n_cols)
        LAST_LAUNCH["backend"] = "triton"
        LAST_LAUNCH["restrided"] = not x.is_contiguous()
        out = torch.empty(
            (rows, n_cols), dtype=x.dtype, device=x.device, memory_format=torch.contiguous_format
        )
        _rmsnorm_kernel[(rows,)](
            x2d,
            weight,
            out,
            x2d.stride(0),
            x2d.stride(1),
            out.stride(0),
            n_cols,
            eps,
        )
        return out.reshape(x.shape)

    if allow_sycl and tuned.prefer_sycl_op and sycl_op.is_available(x.device):
        LAST_LAUNCH["backend"] = "sycl"
        LAST_LAUNCH["restrided"] = False
        return sycl_op.rmsnorm(x, weight, eps)

    # Torch handles arbitrary strides, so the fallback consumes `x` as it was
    # handed over — no `.contiguous()` anywhere on this path.
    LAST_LAUNCH["backend"] = "torch"
    LAST_LAUNCH["restrided"] = False
    return _rms_norm_torch(x, weight, eps, tuned.clamp_limit)


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float, clamp: float
) -> torch.Tensor:
    """Pure-torch reference, numerically matched to the Triton path.

    Mirrors the device-helper chain step for step — sum of squares in fp32
    (``helpers_a``), reciprocal-RMS scale (``helpers_a``), saturate then apply
    the learned weight (``helpers_b`` -> ``helpers_c``) — so a bundle extracted
    from either implementation can be compared against the same reference.
    """
    xf = x.to(torch.float32)
    sum_sq = xf.pow(2).sum(dim=-1, keepdim=True)
    scale = torch.rsqrt(sum_sq / x.shape[-1] + eps)
    normed = torch.clamp(xf * scale, -clamp, clamp)
    return (normed * weight.to(torch.float32)).to(x.dtype)
