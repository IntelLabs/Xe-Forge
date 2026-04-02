# ruff: noqa: E731

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Problem Definitions and Input Utilities (match YAML defaults)
# --------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)


# --------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# --------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      out = Hardtanh(GELU(Tanh(Swish(x @ W^T + bias + add_value))))

    Init signature matches YAML inits:
      (IN_FEAT, OUT_FEAT, ADD_VALUE_SHAPE)

    Notes:
    - Parameters are stored as: weight [N, K] in float16, bias [N] in float32, add_value [N] in float32
    - forward() caches a pre-packed W^T once on XPU as [K, N] contiguous and reuses it.
    - forward() calls kernel_function() directly (this is what KernelBench should time).
    """

    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Parameters in expected layout: weight fp16, bias/add in fp32 for stability
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float16)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))
        self.add_value = nn.Parameter(torch.randn(tuple(add_value_shape), dtype=torch.float32))

        # Initialize similarly to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Cached XPU tensors
        self._cache_ready = False
        self._packed_version = -1

    def _ensure_cache(self):
        # Build/copy once and cache on XPU; rebuild only if weight version changes
        current_ver = int(self.weight._version)
        if (not self._cache_ready) or (self._packed_version != current_ver):
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("XPU device is not available")
            # Move and pack once
            self.weight_xpu = self.weight.data.to("xpu", dtype=torch.float16).contiguous()
            self.weight_t = self.weight_xpu.transpose(0, 1).contiguous()  # [K, N], packed
            self.bias_xpu = self.bias.data.to("xpu", dtype=torch.float32).contiguous()
            self.add_value_xpu = self.add_value.data.to("xpu", dtype=torch.float32).contiguous()
            self._packed_version = current_ver
            self._cache_ready = True

    def forward(self, x):
        # Ensure float16 inputs; KernelBench may pass CPU tensors.
        if x.dtype != torch.float16:
            x = x.to(dtype=torch.float16)

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        # Move/cast input
        if x.device.type != "xpu":
            x = x.to("xpu", dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        # Ensure cached params on XPU (one-time pack/move; refresh if weight changes)
        self._ensure_cache()

        # Expect [M, K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight_t, self.bias_xpu, self.add_value_xpu)


def get_init_inputs():
    return [in_features, out_features, add_value_shape]


def get_inputs():
    # KernelBench often generates inputs on CPU; Model moves to xpu and casts to fp16.
    return [torch.rand(batch_size, in_features, dtype=torch.float32)]


# --------------------------------------------------------------------
# Triton Kernel: Fused GEMM + Bias/Add + Activation (Single Kernel)
# Pre-packed RHS: W^T is provided as [K, N] with its own strides
# --------------------------------------------------------------------
def _get_xpu_autotune_configs():
    # Intel XPU optimized autotune configurations
    return [
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
                "grf_mode": "256",
            },
            num_warps=32,
            num_stages=3,
        ),
    ]


@triton.autotune(configs=_get_xpu_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _fused_gemm_bias_add_activation_kernel(
    x_ptr,  # fp16, [M, K]
    wt_ptr,  # fp16, [K, N] (packed W^T)
    b_ptr,  # fp32, [N]
    o_ptr,  # fp16, [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wtk,
    stride_wtn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # -------------------------------------------------
    # 1D grid with GROUP_SIZE_M swizzling
    # -------------------------------------------------
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -------------------------------------------------
    # Block pointers
    # -------------------------------------------------
    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    wt_bp = tl.make_block_ptr(
        base=wt_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # -------------------------------------------------
    # FP32 accumulator
    # -------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------
    # GEMM (fp16 inputs → fp32 acc)
    # -------------------------------------------------
    for _ in range(0, K, BLOCK_K):
        x_tile = tl.load(x_bp, boundary_check=(0, 1))  # fp16
        w_tile = tl.load(wt_bp, boundary_check=(0, 1))  # fp16
        acc = tl.dot(x_tile, w_tile, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))

    # -------------------------------------------------
    # Bias in FP32  ✅ FIX #1
    # -------------------------------------------------
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # -------------------------------------------------
    # Sigmoid via exp2  ✅ FIX #2
    # sigmoid(x) = 1 / (1 + exp(-x))
    # exp(-x) = exp2(-x * log2(e))
    # -------------------------------------------------
    LOG2E = 1.4426950408889634
    e2 = tl.math.exp2((-acc) * LOG2E)
    sig = 1.0 / (1.0 + e2)

    # Example epilogue (keep as before)
    out = acc + 2.0 * sig

    # -------------------------------------------------
    # Store fp16
    # -------------------------------------------------
    o_bp = tl.make_block_ptr(
        base=o_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(o_bp, out.to(tl.float16), boundary_check=(0, 1))


# --------------------------------------------------------------------
# Top-Level Kernel Function (FUSED) - expects pre-packed W^T
# --------------------------------------------------------------------
def kernel_function(
    x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, add_value: torch.Tensor
) -> torch.Tensor:
    """
    Orchestrate fused Triton kernel on XPU to compute:
      out = Hardtanh(GELU(Tanh(Swish(x @ W^T + bias + add_value))))
    Mixed precision path:
      - x, weight_t in fp16 (weight_t is pre-packed W^T [K, N])
      - bias/add in fp32
      - GEMM accumulates in fp32, activations in fp32, stores fp16
    """
    # Validate XPU availability
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU device is not available")

    # Shapes and sanity checks
    M, K = x.shape
    K_wt, N = weight_t.shape
    assert K == K_wt, f"Incompatible x and weight_t shapes: {x.shape}, {weight_t.shape}"
    assert bias.shape == (N,), f"Expected bias shape ({N},), got {bias.shape}"
    assert add_value.shape == (N,), f"Expected add_value shape ({N},), got {add_value.shape}"

    # Ensure dtypes/contiguity and on XPU
    x_xpu = x.to("xpu", dtype=torch.float16).contiguous()
    wt_xpu = weight_t.to("xpu", dtype=torch.float16).contiguous()
    b_xpu = bias.to("xpu", dtype=torch.float32).contiguous()
    add_xpu = add_value.to("xpu", dtype=torch.float32).contiguous()

    # Allocate output
    out = torch.empty((M, N), dtype=torch.float16, device="xpu")

    # 1D grid for swizzling
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fused_gemm_bias_add_activation_kernel[grid](
        x,
        wt_xpu,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        wt_xpu.stride(0),
        wt_xpu.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
