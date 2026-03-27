# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton kernels for Intel XPU (DO NOT MODIFY)
# -----------------------------------------------------------------------------


@triton.jit
def _linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wm,
    stride_wk,
    stride_om,
    stride_on,
    stride_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for kt in range(num_k_tiles):
        offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < K
        # load x_sub: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_sub = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
        # load w_sub: [BLOCK_K, BLOCK_N], W is [N, K]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w_sub = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(x_sub, w_sub)
    # add bias
    bias_vals = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :]
    # store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def _pool_sum_scale_kernel(x_ptr, out_ptr, rows, cols, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    row_start = pid * cols
    num_pairs = cols // 2
    sum_val = tl.zeros((), dtype=tl.float32)
    # iterate over pooling pairs
    for pair_start in range(0, num_pairs, BLOCK_SIZE):
        idx = pair_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        offs0 = row_start + idx * 2
        offs1 = offs0 + 1
        mask = idx < num_pairs
        x0 = tl.load(x_ptr + offs0, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + offs1, mask=mask, other=0.0)
        m = tl.maximum(x0, x1)  # max pool
        s = tl.sum(m, axis=0)  # reduce
        sum_val += s
    # scale
    result = sum_val * scale
    tl.store(out_ptr + pid, result, mask=pid < rows)


# -----------------------------------------------------------------------------
# Top-level kernel_function: matmul -> maxpool+sum+scale (DO NOT MODIFY)
# -----------------------------------------------------------------------------


def kernel_function(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    Args:
      x: float32 tensor on 'xpu' of shape [M, K]
      weight: float32 tensor on 'xpu' of shape [N, K]
      bias: float32 tensor on 'xpu' of shape [N]
      scale: float scalar (float32)
    Returns:
      out: float32 tensor on 'xpu' of shape [M], where each row is
           scale * sum over max-pool 1d with kernel=2,stride=2 of (x @ weight^T + bias)
    """
    # ensure XPU device
    if not (hasattr(torch, "xpu") and x.device.type == "xpu"):
        raise RuntimeError("Input tensor must be on Intel XPU ('xpu')")
    # shapes
    M, K = x.shape
    N, K_w = weight.shape
    assert K_w == K, "Weight inner dim must match input"
    assert bias.shape[0] == N, "Bias length must match output dim"
    # make contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    # intermediate output of linear: [M, N]
    out_lin = torch.empty((M, N), device="xpu", dtype=torch.float32)
    # strides
    sxm, sxk = x.stride(0), x.stride(1)
    swm, swk = weight.stride(0), weight.stride(1)
    som, son = out_lin.stride(0), out_lin.stride(1)
    sb = bias.stride(0)
    # launch linear
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
    grid_lin = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid_lin](
        x,
        weight,
        bias,
        out_lin,
        M,
        N,
        K,
        sxm,
        sxk,
        swm,
        swk,
        som,
        son,
        sb,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    # max-pool+sum+scale
    out_final = torch.empty((M,), device="xpu", dtype=torch.float32)
    BLOCK_SIZE = 512
    grid_pool = (triton.cdiv(M, 1),)
    _pool_sum_scale_kernel[grid_pool](out_lin, out_final, M, N, scale, BLOCK_SIZE=BLOCK_SIZE)
    torch.xpu.synchronize()
    return out_final


# -----------------------------------------------------------------------------
# KernelBench Model: ONLY place we "adapt" to the harness
# -----------------------------------------------------------------------------


class Model(nn.Module):
    """
    KernelBench model wrapper:
      y = kernel_function(X, W, b, scale)

    Init signature matches YAML inits:
      (IN_FEAT, OUT_FEAT, KERNEL_SIZE, SCALE_FACTOR)

    Notes:
    - kernel_size is accepted for spec compatibility, but Triton kernel assumes pooling by pairs (kernel=2,stride=2).
    - OUT_FEAT must be even.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int, scale_factor: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)

        if (self.out_features % 2) != 0:
            raise ValueError(f"OUT_FEAT (N={self.out_features}) must be even (pool pairs of 2).")

        # Parameters in the exact shapes Triton expects: weight [N,K], bias [N]
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))

        # Reasonable init (similar spirit to nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KernelBench may pass CPU tensors; enforce float32 + xpu
        if x.dtype != torch.float32:
            x = x.float()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU not available")

        if x.device.type != "xpu":
            x = x.to("xpu")

        # Ensure params live on xpu (KernelBench sometimes constructs model on CPU)
        if self.weight.device.type != "xpu":
            self.weight.data = self.weight.data.to("xpu")
        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu")

        # x must be [M,K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight, self.bias, self.scale_factor)


# -----------------------------------------------------------------------------
# KernelBench input generators
# -----------------------------------------------------------------------------
# These should match the YAML dims below.
# You can tweak these defaults, but keep them consistent with YAML.


def get_init_inputs():
    # IN_FEAT подтверждает K, OUT_FEAT подтверждает N
    IN_FEAT = 32768
    OUT_FEAT = 32768  # must be even
    KERNEL_SIZE = 2  # accepted but Triton pooling is pairwise
    SCALE_FACTOR = 0.5
    return [IN_FEAT, OUT_FEAT, KERNEL_SIZE, SCALE_FACTOR]


def get_inputs():
    # X shape: [BATCH, IN_FEAT]
    BATCH = 128
    IN_FEAT, _, _, _ = get_init_inputs()
    # KernelBench usually generates on CPU; Model moves to xpu.
    X = torch.rand((BATCH, IN_FEAT), dtype=torch.float32)
    return [X]
