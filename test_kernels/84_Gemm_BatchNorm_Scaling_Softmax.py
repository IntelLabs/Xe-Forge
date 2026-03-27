# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel: GEMM -> Bias -> BatchNorm (inference) -> Scale (UNCHANGED)
# -------------------------------------------------------------------------
# Reduced block sizes to fit shared memory constraints on XPU
BLOCK_M = 64  # rows per block
BLOCK_N = 128  # cols per block
BLOCK_K = 64  # K inner dim per block
EPS = 1e-5  # BN epsilon


@triton.jit
def _gemm_bn_scale_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    gamma_ptr,
    beta_ptr,
    rm_ptr,
    rv_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_b,
    stride_g,
    stride_be,
    stride_rm,
    stride_rv,
    stride_sc,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    offs_m = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # GEMM loop
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)
        w = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)
        acc = tl.dot(a, w, acc)

    # bias
    b = tl.load(bias_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    # batch-norm (inference): (x - mean) * (1 / sqrt(var + eps)) * gamma + beta
    rm = tl.load(rm_ptr + offs_n * stride_rm, mask=mask_n, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + offs_n * stride_rv, mask=mask_n, other=0.0).to(tl.float32)
    g = tl.load(gamma_ptr + offs_n * stride_g, mask=mask_n, other=1.0).to(tl.float32)
    be = tl.load(beta_ptr + offs_n * stride_be, mask=mask_n, other=0.0).to(tl.float32)
    invstd = 1.0 / tl.sqrt(rv + EPS)
    acc = (acc - rm[None, :]) * invstd[None, :] * g[None, :] + be[None, :]

    # final scale (scalar or small vector)
    s = tl.load(scale_ptr + 0 * stride_sc, mask=True, other=1.0).to(tl.float32)
    acc = acc * s

    # store
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


# -------------------------------------------------------------------------
# Triton kernel: Softmax over dim=1 (UNCHANGED)
# -------------------------------------------------------------------------
@triton.jit
def _softmax_dim1_kernel(input_ptr, output_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # phase 1: compute max
    idx = row * N + offs
    x = tl.load(input_ptr + idx, mask=mask, other=-1e20)
    m = tl.max(x, axis=0)
    nb = tl.cdiv(N, BLOCK)
    for i in range(1, nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=-1e20)
        m = tl.maximum(m, tl.max(xi, axis=0))
    # phase 2: sum of exp
    sum_exp = tl.zeros([], dtype=tl.float32)
    for i in range(nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=0.0)
        sum_exp += tl.sum(tl.exp(xi - m), axis=0)
    # phase 3: normalize and store
    for i in range(nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=0.0)
        y = tl.exp(xi - m) / sum_exp
        tl.store(output_ptr + row * N + offs_i, y, mask=mask_i)


# -------------------------------------------------------------------------
# Top-level wrapper: fuse GEMM+BN+Scale then Softmax (UNCHANGED)
# -------------------------------------------------------------------------
def kernel_function(
    input_tensor: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # Check XPU availability
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU backend is not available")
    device_xpu = torch.device("xpu")
    orig_device = input_tensor.device

    # Move tensors to XPU
    x = input_tensor.to(device_xpu)
    # reshape weight to [K, N]
    w = linear_weight.t().contiguous().to(device_xpu)
    b = linear_bias.to(device_xpu).contiguous()
    g = bn_weight.to(device_xpu).contiguous()
    be = bn_bias.to(device_xpu).contiguous()
    rm = bn_running_mean.to(device_xpu).contiguous()
    rv = bn_running_var.to(device_xpu).contiguous()
    s = scale.to(device_xpu).contiguous().view(-1)

    # dims
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w, "GEMM dimension mismatch"

    # allocate GEMM+BN+Scale output
    y = torch.empty((M, N), device=device_xpu, dtype=torch.float32)

    # strides
    stride_am, stride_ak = x.stride(0), x.stride(1)
    stride_wk, stride_wn = w.stride(0), w.stride(1)
    stride_b = b.stride(0)
    stride_g = g.stride(0)
    stride_be = be.stride(0)
    stride_rm = rm.stride(0)
    stride_rv = rv.stride(0)
    stride_sc = s.stride(0)
    stride_om, stride_on = y.stride(0), y.stride(1)

    # launch GEMM+BN+Scale kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_bn_scale_kernel[grid](
        x,
        w,
        b,
        g,
        be,
        rm,
        rv,
        s,
        y,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wk,
        stride_wn,
        stride_b,
        stride_g,
        stride_be,
        stride_rm,
        stride_rv,
        stride_sc,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EPS,
    )

    # allocate Softmax output
    out = torch.empty_like(y)
    # launch Softmax kernel
    _softmax_dim1_kernel[(M,)](y, out, M, N, BLOCK_N)

    # move back to original device if needed
    if orig_device != device_xpu:
        return out.to(orig_device)
    return out


# -------------------------------------------------------------------------
# KernelBench inputs (match YAML below)
# -------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    # KernelBench often generates on CPU; Model/kernel_function moves to xpu.
    return [torch.rand(batch_size, in_features, dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


# -------------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# -------------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      x -> Linear -> BatchNorm1d (inference) -> scale * x -> Softmax(dim=1)

    Init signature matches YAML inits:
      (IN_FEAT, OUT_FEAT, BN_EPS, BN_MOMENTUM, SCALE_SHAPE)

    Notes:
    - kernel_function uses BN running stats (inference-style); so Model stays in eval mode in bench.
    - scale is treated by kernel as a scalar (loads scale_ptr[0]); so keep SCALE_SHAPE=[1] for exactness.
    """

    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)
        self.scale_shape = (
            tuple(scale_shape) if isinstance(scale_shape, (list, tuple)) else (int(scale_shape),)
        )

        # Linear params in the shapes PyTorch uses (kernel_function accepts these and transposes internally)
        self.gemm = nn.Linear(self.in_features, self.out_features, bias=True)

        # BN params/stats (kernel_function consumes weight/bias + running_mean/var)
        self.bn_weight = nn.Parameter(torch.ones(self.out_features, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("bn_running_mean", torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("bn_running_var", torch.ones(self.out_features, dtype=torch.float32))

        # scale parameter (scalar)
        self.scale = nn.Parameter(torch.ones(self.scale_shape, dtype=torch.float32))

    def forward(self, x):
        # Keep float32
        if x.dtype != torch.float32:
            x = x.float()

        return kernel_function(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.scale,
        )
