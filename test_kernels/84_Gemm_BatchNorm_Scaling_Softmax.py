# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
EPS = 1e-5


@triton.jit
def _gemm_bn_scale_kernel(
    a_ptr, w_ptr, bias_ptr, gamma_ptr, beta_ptr, rm_ptr, rv_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn,
    stride_b, stride_g, stride_be, stride_rm, stride_rv, stride_sc,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)
        acc = tl.dot(a, w, acc)

    b  = tl.load(bias_ptr  + offs_n * stride_b,  mask=mask_n, other=0.0).to(tl.float32)
    # ← explicit .to(tl.float32) on running stats — defensive against fp16 buffers
    rm = tl.load(rm_ptr    + offs_n * stride_rm,  mask=mask_n, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr    + offs_n * stride_rv,  mask=mask_n, other=1.0).to(tl.float32)
    g  = tl.load(gamma_ptr + offs_n * stride_g,   mask=mask_n, other=1.0).to(tl.float32)
    be = tl.load(beta_ptr  + offs_n * stride_be,  mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + b[None, :]
    invstd = 1.0 / tl.sqrt(rv + EPS)
    acc = (acc - rm[None, :]) * invstd[None, :] * g[None, :] + be[None, :]

    s = tl.load(scale_ptr + 0 * stride_sc, mask=True, other=1.0).to(tl.float32)
    acc = acc * s

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _softmax_dim1_kernel(input_ptr, output_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(input_ptr + row * N + offs, mask=mask, other=-1e20)
    m = tl.max(x, axis=0)
    nb = tl.cdiv(N, BLOCK)
    for i in range(1, nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=-1e20)
        m = tl.maximum(m, tl.max(xi, axis=0))
    sum_exp = tl.zeros([], dtype=tl.float32)
    for i in range(nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=0.0)
        sum_exp += tl.sum(tl.exp(xi - m), axis=0)
    for i in range(nb):
        offs_i = i * BLOCK + offs
        mask_i = offs_i < N
        xi = tl.load(input_ptr + row * N + offs_i, mask=mask_i, other=0.0)
        y = tl.exp(xi - m) / sum_exp
        tl.store(output_ptr + row * N + offs_i, y, mask=mask_i)


def kernel_function(
    input_tensor, linear_weight, linear_bias,
    bn_weight, bn_bias, bn_running_mean, bn_running_var, scale,
):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU backend is not available")
    device_xpu = torch.device("xpu")
    orig_device = input_tensor.device

    x  = input_tensor.to(device_xpu)
    w  = linear_weight.t().contiguous().to(device_xpu)
    b  = linear_bias.to(device_xpu).contiguous()
    g  = bn_weight.to(device_xpu).contiguous()
    be = bn_bias.to(device_xpu).contiguous()
    # ← force fp32 regardless of what arrived — defensive against .half() pollution
    rm = bn_running_mean.to(device_xpu, dtype=torch.float32).contiguous()
    rv = bn_running_var.to(device_xpu,  dtype=torch.float32).contiguous()
    s  = scale.to(device_xpu).contiguous().view(-1)

    assert x.dtype == torch.float16, "input must be float16"
    assert w.dtype == torch.float16, "weight must be float16"

    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    y   = torch.empty((M, N), device=device_xpu, dtype=torch.float32)
    out = torch.empty_like(y)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_bn_scale_kernel[grid](
        x, w, b, g, be, rm, rv, s, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0), g.stride(0), be.stride(0),
        rm.stride(0), rv.stride(0), s.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K, EPS,
    )

    _softmax_dim1_kernel[(M,)](y, out, M, N, BLOCK_N)

    if orig_device != device_xpu:
        return out.to(orig_device)
    return out


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs(self=None):
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs(self=None):
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


class Model(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self.bn_eps       = float(bn_eps)
        self.bn_momentum  = float(bn_momentum)
        self.scale_shape  = (
            tuple(scale_shape) if isinstance(scale_shape, (list, tuple)) else (int(scale_shape),)
        )

        self.gemm = nn.Linear(self.in_features, self.out_features, bias=True)
        self.gemm.weight = nn.Parameter(self.gemm.weight.to(torch.float16))
        self.gemm.bias   = nn.Parameter(self.gemm.bias.to(torch.float16))

        self.bn_weight = nn.Parameter(torch.ones(self.out_features,  dtype=torch.float16))
        self.bn_bias   = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float16))
        self.register_buffer("bn_running_mean", torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("bn_running_var",  torch.ones(self.out_features,  dtype=torch.float32))

        self.scale = nn.Parameter(torch.ones(self.scale_shape, dtype=torch.float16))

    def _restore_bn_buffers_fp32(self):
        self.bn_running_mean = self.bn_running_mean.float()
        self.bn_running_var  = self.bn_running_var.float()

    def half(self):
        super().half()
        self._restore_bn_buffers_fp32()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._restore_bn_buffers_fp32()
        return self

    def forward(self, x):
        if x.dtype != torch.float16:
            x = x.half()

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