# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 4096
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


def get_inputs(self=None):
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs(self=None):                          # ← accept optional self
    return [in_features, out_features, list(scale_shape)]  # ← list avoids tuple-int ambiguity


def _configs():
    return [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "grf_mode": "256"}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "grf_mode": "256"}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "grf_mode": "256"}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "grf_mode": "256"}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "grf_mode": "256"}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "grf_mode": "256"}, num_warps=8,  num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _fused_gemm_scale_bn_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, gamma_ptr, beta_ptr, rm_ptr, rv_ptr, y_ptr,
    M, N, K,
    eps: tl.constexpr,
    stride_xm, stride_xk, stride_wm, stride_wk,
    stride_b, stride_s, stride_gamma, stride_beta,
    stride_rm, stride_rv, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        x_ptrs = x_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_ptr + n_offsets[None, :] * stride_wm + k_offsets[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    bias  = tl.load(b_ptr     + n_offsets * stride_b,     mask=n_mask, other=0.0).to(tl.float32)
    scale = tl.load(s_ptr     + n_offsets * stride_s,     mask=n_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + n_offsets * stride_gamma, mask=n_mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + n_offsets * stride_beta,  mask=n_mask, other=0.0).to(tl.float32)
    rm    = tl.load(rm_ptr    + n_offsets * stride_rm,    mask=n_mask, other=0.0)
    rv    = tl.load(rv_ptr    + n_offsets * stride_rv,    mask=n_mask, other=1.0)

    acc = acc + bias[None, :]
    acc = acc * scale[None, :]

    inv_std = 1.0 / tl.sqrt(rv + eps)
    acc = (acc - rm[None, :]) * inv_std[None, :]
    acc = acc * gamma[None, :] + beta[None, :]

    y_ptrs = y_ptr + m_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def kernel_function(x, weight, bias, scale, bn_weight, bn_bias, running_mean, running_var, eps):
    assert x.device.type == "xpu"
    assert x.dtype == torch.float16 and weight.dtype == torch.float16

    if not x.is_contiguous():      x = x.contiguous()
    if not weight.is_contiguous(): weight = weight.contiguous()

    for t, name in [(bias, "bias"), (scale, "scale"), (bn_weight, "bn_weight"), (bn_bias, "bn_bias")]:
        assert t.device.type == "xpu"
        assert t.dtype == torch.float16, f"{name} must be float16"

    for t, name in [(running_mean, "running_mean"), (running_var, "running_var")]:
        assert t.device.type == "xpu"
        assert t.dtype == torch.float32, f"{name} must be float32"

    M, K = x.shape
    N, Kw = weight.shape
    assert K == Kw

    y = torch.empty((M, N), device="xpu", dtype=torch.float32)

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _fused_gemm_scale_bn_kernel[grid](
        x, weight, bias, scale, bn_weight, bn_bias, running_mean, running_var, y,
        M, N, K, eps=eps,
        stride_xm=x.stride(0),           stride_xk=x.stride(1),
        stride_wm=weight.stride(0),       stride_wk=weight.stride(1),
        stride_b=bias.stride(0),          stride_s=scale.stride(0),
        stride_gamma=bn_weight.stride(0), stride_beta=bn_bias.stride(0),
        stride_rm=running_mean.stride(0), stride_rv=running_var.stride(0),
        stride_ym=y.stride(0),            stride_yn=y.stride(1),
    )
    return y


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self.eps          = float(eps)
        self.momentum     = float(momentum)
        # accept int or sequence for scale_shape
        self.scale_shape  = (
            (int(scale_shape),) if isinstance(scale_shape, int) else tuple(scale_shape)
        )

        self.weight    = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=torch.float16))
        self.bias      = nn.Parameter(torch.empty(self.out_features,  dtype=torch.float16))
        self.scale     = nn.Parameter(torch.randn(self.scale_shape,   dtype=torch.float16))
        self.bn_weight = nn.Parameter(torch.ones(self.out_features,   dtype=torch.float16))
        self.bn_bias   = nn.Parameter(torch.zeros(self.out_features,  dtype=torch.float16))

        self.register_buffer("running_mean", torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("running_var",  torch.ones(self.out_features,  dtype=torch.float32))

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        self._moved_to_xpu = False

    def _restore_bn_buffers_fp32(self):
        self.running_mean = self.running_mean.float()
        self.running_var  = self.running_var.float()

    def half(self):
        super().half()
        self._restore_bn_buffers_fp32()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._restore_bn_buffers_fp32()
        return self

    def _move_params_once(self):
        if self._moved_to_xpu:
            return
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")
        dev = torch.device("xpu")
        with torch.no_grad():
            for p in (self.weight, self.bias, self.scale, self.bn_weight, self.bn_bias):
                p.data = p.data.to(dev)
            self.running_mean = self.running_mean.to(dev, dtype=torch.float32)
            self.running_var  = self.running_var.to(dev,  dtype=torch.float32)
        self._moved_to_xpu = True

    def forward(self, x):
        if x.dtype != torch.float16:
            x = x.half()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        self._move_params_once()

        if x.device.type != "xpu":
            x = x.to("xpu")
        if not x.is_contiguous():
            x = x.contiguous()

        return kernel_function(
            x, self.weight, self.bias, self.scale,
            self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var, self.eps,
        )