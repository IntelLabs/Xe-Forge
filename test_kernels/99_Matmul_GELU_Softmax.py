# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    tmp_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_b,
    stride_tm,
    stride_tn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # .to(tl.float32) handles both fp16 and fp32 inputs transparently
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    b_ptrs = b_ptr + offs_n * stride_b
    b = tl.load(b_ptrs, mask=mask_n, other=0.0)
    acc += b.to(tl.float32)[None, :]  # bias may also be fp16

    # GELU in fp32 — unchanged
    S2_PI = 0.7978845608028654
    x_f = acc
    x3 = x_f * x_f * x_f
    inner = x_f + 0.044715 * x3
    z = S2_PI * inner
    ez = tl.exp(z)
    ez_neg = tl.exp(-z)
    tanh_z = (ez - ez_neg) / (ez + ez_neg)
    y = 0.5 * x_f * (1.0 + tanh_z)

    # tmp is fp32 regardless of input dtype
    out_ptrs = tmp_ptr + offs_m[:, None] * stride_tm + offs_n[None, :] * stride_tn
    tl.store(out_ptrs, y, mask=(mask_m[:, None] & mask_n[None, :]))


# softmax kernel is UNCHANGED — it only ever touches fp32 tmp/out
@triton.jit
def _softmax_kernel(
    tmp_ptr,
    out_ptr,
    M,
    N,
    stride_tm,
    stride_tn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    idx_n = pid_m * BLOCK_N + offs_n
    mask_n = idx_n < N

    max_val = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        idx = start + offs_n
        m = idx < N
        ptrs = tmp_ptr + pid_m * stride_tm + idx * stride_tn
        x = tl.load(ptrs, mask=m, other=-float("inf"))
        current_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, current_max)

    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        idx = start + offs_n
        m = idx < N
        ptrs = tmp_ptr + pid_m * stride_tm + idx * stride_tn
        x = tl.load(ptrs, mask=m, other=0.0)
        e = tl.exp(x - max_val)
        sum_val += tl.sum(e, axis=0)

    ptrs = tmp_ptr + pid_m * stride_tm + idx_n * stride_tn
    x = tl.load(ptrs, mask=mask_n, other=0.0)
    e = tl.exp(x - max_val)
    out = e / sum_val

    ptr_out = out_ptr + pid_m * stride_om + idx_n * stride_on
    tl.store(ptr_out, out, mask=mask_n)


def kernel_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU device is required"
    assert input_tensor.device.type == "xpu", "input_tensor must be on xpu"
    assert input_tensor.dtype == torch.float16, "input_tensor must be float16"  # ← float16
    assert weight.dtype == torch.float16, "weight must be float16"  # ← float16
    # bias is kept fp32 — avoids precision loss on the additive term
    assert bias.dtype == torch.float16, "bias must be float16"

    M, K = input_tensor.shape
    K2, N = weight.shape
    assert K == K2
    assert bias.numel() == N

    # tmp and out stay fp32 — softmax kernel is unchanged
    tmp = torch.empty((M, N), dtype=torch.float32, device="xpu")
    out = torch.empty((M, N), dtype=torch.float32, device="xpu")

    sxm, sxk = input_tensor.stride()
    swk, swn = weight.stride()
    (sb,) = bias.stride()
    stm, stn = tmp.stride()
    som, son = out.stride()

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_gelu_kernel[grid1](
        input_tensor,
        weight,
        bias,
        tmp,
        M,
        N,
        K,
        sxm,
        sxk,
        swk,
        swn,
        sb,
        stm,
        stn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    SM_BLOCK = 128
    _softmax_kernel[(M,)](
        tmp,
        out,
        M,
        N,
        stm,
        stn,
        som,
        son,
        BLOCK_N=SM_BLOCK,
    )

    torch.xpu.synchronize()
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight_t = nn.Parameter(
            torch.empty(self.in_features, self.out_features, dtype=torch.float16)  # ← float16
        )
        # bias stays fp32 — higher precision for the additive offset
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float16))

        nn.init.kaiming_uniform_(self.weight_t.t(), a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dtype != torch.float16:  # ← float16
            x = x.half()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU unavailable")

        if x.device.type != "xpu":
            x = x.to("xpu")

        if self.weight_t.device.type != "xpu":
            self.weight_t.data = self.weight_t.data.to("xpu")
        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu")

        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight_t, self.bias)


batch_size = 10
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]  # ← float16


def get_init_inputs():
    return [in_features, out_features]
