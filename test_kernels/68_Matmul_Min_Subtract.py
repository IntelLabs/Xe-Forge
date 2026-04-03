# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl


class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # constant stays fp32 scalar — passed as float() into kernel, dtype-agnostic
        self.constant = nn.Parameter(torch.tensor(float(constant), dtype=torch.float32))

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float16)  # ← float16
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float16))  # ← float16

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dtype != torch.float16:  # ← float16
            x = x.half()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU is not available")

        if x.device.type != "xpu":
            x = x.to("xpu")

        if self.weight.device.type != "xpu":
            self.weight.data = self.weight.data.to("xpu")
        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu")
        if self.constant.device.type != "xpu":
            self.constant.data = self.constant.data.to("xpu")

        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight, self.bias, self.constant)


def get_init_inputs():
    return [16384, 16384, 2.0]


def get_inputs():
    BATCH = 128
    IN_FEAT, _, _ = get_init_inputs()
    return [torch.rand((BATCH, IN_FEAT), dtype=torch.float16)]  # ← float16


@triton.jit
def _linear_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_b,
    stride_om, stride_ok,
    constant,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_blk = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_blk = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        acc += tl.dot(x_blk.to(tl.float32), w_blk.to(tl.float32))  # ← upcast fp16

    b_val = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)  # ← upcast
    acc = acc + b_val[None, :]

    # constant is already a Python float (fp32) — no change needed
    acc = tl.minimum(acc, constant)
    acc = acc - constant

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(input_tensor, linear_weight, linear_bias, constant):
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    assert input_tensor.device.type == "xpu"
    assert linear_weight.device.type == "xpu"
    assert linear_bias.device.type == "xpu"
    assert input_tensor.dtype == torch.float16, "input must be float16"   # ← float16
    assert linear_weight.dtype == torch.float16, "weight must be float16" # ← float16
    assert linear_bias.dtype == torch.float16, "bias must be float16"     # ← float16

    if isinstance(constant, torch.Tensor):
        assert constant.numel() == 1
        c_val = float(constant.item())  # extract as Python float — dtype-agnostic
    else:
        c_val = float(constant)

    M, K = input_tensor.shape
    N = linear_bias.shape[0]
    assert linear_weight.shape == (N, K)

    # output fp32 — acc is computed in fp32 inside kernel
    output = torch.empty((M, N), device="xpu", dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 256, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_min_sub_kernel[grid](
        input_tensor, linear_weight, linear_bias, output,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        linear_weight.stride(0), linear_weight.stride(1),
        linear_bias.stride(0),
        output.stride(0), output.stride(1),
        c_val,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    torch.xpu.synchronize()
    return output