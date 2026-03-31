# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# -------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for fused:
        out = min(x @ W^T + b, constant) - constant

    Init signature matches YAML inits:
        (IN_FEAT, OUT_FEAT, CONSTANT)

    Important:
    - We store weight as [N, K] and bias as [N], matching kernel_function expectations.
    - forward() calls kernel_function() directly (this is what KernelBench should time).
    """

    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Keep constant as a parameter (like your original), but scalar
        self.constant = nn.Parameter(torch.tensor(float(constant), dtype=torch.float32))

        # Params in the exact shapes kernel_function expects: weight [N,K], bias [N]
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))

        # Reasonable init (similar to nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # KernelBench may provide CPU tensors; ensure float32 + xpu.
        if x.dtype != torch.float32:
            x = x.float()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU is not available")

        if x.device.type != "xpu":
            x = x.to("xpu")

        # Ensure params are on xpu (KernelBench often instantiates model on CPU)
        if self.weight.device.type != "xpu":
            self.weight.data = self.weight.data.to("xpu")
        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu")
        if self.constant.device.type != "xpu":
            self.constant.data = self.constant.data.to("xpu")

        # Expect [M, K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight, self.bias, self.constant)


# -------------------------------------------------------------------
# KernelBench input generators (match YAML below)
# -------------------------------------------------------------------


def get_init_inputs():
    IN_FEAT = 16384
    OUT_FEAT = 16384
    CONSTANT = 2.0
    return [IN_FEAT, OUT_FEAT, CONSTANT]


def get_inputs():
    # KernelBench often generates inputs on CPU; Model moves them to xpu.
    BATCH = 128
    IN_FEAT, _, _ = get_init_inputs()
    X = torch.rand((BATCH, IN_FEAT), dtype=torch.float32)
    return [X]


# -------------------------------------------------------------------
# Triton kernel for fused Linear + Min + Sub (UNCHANGED)
# -------------------------------------------------------------------
@triton.jit
def _linear_min_sub_kernel(
    x_ptr,  # pointer to X, shape [M, K]
    w_ptr,  # pointer to W, shape [N, K]
    b_ptr,  # pointer to b, shape [N]
    o_ptr,  # pointer to O, shape [M, N]
    M,
    N,
    K,  # dims
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_b,
    stride_om,
    stride_ok,
    constant,  # scalar clamp value, float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute tile offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # row and column indices for this tile
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    # masks for edge handling
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_blk = tl.load(x_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # load W^T block: W is [N, K], we need W^T -> [K, N]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_blk = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # accumulate
        acc += tl.dot(x_blk, w_blk)

    # add bias
    b_ptrs = b_ptr + offs_n * stride_b
    b_val = tl.load(b_ptrs, mask=mask_n, other=0.0)
    acc = acc + b_val[None, :]

    # apply fused min then subtract
    acc = tl.minimum(acc, constant)
    acc = acc - constant

    # store the result
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, acc, mask=store_mask)


def kernel_function(input_tensor, linear_weight, linear_bias, constant):
    """
    Wrapper for fused Linear‐Min‐Sub on Intel XPU. Computes
        out = min(input @ W^T + b, constant) - constant
    using Triton kernels only.
    """
    # sanity checks
    assert isinstance(input_tensor, torch.Tensor), "input must be a tensor"
    assert isinstance(linear_weight, torch.Tensor), "weight must be a tensor"
    assert isinstance(linear_bias, torch.Tensor), "bias must be a tensor"
    # must be on XPU
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    assert input_tensor.device.type == "xpu", "input_tensor must be on XPU"
    assert linear_weight.device.type == "xpu", "weight must be on XPU"
    assert linear_bias.device.type == "xpu", "bias must be on XPU"

    # extract the constant as Python float
    if isinstance(constant, torch.Tensor):
        assert constant.numel() == 1
        c_val = float(constant.item())
    else:
        c_val = float(constant)

    # dimensions
    M, K = input_tensor.shape
    N = linear_bias.shape[0]
    assert linear_weight.shape == (N, K), f"Weight must be [N,K], got {tuple(linear_weight.shape)}"

    # allocate output
    output = torch.empty((M, N), device="xpu", dtype=input_tensor.dtype)

    # tuning parameters (fits XPU shared memory limits)
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 64

    # compute grid sizes
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # launch the kernel
    _linear_min_sub_kernel[grid](
        # pointers
        input_tensor,
        linear_weight,
        linear_bias,
        output,
        # dimensions
        M,
        N,
        K,
        # strides for input
        input_tensor.stride(0),
        input_tensor.stride(1),
        # strides for weight
        linear_weight.stride(0),
        linear_weight.stride(1),
        # stride for bias
        linear_bias.stride(0),
        # strides for output
        output.stride(0),
        output.stride(1),
        # constant clamp value
        c_val,
        # block sizes
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    # synchronize to ensure completion
    torch.xpu.synchronize()
    return output
