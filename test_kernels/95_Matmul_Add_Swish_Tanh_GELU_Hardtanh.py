# ruff: noqa: E731

import torch
import torch.nn as nn
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
    - Parameters are stored in shapes expected by kernel_function:
        weight [N, K], bias [N], add_value [N]
    - forward() calls kernel_function() directly (this is what KernelBench should time).
    """

    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Parameters in expected layout
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))
        self.add_value = nn.Parameter(torch.randn(tuple(add_value_shape), dtype=torch.float32))

        # Initialize similarly to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # KernelBench may pass CPU tensors; ensure float32 + xpu.
        if x.dtype != torch.float32:
            x = x.float()

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        if x.device.type != "xpu":
            x = x.to("xpu")

        # Ensure params on xpu
        if self.weight.device.type != "xpu":
            self.weight.data = self.weight.data.to("xpu")
        if self.bias.device.type != "xpu":
            self.bias.data = self.bias.data.to("xpu")
        if self.add_value.device.type != "xpu":
            self.add_value.data = self.add_value.data.to("xpu")

        # Expect [M, K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        return kernel_function(x, self.weight, self.bias, self.add_value)


def get_init_inputs():
    return [in_features, out_features, add_value_shape]


def get_inputs():
    # KernelBench often generates inputs on CPU; Model moves to xpu.
    return [torch.rand(batch_size, in_features, dtype=torch.float32)]


# --------------------------------------------------------------------
# Triton Kernels (UNCHANGED)
# --------------------------------------------------------------------
@triton.jit
def _fused_linear_bias_add_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    add_ptr,
    out_ptr,
    M,
    N,
    K,
    S0_x,
    S1_x,
    S0_w,
    S1_w,
    S0_o,
    S1_o,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused matmul (x @ weight^T) + bias + add_value, tiled 2D over M×N.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Compute dot-product tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * S0_x + offs_k[None, :] * S1_x
        w_ptrs = weight_ptr + offs_n[:, None] * S0_w + offs_k[None, :] * S1_w
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        x_block = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc = tl.dot(x_block, w_block.T, acc)
    # Add bias and add_value
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    add_vals = tl.load(add_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias_vals[None, :] + add_vals[None, :]
    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * S0_o + offs_n[None, :] * S1_o
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def _fused_activation_kernel(inp_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Fused Swish → Tanh → exact GELU → Hardtanh clamp over a 1D tensor.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Load input
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # 1) Swish: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    # 2) Tanh(y)
    exp_p = tl.exp(y)
    exp_n = tl.exp(-y)
    y = (exp_p - exp_n) / (exp_p + exp_n)
    # 3) exact GELU: y * 0.5 * (1 + erf(y/sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_inner = 0.5 * (1.0 + tl.math.erf(y * inv_sqrt2))
    y = y * gelu_inner
    # 4) Hardtanh clamp to [-1, 1]
    y = tl.where(y < -1.0, -1.0, y)
    y = tl.where(y > 1.0, 1.0, y)
    # Store output
    tl.store(out_ptr + offs, y, mask=mask)


# --------------------------------------------------------------------
# Top-Level Kernel Function (UNCHANGED)
# --------------------------------------------------------------------
def kernel_function(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, add_value: torch.Tensor
) -> torch.Tensor:
    """
    Orchestrate Triton kernels on XPU to compute:
      out = Hardtanh(GELU(Tanh(Swish(x @ W^T + bias + add_value))))
    """
    # Validate XPU availability
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU device is not available")
    # Shapes and sanity checks
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Incompatible x and weight shapes: {x.shape}, {weight.shape}"
    assert bias.shape == (N,), f"Expected bias shape ({N},), got {bias.shape}"
    assert add_value.shape == (N,), f"Expected add_value shape ({N},), got {add_value.shape}"
    # Move to XPU
    x_xpu = x.to("xpu")
    w_xpu = weight.to("xpu")
    b_xpu = bias.to("xpu")
    add_xpu = add_value.to("xpu")
    # 1) Fused Linear + Bias + Add
    out1 = torch.empty((M, N), dtype=x.dtype, device="xpu")
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_linear_bias_add_kernel[grid1](
        x_xpu,
        w_xpu,
        b_xpu,
        add_xpu,
        out1,
        M,
        N,
        K,
        x_xpu.stride(0),
        x_xpu.stride(1),
        w_xpu.stride(0),
        w_xpu.stride(1),
        out1.stride(0),
        out1.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    # 2) Fused Activations
    out_flat = out1.view(-1)
    n_elems = out_flat.numel()
    out2_flat = torch.empty(n_elems, dtype=x.dtype, device="xpu")
    BLOCK_ACT = 256
    grid2 = (triton.cdiv(n_elems, BLOCK_ACT),)
    _fused_activation_kernel[grid2](out_flat, out2_flat, n_elems, BLOCK_ACT)
    out2 = out2_flat.view(M, N)
    return out2
