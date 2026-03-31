# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Problem definitions (match YAML defaults)
# -----------------------------------------------------------------------------
batch_size = 100
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5


# -----------------------------------------------------------------------------
# Triton Kernels (UNCHANGED)
# -----------------------------------------------------------------------------
@triton.jit
def sum_weight_kernel(
    w_ptr,  # pointer to weight [N, K], float32
    wcs_ptr,  # pointer to output [K], float32
    N,
    K,  # dims
    stride_w0,
    stride_w1,  # strides of w
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_k = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K

    # accumulator in fp32, one element per column tile
    sum_k = tl.zeros((BLOCK_N,), tl.float32)

    # tile over rows of w
    for start_n in tl.range(0, N, BLOCK_M):
        offs_n = start_n + tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        ptrs = w_ptr + offs_n[:, None] * stride_w0 + offs_k[None, :] * stride_w1
        tile = tl.load(ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        # sum over rows to accumulate per column
        sum_k += tl.sum(tile, axis=0)

    out_ptrs = wcs_ptr + offs_k
    tl.store(out_ptrs, sum_k, mask=mask_k)


@triton.jit
def dot_row_kernel(
    x_ptr,  # pointer to input [M, K], float32
    wcs_ptr,  # pointer to w_colsum [K], float32
    y_ptr,  # pointer to output [M, 1], float32
    M,
    K,
    stride_x0,
    stride_x1,  # strides of x
    stride_wcs,  # stride of w_colsum
    stride_y0,  # stride of y
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    # scalar accumulator in fp32
    acc = 0.0

    # iterate over the K dimension in blocks
    for start_k in tl.range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load a block from x
        x_vals = tl.load(x_ptr + row * stride_x0 + offs_k * stride_x1, mask=mask_k, other=0.0)

        # load the corresponding block from wcs
        wcs_vals = tl.load(wcs_ptr + offs_k * stride_wcs, mask=mask_k, other=0.0)

        # elementwise multiply and sum into the accumulator
        acc += tl.sum(x_vals * wcs_vals, axis=0)

    # combined scale = (1/2) * 1.5 = 0.75
    acc = acc * 0.75

    # write out the result
    out_ptr = y_ptr + row * stride_y0
    tl.store(out_ptr, acc)


# -----------------------------------------------------------------------------
# Top-level Triton wrapper (UNCHANGED)
# -----------------------------------------------------------------------------
def kernel_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    input_tensor: [M, K], float32
    weight:       [N, K], float32
    Returns:
      y: [M, 1], float32 where
         y[i,0] = 1.5 * sum_j( (input @ weight.T)[i,j] / 2 )
               = 0.75 * sum_k( input[i,k] * sum_n weight[n,k] )
    """
    assert input_tensor.dtype == torch.float32 and weight.dtype == torch.float32, (
        "Both inputs must be float32"
    )
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    device = "xpu"

    # Move to XPU
    x = input_tensor.to(device=device, dtype=torch.float32)
    w = weight.to(device=device, dtype=torch.float32)
    M, K = x.shape
    N, K2 = w.shape
    assert K2 == K, "Incompatible shapes"

    # 1) sum weight columns
    w_colsum = torch.empty((K,), device=device, dtype=torch.float32)
    BLOCK_M = 256
    BLOCK_N = 128
    grid0 = (triton.cdiv(K, BLOCK_N),)
    sum_weight_kernel[grid0](
        w,
        w_colsum,
        N,
        K,
        w.stride(0),
        w.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # 2) dot each row of input with w_colsum
    y = torch.empty((M, 1), device=device, dtype=torch.float32)
    BLOCK_K = 256
    grid1 = (M,)
    dot_row_kernel[grid1](
        x,
        w_colsum,
        y,
        M,
        K,
        x.stride(0),
        x.stride(1),
        w_colsum.stride(0),
        y.stride(0),
        BLOCK_K=BLOCK_K,
    )

    # synchronize and move back to CPU
    torch.xpu.synchronize()
    return y.cpu()


# -----------------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      x -> (x @ W^T) -> /2 -> sum(dim=1, keepdim=True) -> * scaling_factor

    Init signature matches YAML inits:
      (INPUT_SIZE, HIDDEN_SIZE, SCALING_FACTOR)

    Important:
    - Triton kernel hardcodes combined scale = 0.75 (i.e., assumes scaling_factor=1.5 and divide by 2).
      So YAML should keep SCALING_FACTOR = 1.5 for correctness.
    - kernel_function returns CPU tensor; KernelBench usually expects a tensor output (CPU is fine).
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.scaling_factor = float(scaling_factor)

        # weight parameter in shape [N, K] = [hidden_size, input_size]
        self.weight = nn.Parameter(
            torch.randn(self.hidden_size, self.input_size, dtype=torch.float32)
        )

    def forward(self, x):
        # KernelBench may pass CPU tensors; keep float32.
        if x.dtype != torch.float32:
            x = x.float()

        # Ensure params are float32 (kernel asserts float32)
        if self.weight.dtype != torch.float32:
            self.weight.data = self.weight.data.float()

        # kernel_function handles move to xpu internally and returns CPU
        return kernel_function(x, self.weight)


# -----------------------------------------------------------------------------
# KernelBench input generators (match YAML below)
# -----------------------------------------------------------------------------
def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=torch.float32)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
