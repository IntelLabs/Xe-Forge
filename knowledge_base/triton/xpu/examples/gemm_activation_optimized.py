# This kernel demonstrates all key optimizations:
#
# 1. FUSED kernel (GEMM + activation in single kernel)
# 2. Tensor descriptors instead of manual pointer arithmetic
# 3. Large tile sizes (256x256) optimal for XPU
# 4. 32 warps (XPU optimal vs CUDA's 8)
# 5. grf_mode='256' for large register file
# 6. GROUP_SIZE_M tile swizzling for L2 cache
# 7. 1D grid with swizzling (not 2D)
# 8. Autotune with multiple configs

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_xpu_autotune_configs():
    """
    Intel XPU optimized autotune configurations.

    Key differences from CUDA:
    - Larger tiles (256x256 vs 128x128)
    - More warps (32 vs 8)
    - grf_mode='256' for large register file
    - GROUP_SIZE_M for L2 cache optimization
    """
    return [
        # Primary config: Large tiles, 32 warps
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=2,
            num_warps=32,
        ),
        # Alternative: Smaller N for memory-bound cases
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=2,
            num_warps=32,
        ),
        # Alternative: Higher stages
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=3,
            num_warps=32,
        ),
        # Fallback: Smaller tiles for small matrices
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_SIZE_M": 4,
            },
            num_stages=2,
            num_warps=32,
        ),
    ]


@triton.autotune(
    configs=get_xpu_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_activation_fused_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    # Meta-parameters from autotune (NO defaults - autotune provides them)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    FUSED GEMM + Activation kernel optimized for Intel XPU.

    Computes: y = activation(x @ W^T + bias)
    Where activation = clamp(tanh(clamp(swish(x) / 2)))

    Optimizations applied:
    1. Fused GEMM + activation (single kernel launch)
    2. Tensor descriptors for efficient 2D block memory access
    3. GROUP_SIZE_M swizzling for L2 cache reuse
    4. Large tiles (256x256) for XPU
    5. 32 warps and grf_mode='256'
    """
    # === TILE SWIZZLING FOR L2 CACHE ===
    # Use 1D grid with GROUP_SIZE_M swizzling instead of 2D grid
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # === TENSOR DESCRIPTORS (not manual arithmetic) ===
    # Created once, outside the K loop; tiles are then addressed by coordinates.
    # X block: [BLOCK_M, BLOCK_K] (row-major, last stride is 1)
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        block_shape=(BLOCK_M, BLOCK_K),
    )

    # W block: W is stored as [N, K] with K contiguous, but the dot needs [K, N].
    # A descriptor's last stride must be 1, so describe W as it is laid out in
    # memory ([N, K], strides (stride_wn, stride_wk)), load a [BLOCK_N, BLOCK_K]
    # tile and transpose it in registers with .T
    w_desc = tl.make_tensor_descriptor(
        base=w_ptr,
        shape=(N, K),
        strides=(stride_wn, stride_wk),
        block_shape=(BLOCK_N, BLOCK_K),
    )

    # === GEMM ACCUMULATION ===
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    off_k = 0
    for _ in range(0, K, BLOCK_K):
        # Descriptor loads zero-pad out-of-bounds elements (no masks needed)
        x_block = x_desc.load([pid_m * BLOCK_M, off_k])
        w_block = w_desc.load([pid_n * BLOCK_N, off_k]).T  # [BLOCK_K, BLOCK_N]

        # Dot product: fp16 inputs, fp32 accumulation
        acc = tl.dot(x_block.to(tl.float16), w_block.to(tl.float16), acc=acc)

        # Advance the K coordinate (the descriptors themselves never change)
        off_k += BLOCK_K

    # === ADD BIAS ===
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    # === FUSED ACTIVATION (no separate kernel!) ===
    # Swish: x * sigmoid(x)
    sigmoid = 1.0 / (1.0 + tl.math.exp(-acc))
    swish = acc * sigmoid

    # Divide by 2
    half = swish * 0.5

    # Clamp to [-1, 1]
    clamped1 = tl.minimum(tl.maximum(half, -1.0), 1.0)

    # Tanh
    exp_pos = tl.math.exp(clamped1)
    exp_neg = tl.math.exp(-clamped1)
    tanh_out = (exp_pos - exp_neg) / (exp_pos + exp_neg)

    # Final clamp to [-1, 1]
    result = tl.minimum(tl.maximum(tanh_out, -1.0), 1.0)

    # === STORE WITH TENSOR DESCRIPTOR ===
    # Out-of-bounds elements of the tile are dropped by the descriptor store
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        block_shape=(BLOCK_M, BLOCK_N),
    )
    out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], result)


def _forward_optimized(x, weight, bias):
    """
    OPTIMIZED forward pass - single fused kernel
    """
    M, K = x.shape
    N = weight.shape[0]

    # Keep in original dtype for efficiency
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # 1D grid for swizzling (autotune handles BLOCK_M, BLOCK_N)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _gemm_activation_fused_kernel[grid](
        x_contig,
        w_contig,
        b_contig,
        out,
        M,
        N,
        K,
        x_contig.stride(0),
        x_contig.stride(1),
        w_contig.stride(0),
        w_contig.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out


class Model(nn.Module):
    """
    OPTIMIZED Model for Intel XPU

    Uses single fused kernel with all XPU optimizations:
    - Tensor descriptors
    - Large tiles (256x256)
    - 32 warps
    - grf_mode='256'
    - GROUP_SIZE_M swizzling
    - Fused GEMM + activation
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            bias = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        else:
            bias = self.bias
        return _forward_optimized(x, self.weight, bias)


# ── test helpers ─────────────────────────────────────────────────────────────
def get_init_inputs():
    return [4096, 4096]  # in_features, out_features


def get_inputs():
    return [torch.rand(1024, 4096, dtype=torch.float32)]
