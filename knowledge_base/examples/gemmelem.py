import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_autotune_configs():
    """Intel XPU optimized configs from gemm_benchmark."""
    configs = (
        [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 4,
                    "grf_mode": "256",
                },
                num_stages=s,
                num_warps=32,
            )
            for s in [1, 2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 4,
                    "grf_mode": m,
                },
                num_stages=s,
                num_warps=w,
            )
            for s in [2, 3, 4]
            for (m, w) in [("256", 32), ("128", 64)]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 4,
                    "grf_mode": "256",
                },
                num_stages=s,
                num_warps=32,
            )
            for s in [2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 4,
                    "grf_mode": "256",
                },
                num_stages=s,
                num_warps=32,
            )
            for s in [2, 3]
        ]
        + [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 4,
                    "grf_mode": "256",
                },
                num_stages=2,
                num_warps=32,
            )
        ]
    )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _fused_gemm_activation_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    # Dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused GEMM + activations using Intel XPU optimized patterns.

    Y = X @ W + bias
    Y = swish(Y) / 2
    Y = clamp(tanh(clamp(Y)))
    """
    pid = tl.program_id(axis=0)

    # GROUP_SIZE_M swizzling for L2 cache optimization
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers with boundary checking (Intel optimized)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main GEMM loop with block pointer advancement
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Add bias
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # ========== Fused Activations (all in registers) ==========

    # Swish: x * sigmoid(x)
    sigmoid = 1.0 / (1.0 + tl.exp(-accumulator))
    accumulator = accumulator * sigmoid

    # Divide by 2
    accumulator = accumulator * 0.5

    # Clamp [-1, 1]
    accumulator = tl.maximum(accumulator, -1.0)
    accumulator = tl.minimum(accumulator, 1.0)

    # Tanh: (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * accumulator)
    accumulator = (exp_2x - 1.0) / (exp_2x + 1.0)

    # Final clamp [-1, 1]
    accumulator = tl.maximum(accumulator, -1.0)
    accumulator = tl.minimum(accumulator, 1.0)

    # Store with block pointer
    c = accumulator.to(tl.float16)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


class Model(nn.Module):
    """
    Fused GEMM + activations using Intel XPU optimized patterns.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight stored as [K, N] for B matrix (K x N layout)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_buffer("bias", None)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init (weight.T is the logical [N, K] weight matrix)
        nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()

        M, K = x.shape
        N = self.out_features

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        if self.bias is None:
            bias = torch.zeros(N, device=x.device, dtype=x.dtype)
        else:
            bias = self.bias

        # Grid: total tiles
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

        _fused_gemm_activation_kernel[grid](
            x,
            self.weight,
            bias,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            self.weight.stride(0),
            self.weight.stride(1),
            out.stride(0),
            out.stride(1),
        )

        return out
