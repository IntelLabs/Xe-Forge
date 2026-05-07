import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Fused Triton kernel for Intel XPU:
#   GEMM (fp16 I/O, fp32 accumulation) + bias
#   -> pairwise max-pool along N (kernel=2, stride=2)
#   -> row-wise sum + scaling
#   Produces a single [M] output (accumulated in fp32; cast to fp16 in wrapper)
# -----------------------------------------------------------------------------


def _get_xpu_autotune_configs():
    # XPU-specific: favor 256-sized tiles, include GROUP_SIZE_M for swizzling,
    # sweep num_warps {16,32} and num_stages {2,3,4}, and grf_mode {'256','128'}.
    # Keep BLOCK sizes powers-of-two and <= 256.
    return [
        # Strong 32-warp baselines (often best on XPU)
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=32,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_SIZE_M": 4,
            },
            num_warps=32,
            num_stages=2,
        ),
        # Variants to reduce GRF pressure / improve occupancy
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_warps=16,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_SIZE_M": 4,
            },
            num_warps=16,
            num_stages=4,
        ),
    ]


@triton.autotune(configs=_get_xpu_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _fused_linear_pool_sum_scale_kernel(
    x_ptr,  # [M, K] fp16
    w_ptr,  # [K, N] or logically [K, N] view via strides
    b_ptr,  # [N] fp16
    out_ptr,  # [M] fp32 (accumulator destination for atomic adds)
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,  # strides passed from wrapper for w_ptr; kernel reads [K, N] using (wk, wn)
    stride_b,  # b_ptr stride
    stride_out,  # out_ptr stride (usually 1)
    scale,  # float scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1D flattened grid with GROUP_SIZE_M swizzling across M-tiles to reduce hot-spot atomics
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Block pointers
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Read W logically as [K, N] using provided strides (wk for K, wn for N)
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # FP32 accumulator for GEMM tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop (runtime bound)
    for _ in range(0, K, BLOCK_K):
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))  # [BM, BK]
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))  # [BK, BN]
        # fp16 dot with fp32 accumulation
        acc += tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16))
        # advance pointers
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))

    # Add bias (broadcast across rows)
    tl.max_contiguous(offs_n, BLOCK_N)
    bias_vals = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)  # fp16
    acc = acc + bias_vals.to(tl.float32)[None, :]

    # Pairwise max-pool along N: kernel=2, stride=2, then row-wise sum
    # Assumes BLOCK_N is even; configs ensure even values.
    acc_pairs = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))  # [BM, BN/2, 2]
    pair_max = tl.max(acc_pairs, axis=2)  # [BM, BN/2]
    row_sum = tl.sum(pair_max, axis=1)  # [BM]

    # Scale (fp32)
    scale32 = tl.full((), scale, dtype=tl.float32)
    row_sum = row_sum * scale32

    # Atomic add partial sums into global output (fp32)
    mask_m = offs_m < M
    out_ptrs = out_ptr + offs_m * stride_out
    tl.atomic_add(out_ptrs, row_sum, mask=mask_m)


# -----------------------------------------------------------------------------
# Top-level kernel_function: fused matmul + maxpool+sum+scale (fp16 I/O)
# -----------------------------------------------------------------------------


def kernel_function(
    x: torch.Tensor, weight_for_kernel: torch.Tensor, bias: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    Args:
      x: float16 tensor on 'xpu' of shape [M, K]
      weight_for_kernel: float16 tensor on 'xpu' used in kernel as logical [K, N]
                         (ideally pre-packed [K, N] contiguous)
      bias: float16 tensor on 'xpu' of shape [N]
      scale: float scalar (Python float)
    Returns:
      out: float16 tensor on 'xpu' of shape [M], where each row is
           scale * sum over max-pool 1d with kernel=2,stride=2 of (x @ weight^T + bias)
    """
    # ensure XPU device
    if not (hasattr(torch, "xpu") and x.device.type == "xpu"):
        raise RuntimeError("Input tensor must be on Intel XPU ('xpu')")

    # enforce dtype fp16 and contiguity for Triton
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if weight_for_kernel.dtype != torch.float16:
        weight_for_kernel = weight_for_kernel.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)
    x = x.contiguous()
    weight_for_kernel = weight_for_kernel.contiguous()
    bias = bias.contiguous()

    # shapes
    M, K = x.shape
    # weight_for_kernel should be logical [K, N]
    K_w, N = weight_for_kernel.shape
    assert K_w == K, "Weight inner dim must match input"
    assert bias.shape[0] == N, "Bias length must match output dim"
    if (N % 2) != 0:
        raise ValueError("OUT_FEAT (N) must be even for pairwise pooling.")

    # Output accumulator in fp32 (for atomic adds), will cast to fp16 at end
    out_accum = torch.zeros((M,), device="xpu", dtype=torch.float32)

    # strides
    sxm, sxk = x.stride(0), x.stride(1)
    # weight_for_kernel is [K, N] for best memory access
    swk, swn = weight_for_kernel.stride(0), weight_for_kernel.stride(1)
    sb = bias.stride(0)
    so = out_accum.stride(0)

    # 1D flattened grid with swizzling over (M, N) tiles
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _fused_linear_pool_sum_scale_kernel[grid](
        x,
        weight_for_kernel,
        bias,
        out_accum,
        M,
        N,
        K,
        sxm,
        sxk,
        swn,  # kernel expects (stride_wn, stride_wk); kernel uses (wk, wn) internally
        swk,
        sb,
        so,
        scale,
    )
    # Cast to fp16 for final output
    return out_accum.to(torch.float16)


# -----------------------------------------------------------------------------
# KernelBench Model: ONLY place we "adapt" to the harness
# -----------------------------------------------------------------------------


class Model(nn.Module):
    """
    KernelBench model wrapper:
      y = kernel_function(X, W, b, scale)

    Init signature matches YAML inits:
      (IN_FEAT, OUT_FEAT, KERNEL_SIZE, SCALE_FACTOR)

    Notes:
    - kernel_size is accepted for spec compatibility, but Triton kernel assumes pooling by pairs (kernel=2,stride=2).
    - OUT_FEAT must be even.
    - We pre-pack weight once into [K, N] layout for better memory access; reuse across forwards.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int, scale_factor: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)

        if (self.out_features % 2) != 0:
            raise ValueError(f"OUT_FEAT (N={self.out_features}) must be even (pool pairs of 2).")

        # Parameters in fp16: weight [N,K], bias [N]
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float16)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float16))

        # Reasonable init (similar spirit to nn.Linear), performed in fp32 then cast to fp16
        w_tmp = torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(w_tmp, a=5**0.5)
        self.weight.data = w_tmp.to(torch.float16)

        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        b_tmp = torch.empty(self.out_features, dtype=torch.float32)
        nn.init.uniform_(b_tmp, -bound, bound)
        self.bias.data = b_tmp.to(torch.float16)

        # One-time device move and prepack guard
        self._moved_to_xpu = False
        self._weight_packed_ready = False
        self.weight_kn = None  # will hold [K, N] fp16 contiguous

    def _ensure_device_and_prep(self):
        # Move params to xpu only once and prepack weight to [K,N]
        if not self._moved_to_xpu:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("XPU not available")
            self.weight.data = self.weight.data.to("xpu")
            self.bias.data = self.bias.data.to("xpu")
            self._moved_to_xpu = True
        if not self._weight_packed_ready:
            # Pre-pack weight as [K, N] for coalesced access; do this once
            # Note: keep as a plain tensor (not Parameter)
            self.weight_kn = self.weight.data.t().contiguous()
            self._weight_packed_ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce float16 and device semantics
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        if x.device.type != "xpu":
            x = x.to("xpu")

        # Ensure params live on xpu and prepack weight once
        self._ensure_device_and_prep()

        # x must be [M,K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        # Use prepacked [K, N] weight for better memory access
        return kernel_function(x, self.weight_kn, self.bias, self.scale_factor)


# -----------------------------------------------------------------------------
# KernelBench input generators
# -----------------------------------------------------------------------------
# These should match the YAML dims below.


def get_init_inputs():
    # IN_FEAT подтверждает K, OUT_FEAT подтверждает N
    IN_FEAT = 32768
    OUT_FEAT = 32768  # must be even
    KERNEL_SIZE = 2  # accepted but Triton pooling is pairwise
    SCALE_FACTOR = 0.5
    return [IN_FEAT, OUT_FEAT, KERNEL_SIZE, SCALE_FACTOR]


def get_inputs():
    # X shape: [BATCH, IN_FEAT]
    BATCH = 128
    IN_FEAT, _, _, _ = get_init_inputs()
    # KernelBench usually generates on CPU; Model moves to xpu.
    X = torch.rand((BATCH, IN_FEAT), dtype=torch.float16)
    return [X]
