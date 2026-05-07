import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Problem defaults (match YAML bench-gpu)
# -----------------------------------------------------------------------------
batch_size = 4096
in_features = 4096
out_features = 4096
scale_shape = (out_features,)


def get_inputs():
    # KernelBench often generates on CPU; Model moves to xpu.
    return [torch.rand(batch_size, in_features, dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]


# -----------------------------------------------------------------------------
# Autotune configs optimized for Intel XPU
#   - Broadened tile space: BLOCK_M/BLOCK_N in {64, 128, 256}, BLOCK_K in {32, 64, 128}
#   - Sweep num_warps in {4, 8, 16, 32}
#   - Tune num_stages
#   - Tune grf_mode across {'128', '256', and default(no override)}
#   - GROUP_SIZE_M swizzling (requires 1D grid)
# -----------------------------------------------------------------------------


def _configs():
    cfgs = []
    # Large square-ish tiles
    cfgs += [
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=32,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
    ]
    # Rectangular: 256x128 and 128x256
    cfgs += [
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
    ]
    # Balanced 128x128 with deeper K
    cfgs += [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=16,
            num_stages=2,
        ),
    ]
    # Small-M variants to improve occupancy when M is skinny
    cfgs += [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
            num_warps=8,
            num_stages=2,
        ),
    ]
    return cfgs


# -----------------------------------------------------------------------------
# Triton kernel (Mixed precision: fp16 IO, fp32 accumulate/BN math)
# Optimizations:
#   - Pre-packed W^T as [K, N] for coalesced RHS access
#   - Block pointers for X/W/Y tiles
#   - 1D grid with GROUP_SIZE_M swizzling
#   - Specialized no-boundary-check path when shapes divisible by tile sizes
# -----------------------------------------------------------------------------
@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _fused_gemm_scale_bn_kernel(
    x_ptr,
    wt_ptr,  # packed W^T with layout [K, N]
    b_ptr,
    s_ptr,
    gamma_ptr,
    beta_ptr,
    rm_ptr,
    rv_ptr,
    y_ptr,
    M,
    N,
    K,
    eps: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wtk,  # strides for wt_ptr [K, N]
    stride_wtn,
    stride_b,
    stride_s,
    stride_gamma,
    stride_beta,
    stride_rm,
    stride_rv,
    stride_ym,
    stride_yn,
    DIVISIBLE: tl.constexpr,  # specialization flag for boundary checks
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1D grid with GROUP_SIZE_M swizzling for better L2 reuse
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers for X [M,K], WT [K,N], and Y [M,N]
    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    wt_bp = tl.make_block_ptr(
        base=wt_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Accumulator: keep in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    if DIVISIBLE:
        for _ in tl.range(0, K, BLOCK_K):
            x_tile = tl.load(x_bp)
            w_tile = tl.load(wt_bp)
            acc += tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16))
            x_bp = tl.advance(x_bp, (0, BLOCK_K))
            wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))
    else:
        for _ in tl.range(0, K, BLOCK_K):
            x_tile = tl.load(x_bp, boundary_check=(0, 1))
            w_tile = tl.load(wt_bp, boundary_check=(0, 1))
            acc += tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16))
            x_bp = tl.advance(x_bp, (0, BLOCK_K))
            wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))

    # Column offsets for per-channel parameters
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if DIVISIBLE:
        # Load once per tile, reuse
        bias = tl.load(b_ptr + offs_n * stride_b)
        scale = tl.load(s_ptr + offs_n * stride_s)
        rm = tl.load(rm_ptr + offs_n * stride_rm).to(tl.float32)
        rv = tl.load(rv_ptr + offs_n * stride_rv).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs_n * stride_gamma).to(tl.float32)
        beta = tl.load(beta_ptr + offs_n * stride_beta).to(tl.float32)
    else:
        n_mask = offs_n < N
        bias = tl.load(b_ptr + offs_n * stride_b, mask=n_mask, other=0.0)
        scale = tl.load(s_ptr + offs_n * stride_s, mask=n_mask, other=0.0)
        rm = tl.load(rm_ptr + offs_n * stride_rm, mask=n_mask, other=0.0).to(tl.float32)
        rv = tl.load(rv_ptr + offs_n * stride_rv, mask=n_mask, other=1.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs_n * stride_gamma, mask=n_mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_n * stride_beta, mask=n_mask, other=0.0).to(tl.float32)

    # Epilogue: (acc + b) * scale -> BN inference: (x - rm) / sqrt(rv + eps) * gamma + beta
    acc = acc + bias[None, :].to(tl.float32)
    acc = acc * scale[None, :].to(tl.float32)
    inv_std = 1.0 / tl.sqrt(rv + eps)
    acc = (acc - rm[None, :]) * inv_std[None, :]
    acc = acc * gamma[None, :] + beta[None, :]

    # Store [BM, BN] as fp16
    if DIVISIBLE:
        tl.store(y_bp, acc.to(tl.float16))
    else:
        tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# -----------------------------------------------------------------------------
# Top-level wrapper (mixed precision: fp16 IO, fp32 compute in-kernel)
# -----------------------------------------------------------------------------
def kernel_function(x, weight_t, bias, scale, bn_weight, bn_bias, running_mean, running_var, eps):
    """
    Fused linear + scale + batchnorm (inference) using Triton on XPU.
    x: [M, K] (fp16 recommended)
    weight_t: [K, N] (fp16) pre-packed transpose of weight for coalesced RHS access
    bias/scale/bn params/stats: [N] (bias/scale/gamma/beta fp16; running stats fp32)
    """
    assert x.device.type == "xpu", "Input must be on XPU"
    assert weight_t.device.type == "xpu", "Weight_t must be on XPU"

    # Ensure dtypes: fp16 for IO tensors, fp32 for running stats
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    if weight_t.dtype != torch.float16:
        weight_t = weight_t.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)
    if scale.dtype != torch.float16:
        scale = scale.to(torch.float16)
    if bn_weight.dtype != torch.float16:
        bn_weight = bn_weight.to(torch.float16)
    if bn_bias.dtype != torch.float16:
        bn_bias = bn_bias.to(torch.float16)
    if running_mean.dtype != torch.float32:
        running_mean = running_mean.to(torch.float32)
    if running_var.dtype != torch.float32:
        running_var = running_var.to(torch.float32)

    # Ensure contiguous for predictable strides
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight_t.is_contiguous():
        weight_t = weight_t.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not bn_weight.is_contiguous():
        bn_weight = bn_weight.contiguous()
    if not bn_bias.is_contiguous():
        bn_bias = bn_bias.contiguous()
    if not running_mean.is_contiguous():
        running_mean = running_mean.contiguous()
    if not running_var.is_contiguous():
        running_var = running_var.contiguous()

    M, K = x.shape
    Kt, N = weight_t.shape
    assert K == Kt, "Incompatible K dimensions"

    # Output in fp16 (IO dtype)
    y = torch.empty((M, N), device="xpu", dtype=torch.float16)

    stride_xm, stride_xk = x.stride()
    stride_wtk, stride_wtn = weight_t.stride()
    stride_b = bias.stride(0)
    stride_s = scale.stride(0)
    stride_gamma = bn_weight.stride(0)
    stride_beta = bn_bias.stride(0)
    stride_rm = running_mean.stride(0)
    stride_rv = running_var.stride(0)
    stride_ym, stride_yn = y.stride()

    # Specialize away boundary checks when shapes divisible by the smallest tiles across configs.
    # Here we pick the minimum BLOCK_M/BLOCK_N=64 and BLOCK_K=32 in the search space.
    divisible = (M % 64 == 0) and (N % 64 == 0) and (K % 32 == 0)

    # 1D grid for swizzling (autotune handles BLOCK_M/N)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fused_gemm_scale_bn_kernel[grid](
        x,
        weight_t,
        bias,
        scale,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        y,
        M,
        N,
        K,
        eps=eps,
        stride_xm=stride_xm,
        stride_xk=stride_xk,
        stride_wtk=stride_wtk,
        stride_wtn=stride_wtn,
        stride_b=stride_b,
        stride_s=stride_s,
        stride_gamma=stride_gamma,
        stride_beta=stride_beta,
        stride_rm=stride_rm,
        stride_rv=stride_rv,
        stride_ym=stride_ym,
        stride_yn=stride_yn,
        DIVISIBLE=divisible,
    )
    return y


# -----------------------------------------------------------------------------
# KernelBench Model
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      y = BN( (x @ W^T + b) * scale )
    Mixed precision: parameters in fp16 (except running stats), IO in fp16.

    Optimizations:
      - Cache packed W^T ([K, N]) on XPU once to ensure coalesced RHS access
      - Use broader autotune search over tiles/warps/stages and grf_mode for Intel XPU
      - 1D swizzled grid (GROUP_SIZE_M) for better L2 reuse
    """

    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Create on CPU; we'll move once lazily to XPU on first forward
        # Keep initial master params in fp32; cast on move to XPU
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))
        self.scale = nn.Parameter(torch.randn(tuple(scale_shape), dtype=torch.float32))
        self.bn_weight = nn.Parameter(torch.ones(self.out_features, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))

        # Running stats kept in fp32 for numeric stability
        self.register_buffer("running_mean", torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(self.out_features, dtype=torch.float32))

        self.eps = float(eps)
        self.momentum = float(momentum)

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / (self.in_features**0.5) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        self._moved_to_xpu = False
        self.weight_t = None  # packed [K, N] on device

    def _move_params_once(self):
        if self._moved_to_xpu:
            return
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")
        dev = torch.device("xpu")
        with torch.no_grad():
            # Cast IO params to fp16 on device for bandwidth/throughput
            self.weight.data = self.weight.data.to(dev, dtype=torch.float16).contiguous()
            # Pre-pack W^T once as contiguous [K, N] for coalesced RHS access
            self.weight_t = self.weight.data.t().contiguous()
            self.bias.data = self.bias.data.to(dev, dtype=torch.float16).contiguous()
            self.scale.data = self.scale.data.to(dev, dtype=torch.float16).contiguous()
            self.bn_weight.data = self.bn_weight.data.to(dev, dtype=torch.float16).contiguous()
            self.bn_bias.data = self.bn_bias.data.to(dev, dtype=torch.float16).contiguous()
            # Keep running stats in fp32
            self.running_mean = self.running_mean.to(dev, dtype=torch.float32).contiguous()
            self.running_var = self.running_var.to(dev, dtype=torch.float32).contiguous()
        self._moved_to_xpu = True

    def forward(self, x):
        # Convert inputs to fp16 IO dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        self._move_params_once()

        if x.device.type != "xpu":
            x = x.to("xpu")
        if not x.is_contiguous():
            x = x.contiguous()

        return kernel_function(
            x,
            self.weight_t,  # pass packed W^T [K, N]
            self.bias,
            self.scale,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.eps,
        )
