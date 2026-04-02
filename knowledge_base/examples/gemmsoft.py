# ruff: noqa: E731

import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# GEMM -> Bias -> BatchNorm(inference) -> Scale (fp16 IO, fp32 compute)
# + 2D-parallel Softmax for N=8192 (fp16 IO, fp32 reductions)
# -------------------------------------------------------------------------

EPS = 1e-5


def _get_xpu_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "128"},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=16,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4, "grf_mode": "256"},
            num_warps=16,
            num_stages=3,
        ),
    ]


@triton.autotune(configs=_get_xpu_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_bn_scale_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    gamma_ptr,
    beta_ptr,
    rm_ptr,
    rv_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_b,
    stride_g,
    stride_be,
    stride_rm,
    stride_rv,
    stride_sc,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    grf_mode: tl.constexpr,  # autotune meta; not used directly
    EPS: tl.constexpr,
):
    # 1D grid with M-swizzle
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(row_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, col_start),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # GEMM loop
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))  # fp16 in memory
        w = tl.load(w_bp, boundary_check=(0, 1))  # fp16 in memory
        acc = tl.dot(a, w, acc)  # fp16 dot -> fp32 acc
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    offs_n = col_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    b = tl.load(bias_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    rm = tl.load(rm_ptr + offs_n * stride_rm, mask=mask_n, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + offs_n * stride_rv, mask=mask_n, other=0.0).to(tl.float32)
    g = tl.load(gamma_ptr + offs_n * stride_g, mask=mask_n, other=1.0).to(tl.float32)
    be = tl.load(beta_ptr + offs_n * stride_be, mask=mask_n, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(rv + EPS)
    acc = (acc - rm[None, :]) * invstd[None, :] * g[None, :] + be[None, :]

    s = tl.load(scale_ptr + 0 * stride_sc, mask=True, other=1.0).to(tl.float32)
    acc = acc * s

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_bp, acc.to(tl.float16), boundary_check=(0, 1))


# -------------------------------------------------------------------------
# 2D-parallel Softmax over dim=1
#   - per (row, tile) max
#   - reduce max per row
#   - per (row, tile) sumexp + store numerator temp
#   - reduce sum per row
#   - per (row, tile) normalize
# -------------------------------------------------------------------------


@triton.jit
def _softmax_tile_max_kernel(
    x_ptr,
    tile_max_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_tm,
    stride_tn,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    col_start = tile * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=-1e20).to(tl.float32)
    m = tl.max(x, axis=0)
    tl.store(tile_max_ptr + row * stride_tm + tile * stride_tn, m)


@triton.jit
def _softmax_row_reduce_max_kernel(
    tile_max_ptr,
    row_max_ptr,
    M,
    NBLOCKS,
    stride_tm,
    stride_tn,
    stride_rm,
    RBLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, RBLOCK)
    mask = offs < NBLOCKS
    vals = tl.load(tile_max_ptr + row * stride_tm + offs * stride_tn, mask=mask, other=-1e20).to(
        tl.float32
    )
    m = tl.max(vals, axis=0)
    tl.store(row_max_ptr + row * stride_rm, m)


@triton.jit
def _softmax_tile_sumexp_kernel(
    x_ptr,
    tmp_ptr,
    tile_sum_ptr,
    row_max_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_tmpm,
    stride_tmpn,
    stride_sm,
    stride_sn,
    stride_rm,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    col_start = tile * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    m = tl.load(row_max_ptr + row * stride_rm).to(tl.float32)

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=-1e20).to(tl.float32)
    e = tl.exp(x - m)

    # store numerator temp (fp16 IO)
    tl.store(tmp_ptr + row * stride_tmpm + offs * stride_tmpn, e.to(tl.float16), mask=mask)

    s = tl.sum(e, axis=0)
    tl.store(tile_sum_ptr + row * stride_sm + tile * stride_sn, s)


@triton.jit
def _softmax_row_reduce_sum_kernel(
    tile_sum_ptr,
    row_sum_ptr,
    M,
    NBLOCKS,
    stride_sm,
    stride_sn,
    stride_rs,
    RBLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, RBLOCK)
    mask = offs < NBLOCKS
    vals = tl.load(tile_sum_ptr + row * stride_sm + offs * stride_sn, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.sum(vals, axis=0)
    tl.store(row_sum_ptr + row * stride_rs, s)


@triton.jit
def _softmax_tile_norm_kernel(
    tmp_ptr,
    out_ptr,
    row_sum_ptr,
    M,
    N,
    stride_tmpm,
    stride_tmpn,
    stride_om,
    stride_on,
    stride_rs,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    col_start = tile * BLOCK
    offs = col_start + tl.arange(0, BLOCK)
    mask = offs < N

    s = tl.load(row_sum_ptr + row * stride_rs).to(tl.float32)
    num = tl.load(tmp_ptr + row * stride_tmpm + offs * stride_tmpn, mask=mask, other=0.0).to(
        tl.float32
    )
    y = num / s
    tl.store(out_ptr + row * stride_om + offs * stride_on, y.to(tl.float16), mask=mask)


# -------------------------------------------------------------------------
# Packed wrapper: assumes params already moved/packed on XPU
# -------------------------------------------------------------------------


def kernel_function_packed(
    x16: torch.Tensor,
    w_packed_ktn: torch.Tensor,  # (K, N) fp16 contiguous on XPU
    b16: torch.Tensor,
    g16: torch.Tensor,
    be16: torch.Tensor,
    rm32: torch.Tensor,
    rv32: torch.Tensor,
    s16: torch.Tensor,
) -> torch.Tensor:
    M, K = x16.shape
    K_w, N = w_packed_ktn.shape
    assert K == K_w

    y = torch.empty((M, N), device=x16.device, dtype=torch.float16)

    stride_am, stride_ak = x16.stride(0), x16.stride(1)
    stride_wk, stride_wn = w_packed_ktn.stride(0), w_packed_ktn.stride(1)
    stride_b = b16.stride(0)
    stride_g = g16.stride(0)
    stride_be = be16.stride(0)
    stride_rm = rm32.stride(0)
    stride_rv = rv32.stride(0)
    stride_sc = s16.stride(0)
    stride_om, stride_on = y.stride(0), y.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _gemm_bn_scale_kernel[grid](
        x16,
        w_packed_ktn,
        b16,
        g16,
        be16,
        rm32,
        rv32,
        s16,
        y,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wk,
        stride_wn,
        stride_b,
        stride_g,
        stride_be,
        stride_rm,
        stride_rv,
        stride_sc,
        stride_om,
        stride_on,
        EPS=EPS,
    )

    out = torch.empty_like(y)

    # 2D-parallel softmax (tuned for N=8192)
    BLOCK = 256
    NBLOCKS = (N + BLOCK - 1) // BLOCK  # for 8192 -> 32
    # RBLOCK must be >= NBLOCKS and ideally power-of-2 for efficiency
    RBLOCK = 32 if NBLOCKS <= 32 else 64

    tile_max = torch.empty((M, NBLOCKS), device=x16.device, dtype=torch.float32)
    row_max = torch.empty((M,), device=x16.device, dtype=torch.float32)
    tile_sum = torch.empty((M, NBLOCKS), device=x16.device, dtype=torch.float32)
    row_sum = torch.empty((M,), device=x16.device, dtype=torch.float32)
    tmp = torch.empty_like(y)  # fp16 numerator

    sx_m, sx_n = y.stride(0), y.stride(1)
    stm_m, stm_n = tile_max.stride(0), tile_max.stride(1)
    srm = row_max.stride(0)
    ssm_m, ssm_n = tile_sum.stride(0), tile_sum.stride(1)
    srs = row_sum.stride(0)
    stmp_m, stmp_n = tmp.stride(0), tmp.stride(1)
    so_m, so_n = out.stride(0), out.stride(1)

    _softmax_tile_max_kernel[(M, NBLOCKS)](
        y,
        tile_max,
        M,
        N,
        sx_m,
        sx_n,
        stm_m,
        stm_n,
        BLOCK=BLOCK,
        num_warps=4,
    )

    _softmax_row_reduce_max_kernel[(M,)](
        tile_max,
        row_max,
        M,
        NBLOCKS,
        stm_m,
        stm_n,
        srm,
        RBLOCK=RBLOCK,
        num_warps=1,
    )

    _softmax_tile_sumexp_kernel[(M, NBLOCKS)](
        y,
        tmp,
        tile_sum,
        row_max,
        M,
        N,
        sx_m,
        sx_n,
        stmp_m,
        stmp_n,
        ssm_m,
        ssm_n,
        srm,
        BLOCK=BLOCK,
        num_warps=4,
    )

    _softmax_row_reduce_sum_kernel[(M,)](
        tile_sum,
        row_sum,
        M,
        NBLOCKS,
        ssm_m,
        ssm_n,
        srs,
        RBLOCK=RBLOCK,
        num_warps=1,
    )

    _softmax_tile_norm_kernel[(M, NBLOCKS)](
        tmp,
        out,
        row_sum,
        M,
        N,
        stmp_m,
        stmp_n,
        so_m,
        so_n,
        srs,
        BLOCK=BLOCK,
        num_warps=4,
    )

    return out


# -------------------------------------------------------------------------
# KernelBench inputs / inits
# -------------------------------------------------------------------------

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)


def get_inputs():
    # Match YAML: float16 input
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]


# -------------------------------------------------------------------------
# KernelBench Model with parameter packing/caching to avoid per-iter transpose/copies
# -------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)
        self.scale_shape = (
            tuple(scale_shape) if isinstance(scale_shape, (list, tuple)) else (int(scale_shape),)
        )

        self.gemm = nn.Linear(self.in_features, self.out_features, bias=True)

        # BN params/stats used in inference-style BN
        self.bn_weight = nn.Parameter(torch.ones(self.out_features, dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("bn_running_mean", torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer("bn_running_var", torch.ones(self.out_features, dtype=torch.float32))

        self.scale = nn.Parameter(torch.ones(self.scale_shape, dtype=torch.float32))

        # caches (device-specific)
        self._packed_ready = False
        self._packed_device = None
        self._w_packed = None
        self._b16 = None
        self._g16 = None
        self._be16 = None
        self._rm32 = None
        self._rv32 = None
        self._s16 = None

    @torch.no_grad()
    def _ensure_packed(self, device: torch.device):
        if self._packed_ready and self._packed_device == device:
            return

        w16 = self.gemm.weight.to(device, dtype=torch.float16)
        self._w_packed = w16.t().contiguous()  # (K, N) once

        self._b16 = self.gemm.bias.to(device, dtype=torch.float16).contiguous()
        self._g16 = self.bn_weight.to(device, dtype=torch.float16).contiguous()
        self._be16 = self.bn_bias.to(device, dtype=torch.float16).contiguous()

        self._rm32 = self.bn_running_mean.to(device, dtype=torch.float32).contiguous()
        self._rv32 = self.bn_running_var.to(device, dtype=torch.float32).contiguous()

        self._s16 = self.scale.to(device, dtype=torch.float16).contiguous().view(-1)

        self._packed_ready = True
        self._packed_device = device

    def forward(self, x):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU backend is not available")

        device = torch.device("xpu")

        if x.device != device or x.dtype != torch.float16:
            x = x.to(device, dtype=torch.float16)
        if not x.is_contiguous():
            x = x.contiguous()

        self._ensure_packed(device)

        return kernel_function_packed(
            x,
            self._w_packed,
            self._b16,
            self._g16,
            self._be16,
            self._rm32,
            self._rv32,
            self._s16,
        )
