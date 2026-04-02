# ruff: noqa: E731
import torch
import torch.nn as nn
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Fast structure for: Gemm -> Sigmoid -> Gemm -> LogSumExp(row)
#
# Key fixes vs slow version:
#  1) Cache packed transposes W1^T and W2^T once on XPU (no per-forward repack)
#  2) GEMM2 is a normal 2D-tiled GEMM (parallel over M and N) producing logits
#  3) LogSumExp is a separate row-wise reduction kernel (keeps parallelism)
#
# Notes:
#  - fp16 inputs/weights/logits, fp32 accumulation
#  - Sigmoid uses exp2-based approximation (faster/more portable on XPU)
#  - Returns Y on XPU (no device->host copy in hot path)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Autotune configs
# -----------------------------------------------------------------------------
def _cfgs_gemm():
    # Keep modest search space; you can expand if needed.
    return [
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 32, "grf_mode": "256"}, num_warps=16, num_stages=3
        ),
        triton.Config(
            {"BM": 128, "BN": 64, "BK": 32, "grf_mode": "256"}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 32, "grf_mode": "256"}, num_warps=8, num_stages=4
        ),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "grf_mode": "256"}, num_warps=4, num_stages=4),
        # GRF=128 variants (sometimes win with lower reg pressure)
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 32, "grf_mode": "128"}, num_warps=16, num_stages=3
        ),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "grf_mode": "128"}, num_warps=4, num_stages=4),
    ]


def _cfgs_lse():
    # One kernel: reduce across N in tiles; BM controls rows-per-program.
    return [
        triton.Config({"BM": 128, "BN": 256, "grf_mode": "256"}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "grf_mode": "256"}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 256, "grf_mode": "256"}, num_warps=4, num_stages=4),
    ]


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
@triton.jit
def _sigmoid_exp2(x_f32):
    # sigmoid(x) = 1 / (1 + exp(-x))
    # exp(-x) = exp2((-x) / ln(2))
    inv_ln2 = 1.4426950408889634  # 1/ln(2)
    e = tl.math.exp2((-x_f32) * inv_ln2)
    return 1.0 / (1.0 + e)


# -----------------------------------------------------------------------------
# Kernel 1: GEMM1 + Bias1 + Sigmoid
#   A:   [M, K]
#   W1t: [K, H]  (packed transpose, contiguous)
#   B1:  [H]
#   OUT: [M, H]
# -----------------------------------------------------------------------------
@triton.autotune(configs=_cfgs_gemm(), key=["M", "N", "K"])
@triton.jit
def gemm_bias_sigmoid_kernel(
    A_ptr,
    B_ptr,  # Wt [K, N]
    bias_ptr,  # [N] (here N=H)
    C_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_base = pid_m * BM
    n_base = pid_n * BN

    # Block pointers
    a_bp = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_base, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    b_bp = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_base),
        block_shape=(BK, BN),
        order=(1, 0),
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # K loop (dynamic-safe)
    for _ in tl.range(0, K, BK):
        a = tl.load(a_bp, boundary_check=(0, 1))  # fp16
        b = tl.load(b_bp, boundary_check=(0, 1))  # fp16
        acc += tl.dot(a, b)  # fp32 accumulate
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))

    # Add bias
    offs_n = n_base + tl.arange(0, BN)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Sigmoid (fp32)
    acc = _sigmoid_exp2(acc)

    # Store fp16
    c_bp = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_base, n_base),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(tl.float16), boundary_check=(0, 1))


# -----------------------------------------------------------------------------
# Kernel 2: GEMM2 + Bias2  (NO LSE here; parallel 2D)
#   A:   [M, H] (inter)
#   W2t: [H, O] (packed transpose, contiguous)
#   B2:  [O]
#   OUT: [M, O] (logits)
# -----------------------------------------------------------------------------
@triton.autotune(configs=_cfgs_gemm(), key=["M", "N", "K"])
@triton.jit
def gemm_bias_kernel(
    A_ptr,
    B_ptr,  # Wt [K, N]
    bias_ptr,  # [N]
    C_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_base = pid_m * BM
    n_base = pid_n * BN

    a_bp = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_base, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    b_bp = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_base),
        block_shape=(BK, BN),
        order=(1, 0),
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in tl.range(0, K, BK):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))

    offs_n = n_base + tl.arange(0, BN)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_bp = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_base, n_base),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(tl.float16), boundary_check=(0, 1))


# -----------------------------------------------------------------------------
# Kernel 3: Row-wise LogSumExp over logits [M, O] -> Y [M]
#   logits: [M, N] fp16
#   Y:      [M]    fp16
# -----------------------------------------------------------------------------
@triton.autotune(configs=_cfgs_lse(), key=["M", "N"])
@triton.jit
def row_lse_kernel(
    X_ptr,  # [M, N]
    Y_ptr,  # [M]
    M,
    N,
    stride_xm,
    stride_xn,
    stride_y,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_base = pid_m * BM
    offs_m = m_base + tl.arange(0, BM)
    mask_m = offs_m < M

    # Running max and sum(exp(. - max))
    m_row = tl.full((BM,), -float("inf"), tl.float32)
    s_row = tl.zeros((BM,), tl.float32)

    # Sweep N in tiles
    for n_base in tl.range(0, N, BN):
        offs_n = n_base + tl.arange(0, BN)
        mask_n = offs_n < N

        # Load tile [BM, BN]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=-float("inf")).to(
            tl.float32
        )

        # Stable LSE update
        tile_max = tl.max(x, axis=1)
        m_new = tl.maximum(m_row, tile_max)
        s_row = s_row * tl.exp(m_row - m_new) + tl.sum(tl.exp(x - m_new[:, None]), axis=1)
        m_row = m_new

    res = tl.log(s_row) + m_row
    tl.store(Y_ptr + offs_m * stride_y, res.to(tl.float16), mask=mask_m)


# -----------------------------------------------------------------------------
# Top-level wrapper (expects weights as [H, K] and [O, H] like nn.Linear)
# Uses cached packed transposes built in Model._move_params_once().
# -----------------------------------------------------------------------------
def kernel_function_fast(
    x: torch.Tensor,
    W1_t: torch.Tensor,  # [K, H] fp16 contiguous on XPU
    B1: torch.Tensor,  # [H] fp16 on XPU
    W2_t: torch.Tensor,  # [H, O] fp16 contiguous on XPU
    B2: torch.Tensor,  # [O] fp16 on XPU
) -> torch.Tensor:
    assert hasattr(torch, "xpu") and torch.xpu.is_available()
    assert x.device.type == "xpu"
    assert x.dtype == torch.float16 and x.is_contiguous()
    assert W1_t.dtype == torch.float16 and W1_t.is_contiguous()
    assert W2_t.dtype == torch.float16 and W2_t.is_contiguous()

    M, K = x.shape
    K2, H = W1_t.shape
    H2, O = W2_t.shape
    assert K2 == K and H2 == H

    # inter [M, H]
    inter = torch.empty((M, H), device="xpu", dtype=torch.float16)
    # logits [M, O]
    logits = torch.empty((M, O), device="xpu", dtype=torch.float16)
    # output [M]
    y = torch.empty((M,), device="xpu", dtype=torch.float16)

    # GEMM1+sigmoid
    grid1 = (triton.cdiv(M, 128), triton.cdiv(H, 128))
    gemm_bias_sigmoid_kernel[grid1](
        x,
        W1_t,
        B1,
        inter,
        M,
        H,
        K,
        x.stride(0),
        x.stride(1),
        W1_t.stride(0),
        W1_t.stride(1),
        inter.stride(0),
        inter.stride(1),
    )

    # GEMM2
    grid2 = (triton.cdiv(M, 128), triton.cdiv(O, 128))
    gemm_bias_kernel[grid2](
        inter,
        W2_t,
        B2,
        logits,
        M,
        O,
        H,
        inter.stride(0),
        inter.stride(1),
        W2_t.stride(0),
        W2_t.stride(1),
        logits.stride(0),
        logits.stride(1),
    )

    # Row LSE
    grid3 = (triton.cdiv(M, 128),)
    row_lse_kernel[grid3](
        logits,
        y,
        M,
        O,
        logits.stride(0),
        logits.stride(1),
        y.stride(0),
    )

    return y


# -----------------------------------------------------------------------------
# KernelBench input generators
# -----------------------------------------------------------------------------
def get_inputs():
    BATCH = 16384
    INPUT_SIZE = 2048
    return [torch.rand(BATCH, INPUT_SIZE, dtype=torch.float16)]


def get_init_inputs():
    # (input_size, hidden_size, output_size)
    return [2048, 4096, 1024]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for:
      x -> Linear1 -> Sigmoid -> Linear2 -> LogSumExp(dim=1)

    We cache W1^T and W2^T once on XPU (fp16 contiguous).
    Biases are kept fp16 on XPU for cheap loads.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        # Store params in fp32 on CPU initially (KernelBench style)
        self.weight1 = nn.Parameter(
            torch.empty(self.hidden_size, self.input_size, dtype=torch.float32)
        )
        self.bias1 = nn.Parameter(torch.empty(self.hidden_size, dtype=torch.float32))
        self.weight2 = nn.Parameter(
            torch.empty(self.output_size, self.hidden_size, dtype=torch.float32)
        )
        self.bias2 = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.weight1, a=5**0.5)
        bound1 = 1.0 / (self.input_size**0.5) if self.input_size > 0 else 0.0
        nn.init.uniform_(self.bias1, -bound1, bound1)

        nn.init.kaiming_uniform_(self.weight2, a=5**0.5)
        bound2 = 1.0 / (self.hidden_size**0.5) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self._moved_to_xpu = False
        self.W1_t = None
        self.B1 = None
        self.W2_t = None
        self.B2 = None

    def _move_params_once(self):
        if self._moved_to_xpu:
            return
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        dev = torch.device("xpu")
        with torch.no_grad():
            # Move and cast to fp16 on XPU
            W1 = self.weight1.data.to(device=dev, dtype=torch.float16).contiguous()  # [H, K]
            W2 = self.weight2.data.to(device=dev, dtype=torch.float16).contiguous()  # [O, H]
            self.B1 = self.bias1.data.to(device=dev, dtype=torch.float16).contiguous()  # [H]
            self.B2 = self.bias2.data.to(device=dev, dtype=torch.float16).contiguous()  # [O]

            # Pack transposes once
            self.W1_t = W1.t().contiguous()  # [K, H]
            self.W2_t = W2.t().contiguous()  # [H, O]

        self._moved_to_xpu = True

    def forward(self, x):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU device is not available")

        self._move_params_once()

        # Ensure x is fp16 contiguous on XPU
        if x.device.type != "xpu":
            x = x.to("xpu")
        if x.dtype != torch.float16:
            x = x.half()
        if not x.is_contiguous():
            x = x.contiguous()

        return kernel_function_fast(x, self.W1_t, self.B1, self.W2_t, self.B2)
