import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton kernel: Linear (x @ W^T) + bias + GELU
#    x [M,K], w [K,N], b [N] in float16
#    tmp [M,N] float16, accumulate in float32
# -----------------------------------------------------------------------------
@triton.jit
def _linear_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    tmp_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_b,
    stride_tm,
    stride_tn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers for X: [M, K]
    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Block pointers for W: [K, N]
    w_bp = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for _k in range(0, K, BLOCK_K):
        x_tile = tl.load(x_bp, boundary_check=(0, 1)).to(tl.float16)
        w_tile = tl.load(w_bp, boundary_check=(0, 1)).to(tl.float16)
        acc = tl.dot(x_tile, w_tile, acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # bias
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    b = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    # GELU (tanh formulation) using tl.math.tanh (avoids 2 exp)
    # y = 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    S2_PI = 0.7978845608028654  # sqrt(2/pi)
    x_f = acc  # fp32
    z = S2_PI * (x_f + 0.044715 * x_f * x_f * x_f)

    # Prefer tl.tanh if present; otherwise approximate tanh via exp2
    # (works on older Triton where tl.math.tanh is missing)

    # tanh(z) = (e^{2z}-1)/(e^{2z}+1)
    # Use exp2: e^{2z} = 2^{(2z * log2(e))}
    a = 1.702
    LOG2E = 1.4426950408889634
    t = -a * acc
    sig = 1.0 / (1.0 + tl.math.exp2(t * LOG2E))
    y = acc * sig

    # store tmp tile
    tmp_bp = tl.make_block_ptr(
        base=tmp_ptr,
        shape=(M, N),
        strides=(stride_tm, stride_tn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(tmp_bp, y.to(tl.float16), boundary_check=(0, 1))


# -----------------------------------------------------------------------------
# Triton kernel: Row-wise Softmax over N
# tmp [M,N] fp16 -> out [M,N] fp16, compute in fp32
#
# This is the canonical 3-phase pattern:
#   1) row_max (scalar)
#   2) row_sum (scalar)
#   3) write all tiles
# -----------------------------------------------------------------------------
import triton
import triton.language as tl


def _softmax_xpu_configs():
    # N=8192: these are reasonable candidates; try more if needed.
    return [
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=16, num_stages=2),
    ]


@triton.autotune(configs=_softmax_xpu_configs(), key=["N"])
@triton.jit
def _softmax_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row_inp = inp_ptr + pid_m * stride_im
    row_out = out_ptr + pid_m * stride_om

    # Pass 1: max
    row_max = -float("inf")
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < N
        x = tl.load(row_inp + offs * stride_in, mask=m, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))  # scalar

    # Pass 2: sum(exp(x-max))
    LOG2E = 1.4426950408889634
    row_sum = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < N
        x = tl.load(row_inp + offs * stride_in, mask=m, other=-float("inf")).to(tl.float32)
        e = tl.math.exp2((x - row_max) * LOG2E)
        row_sum += tl.sum(e, axis=0)  # scalar

    inv_sum = 1.0 / row_sum

    # Pass 3: store
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        m = offs < N
        x = tl.load(row_inp + offs * stride_in, mask=m, other=-float("inf")).to(tl.float32)
        e = tl.math.exp2((x - row_max) * LOG2E)
        y = (e * inv_sum).to(tl.float16)
        tl.store(row_out + offs * stride_on, y, mask=m)


# -----------------------------------------------------------------------------
# Top-level wrapper
# -----------------------------------------------------------------------------
def kernel_function(input_tensor: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor):
    """
    Linear -> GELU -> Softmax using Triton on Intel XPU.

    input_tensor: [M,K] fp16 on xpu
    weight_t:     [K,N] fp16 on xpu (already transposed / packed)
    bias:         [N] fp16 on xpu
    returns:      [M,N] fp16 on xpu
    """
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU device is required"
    assert input_tensor.device.type == "xpu", "input_tensor must be on xpu"
    assert weight_t.device.type == "xpu", "weight_t must be on xpu"
    assert bias.device.type == "xpu", "bias must be on xpu"

    if input_tensor.dtype != torch.float16:
        input_tensor = input_tensor.to(torch.float16)
    if weight_t.dtype != torch.float16:
        weight_t = weight_t.to(torch.float16)
    if bias.dtype != torch.float16:
        bias = bias.to(torch.float16)

    input_tensor = input_tensor.contiguous()
    weight_t = weight_t.contiguous()
    bias = bias.contiguous()

    M, K = input_tensor.shape
    K2, N = weight_t.shape
    assert K == K2
    assert bias.numel() == N

    tmp = torch.empty((M, N), device="xpu", dtype=torch.float16)
    out = torch.empty((M, N), device="xpu", dtype=torch.float16)

    sxm, sxk = input_tensor.stride()
    swk, swn = weight_t.stride()
    (sb,) = bias.stride()
    stm, stn = tmp.stride()
    som, son = out.stride()

    # GEMM+GELU
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_gelu_kernel[grid1](
        input_tensor,
        weight_t,
        bias,
        tmp,
        M,
        N,
        K,
        sxm,
        sxk,
        swk,
        swn,
        sb,
        stm,
        stn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )

    # Softmax (row-wise)
    # For N=8192, BLOCK_N=1024 is a decent starting point; tune warps.
    SM_BLOCK = 1024
    grid2 = (M,)
    _softmax_kernel[grid2](
        tmp,
        out,
        M,
        N,
        stm,
        stn,
        som,
        son,
    )

    return out


# -----------------------------------------------------------------------------
# KernelBench Model (cached params on XPU)
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    Linear->GELU->Softmax.
    Stores weight in [K,N] (already transposed) and caches fp16 contiguous on XPU.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight_t = nn.Parameter(
            torch.empty(self.in_features, self.out_features, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))

        # init as if original weight were [out,in]
        # fan_in = in_features
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.weight_t, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

        self._cached = False
        self._w_fp16 = None
        self._b_fp16 = None
        self._w_version = -1
        self._b_version = -1

    def _ensure_cached(self):
        dev = torch.device("xpu")
        # refresh caches if params changed
        if (not self._cached) or (self._w_version != self.weight_t._version):
            self._w_fp16 = self.weight_t.to(dev, dtype=torch.float16).contiguous()
            self._w_version = self.weight_t._version
        if (not self._cached) or (self._b_version != self.bias._version):
            self._b_fp16 = self.bias.to(dev, dtype=torch.float16).contiguous()
            self._b_version = self.bias._version
        self._cached = True

    def forward(self, x):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU unavailable")

        if x.device.type != "xpu":
            x = x.to("xpu")
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        x = x.contiguous()

        self._ensure_cached()
        return kernel_function(x, self._w_fp16, self._b_fp16)


# -----------------------------------------------------------------------------
# KernelBench IO
# -----------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]


def get_init_inputs():
    return [in_features, out_features]
