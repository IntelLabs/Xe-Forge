"""
Conv2d(64->64, 3x3) + multiply + LeakyReLU + GELU on Intel Arc B580 (XPU)

Optimization patterns applied (cumulative):
  1. CHANNELS-LAST (NHWC) layout — model and input converted once at setup.

  2. FUSED 5-OP EPILOGUE — conv + bias + multiply + LeakyReLU + GELU in
     one kernel. The original code used 2 kernels (conv+bias, then
     mul+leakyrelu+gelu) with a full global memory roundtrip (~1 GB)
     between them. Fusing eliminates that entirely — the epilogue runs
     on the accumulator in registers after the conv GEMM completes.

  3. SPATIAL TILING — grid is (n, oh, ow_tile) preserving spatial structure.
     Same insight as 1.py: within a single oh row, BLOCK_OW output pixels
     map to CONSECUTIVE x rows in flat NHWC memory. This enables
     block_ptr for x (input), w (weights), AND y (output) — all three
     operands use hardware 2D block IO. No scatter/gather loads at all.

  4. BOUNDARY-LIMITED block_ptr SHAPE — prevents oh-row wrapping in both
     x reads and y writes when BLOCK_OW doesn't evenly divide OW.

  5. C_IN as tl.constexpr — channel reduction loop fully unrolled.

  6. PRE-PREPARED INPUTS — zero per-call overhead.

  7. AUTOTUNED (BLOCK_OW, BLOCK_N, BLOCK_K, num_warps, num_stages).

  Key difference from 1.py: the heavier epilogue (multiply by learnable
  per-channel scalar + LeakyReLU + GELU with erf) makes fusion even more
  valuable — the intermediate tensor that was written/read between the
  two original kernels is ~1 GB for this problem size.
"""

# ruff: noqa: E731
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Original reference model
# ============================================================

class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar,
    applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = F.gelu(x)
        return x


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]


# ============================================================
# Baseline kernels (NCHW conv + flat eltwise) — unchanged
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64,  'BLOCK_H': 8,  'BLOCK_OC': 32, 'BLOCK_CIN': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 8,  'BLOCK_OC': 32, 'BLOCK_CIN': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_W': 64,  'BLOCK_H': 16, 'BLOCK_OC': 32, 'BLOCK_CIN': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_W': 64,  'BLOCK_H': 8,  'BLOCK_OC': 64, 'BLOCK_CIN': 32}, num_warps=16, num_stages=3),
    ],
    key=["H_out", "W_out", "C_in", "C_out"],
)
@triton.jit
def _conv2d_nchw_3x3_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    H_out, W_out,
    BLOCK_W: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_CIN: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_oc = tl.program_id(2)

    num_h_tiles = tl.cdiv(H_out, BLOCK_H)
    n = pid_nh // num_h_tiles
    tile_h_idx = pid_nh % num_h_tiles

    ow0 = pid_w * BLOCK_W
    oh0 = tile_h_idx * BLOCK_H
    oc0 = pid_oc * BLOCK_OC

    offs_w = ow0 + tl.arange(0, BLOCK_W)
    offs_h = oh0 + tl.arange(0, BLOCK_H)
    offs_oc = oc0 + tl.arange(0, BLOCK_OC)
    offs_ci_block = tl.arange(0, BLOCK_CIN)

    mask_hw = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)
    acc = tl.zeros((BLOCK_OC, BLOCK_H * BLOCK_W), dtype=tl.float32)

    for ci0 in range(0, C_in, BLOCK_CIN):
        offs_ci = ci0 + offs_ci_block
        ci_mask = offs_ci < C_in

        for ky in range(3):
            for kx in range(3):
                w_ptrs = (
                    w_ptr
                    + offs_oc[:, None] * stride_w_co
                    + offs_ci[None, :] * stride_w_ci
                    + ky * stride_w_kh
                    + kx * stride_w_kw
                )
                w_mask = (offs_oc[:, None] < C_out) & ci_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                in_h = offs_h + ky
                in_w = offs_w + kx
                x_ptrs = (
                    x_ptr
                    + n * stride_x_n
                    + offs_ci[:, None, None] * stride_x_c
                    + in_h[None, :, None] * stride_x_h
                    + in_w[None, None, :] * stride_x_w
                )
                x_mask = (
                    ci_mask[:, None, None]
                    & (offs_h[None, :, None] < H_out)
                    & (offs_w[None, None, :] < W_out)
                )
                x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_mat = tl.reshape(x_block, (BLOCK_CIN, BLOCK_H * BLOCK_W))
                acc = tl.dot(w_tile, x_mat, acc)

    bias = tl.load(b_ptr + offs_oc, mask=(offs_oc < C_out), other=0.0)
    acc = acc + bias[:, None]

    y_vals = acc.to(y_ptr.dtype.element_ty)
    y_vals = tl.reshape(y_vals, (BLOCK_OC, BLOCK_H, BLOCK_W))
    y_ptrs = (
        y_ptr
        + n * stride_y_n
        + offs_oc[:, None, None] * stride_y_c
        + offs_h[None, :, None] * stride_y_h
        + offs_w[None, None, :] * stride_y_w
    )
    y_mask = (offs_oc[:, None, None] < C_out) & mask_hw[None, :, :]
    tl.store(y_ptrs, y_vals, mask=y_mask)


@triton.jit
def _mul_leakyrelu_gelu_kernel(
    x_ptr, mult_ptr, out_ptr,
    n_elements, C, stride_c, mult_stride0, negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_idx = (offs // stride_c) % C
    mult_offs = c_idx * mult_stride0

    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m_val = tl.load(mult_ptr + mult_offs, mask=mask, other=0.0).to(tl.float32)

    y = x_val * m_val
    y = tl.where(y >= 0, y, y * negative_slope)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.70710678118654752440))

    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ============================================================
# PATTERN: Spatially-tiled fused kernel (ALL block_ptr)
#
# Grid: (n, oh, ow_tile) — C_out=64=BLOCK_N so only 1 N tile.
#   Same spatial tiling as 1.py: preserving (n, oh) in the grid
#   keeps ow pixels consecutive in NHWC memory.
#
# FUSED EPILOGUE: conv + bias + multiply + LeakyReLU + GELU
#   All 5 ops execute on the accumulator in registers.
#   Saves ~1 GB of intermediate memory traffic vs 2-kernel approach.
#
# Memory access: ALL block_ptr (x, w, y) -> zero scatter loads.
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['H', 'W', 'C_IN', 'C_out', 'OH', 'OW'],
)
@triton.jit
def _fused_conv_spatial_tiled(
    x_ptr, w_ptr, conv_bias_ptr, mult_ptr, y_ptr,
    N_batch, H, W, C_out, OH, OW,
    stride_wkh, stride_wkw, stride_wci, stride_wco,
    negative_slope,
    BLOCK_OW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
):
    """Spatially-tiled fused conv: ALL operands use block_ptr.

    Grid: (n, oh, ceil(OW/BLOCK_OW))
    x viewed as flat (N*H*W, C_IN) contiguous matrix.
    Within a single oh row, BLOCK_OW output pixels read consecutive x rows.
    This enables block_ptr for x — no scatter loads!
    """
    n = tl.program_id(0)
    oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    ow0 = pid_ow * BLOCK_OW
    HW = H * W
    OHOW = OH * OW

    acc = tl.zeros((BLOCK_OW, BLOCK_N), dtype=tl.float32)

    # KH*KW small GEMMs — ALL using block_ptr
    for kh in range(KH):
        for kw in range(KW):
            # x rows: n*H*W + (oh+kh)*W + (ow0+kw) .. + (BLOCK_OW-1)
            # These are CONSECUTIVE rows in (N*H*W, C_IN) — block_ptr!
            x_row_start = n * HW + (oh + kh) * W + (ow0 + kw)

            # Limit shape so boundary_check prevents reading past this h-row
            x_valid_rows = W - (ow0 + kw)
            x_bp = tl.make_block_ptr(
                base=x_ptr,
                shape=(x_row_start + x_valid_rows, C_IN),
                strides=(C_IN, 1),
                offsets=(x_row_start, 0),
                block_shape=(BLOCK_OW, BLOCK_K),
                order=(1, 0),
            )

            w_bp = tl.make_block_ptr(
                base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                shape=(C_IN, C_out),
                strides=(stride_wci, stride_wco),
                offsets=(0, 0),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )

            for c0 in range(0, C_IN, BLOCK_K):
                x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
                x_bp = tl.advance(x_bp, (0, BLOCK_K))
                w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # Fused epilogue
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    mult = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)

    acc += conv_b[None, :]
    acc *= mult[None, :]
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678118654752440))

    # Store: limit shape so boundary_check prevents writing past this oh-row
    y_row_start = n * OHOW + oh * OW + ow0
    y_valid_rows = OW - ow0
    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(y_row_start + y_valid_rows, C_out),
        strides=(C_out, 1),
        offsets=(y_row_start, 0),
        block_shape=(BLOCK_OW, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================
# Previous: flat-M fused kernel (kept for comparison)
#   conv(NHWC) + bias + multiply + LeakyReLU + GELU
#   Persistent, block_ptr output, C_IN constexpr
# ============================================================

@triton.autotune(
    configs=[
        # C_out=64 -> BLOCK_N=64 always.
        # 160 EUs — try 1x and 2x occupancy
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4, 'NUM_SMS': 160}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4, 'NUM_SMS': 160}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=4, num_stages=4),
        # 2x occupancy — smaller WGs, more concurrent to hide mem latency
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 320}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 320}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 320}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 320}, num_warps=4, num_stages=3),
        # 4x occupancy
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 640}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 640}, num_warps=4, num_stages=2),
        # Non-persistent: NUM_SMS >= num_tiles (1 tile per WG, HW schedules everything)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 65536}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 65536}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 65536}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 65536}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C_IN', 'C_out', 'OH', 'OW'],
)
@triton.jit
def _fused_conv_mul_leakyrelu_gelu(
    x_ptr, w_ptr, conv_bias_ptr, mult_ptr, y_ptr,
    N, C_out, OH, OW,
    M_total,
    stride_xn, stride_xh, stride_xw,
    stride_wkh, stride_wkw, stride_wci, stride_wco,
    sy_i, sy_k,
    negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
    grf_mode: tl.constexpr = 'large',
):
    """Fused persistent kernel: conv2d + bias + multiply + LeakyReLU + GELU.

    Single kernel, no intermediate global memory writes.
    Bias/mult preloaded before conv loops (stay in registers during GEMM).
    grf_mode='large' doubles register file for the heavy fused epilogue.
    """
    pid = tl.program_id(0)

    nhw = OH * OW
    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)

    num_tiles = num_pid_m * num_pid_n

    offs_k = tl.arange(0, BLOCK_K)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        # GROUP_M swizzle
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group

        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        n_idx = offs_m // nhw
        rem = offs_m % nhw
        oh_idx = rem // OW
        ow_idx = rem % OW

        mask_m = offs_m < M_total
        mask_n = offs_n < C_out

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_base = (
            x_ptr
            + n_idx * stride_xn
            + oh_idx * stride_xh
            + ow_idx * stride_xw
        )

        # Preload bias + multiplier (tiny, stays in registers)
        conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
        mult = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)

        # KH*KW small GEMMs
        for kh in range(KH):
            for kw in range(KW):
                x_kh_kw = x_base + kh * stride_xh + kw * stride_xw

                w_bp = tl.make_block_ptr(
                    base=w_ptr + kh * stride_wkh + kw * stride_wkw,
                    shape=(C_IN, C_out),
                    strides=(stride_wci, stride_wco),
                    offsets=(0, pid_n * BLOCK_N),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                for c0 in range(0, C_IN, BLOCK_K):
                    k_idx = c0 + offs_k

                    x_tile = tl.load(
                        x_kh_kw[:, None] + k_idx[None, :],
                        mask=mask_m[:, None] & (k_idx[None, :] < C_IN),
                        other=0.0,
                    )

                    w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                    acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
                    w_bp = tl.advance(w_bp, (BLOCK_K, 0))

        # Fused epilogue: + bias -> * mult -> LeakyReLU -> GELU (all in registers)
        acc += conv_b[None, :]
        acc *= mult[None, :]
        acc = tl.where(acc >= 0, acc, acc * negative_slope)
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678118654752440))

        # Store via block_ptr: y is contiguous (M, C_out)
        y_bp = tl.make_block_ptr(
            base=y_ptr,
            shape=(M_total, C_out),
            strides=(sy_i, sy_k),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================
# im2col approach: explicit im2col + GEMM with fused epilogue
# Both GEMM operands use block_ptr (fully contiguous)
# ============================================================

@triton.jit
def _im2col_nhwc_kernel(
    x_ptr, col_ptr,
    N, H, W, C,
    OH, OW,
    m_offset,  # global M offset for chunked processing
    M_chunk,   # number of rows in this chunk
    stride_xn, stride_xh, stride_xw,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """im2col: x[N,H,W,C] -> col[M_chunk, KH*KW*C].

    m_offset shifts the global M index for chunked processing.
    """
    pid = tl.program_id(0)
    nhw = OH * OW

    offs_local = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = offs_local + m_offset  # global M index
    mask_m = offs_local < M_chunk

    n_idx = offs_m // nhw
    rem = offs_m % nhw
    oh_idx = rem // OW
    ow_idx = rem % OW

    K_TOTAL: tl.constexpr = KH * KW * C_IN
    col_row_stride = K_TOTAL  # col is contiguous (M, K_total)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh_idx + kh
            iw = ow_idx + kw

            x_base = (
                x_ptr
                + n_idx * stride_xn
                + ih * stride_xh
                + iw * stride_xw
            )

            k_offset = (kh * KW + kw) * C_IN

            for c0 in range(0, C_IN, BLOCK_K):
                offs_c = c0 + tl.arange(0, BLOCK_K)
                mask_c = offs_c < C_IN

                x_vals = tl.load(
                    x_base[:, None] + offs_c[None, :],
                    mask=mask_m[:, None] & mask_c[None, :],
                    other=0.0,
                )

                col_ptrs = (
                    col_ptr
                    + offs_local[:, None] * col_row_stride  # local index into col buffer
                    + (k_offset + offs_c)[None, :]
                )
                tl.store(col_ptrs, x_vals, mask=mask_m[:, None] & mask_c[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4, 'NUM_SMS': 160}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4, 'NUM_SMS': 160}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'NUM_SMS': 160}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'NUM_SMS': 320}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'K_TOTAL', 'C_out'],
)
@triton.jit
def _gemm_fused_epilogue(
    col_ptr, w_ptr, conv_bias_ptr, mult_ptr, y_ptr,
    M_chunk, C_out,
    y_m_offset,  # offset into y for chunked writes
    K_TOTAL: tl.constexpr,
    sy_i, sy_k,
    negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """GEMM with fused epilogue: col @ w + bias * mult -> LeakyReLU -> GELU.

    BOTH operands use block_ptr (fully contiguous after im2col).
    """
    start_pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M_chunk, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = tile_id % num_pid_in_group

        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < C_out

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Block ptrs for BOTH operands — fully contiguous!
        a_bp = tl.make_block_ptr(
            base=col_ptr,
            shape=(M_chunk, K_TOTAL),
            strides=(K_TOTAL, 1),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_bp = tl.make_block_ptr(
            base=w_ptr,
            shape=(K_TOTAL, C_out),
            strides=(C_out, 1),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        for _ in range(0, K_TOTAL, BLOCK_K):
            a_tile = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
            b_tile = tl.load(b_bp, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
            a_bp = tl.advance(a_bp, (0, BLOCK_K))
            b_bp = tl.advance(b_bp, (BLOCK_K, 0))

        # Fused epilogue
        conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
        mult = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)

        acc += conv_b[None, :]
        acc *= mult[None, :]
        acc = tl.where(acc >= 0, acc, acc * negative_slope)
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.70710678118654752440))

        y_bp = tl.make_block_ptr(
            base=y_ptr,
            shape=(y_m_offset + M_chunk, C_out),
            strides=(sy_i, sy_k),
            offsets=(y_m_offset + pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================
# Helpers
# ============================================================

def _ensure_xpu_fp16(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


def _ensure_xpu_fp16_contig(x: torch.Tensor) -> torch.Tensor:
    return _ensure_xpu_fp16(x).contiguous()


def _ensure_xpu_fp16_channels_last(x: torch.Tensor) -> torch.Tensor:
    return _ensure_xpu_fp16(x).contiguous(memory_format=torch.channels_last)


def _pack_weight_oihw_to_hwio(weight: torch.Tensor) -> torch.Tensor:
    return _ensure_xpu_fp16(weight).permute(2, 3, 1, 0).contiguous()


def _pack_weight_identity_oihw(weight: torch.Tensor) -> torch.Tensor:
    return _ensure_xpu_fp16(weight).contiguous()


# ============================================================
# Baseline path (unchanged)
# ============================================================

def baseline_conv_kernel(x, weight_oihw, bias):
    N, C_in, H, W = x.shape
    C_out = weight_oihw.shape[0]
    H_out, W_out = H - 2, W - 2

    y = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    sxn, sxc, sxh, sxw = x.stride()
    sw_co, sw_ci, sw_kh, sw_kw = weight_oihw.stride()
    sy_n, sy_c, sy_h, sy_w = y.stride()

    def grid(meta):
        return (
            triton.cdiv(W_out, meta["BLOCK_W"]),
            N * triton.cdiv(H_out, meta["BLOCK_H"]),
            triton.cdiv(C_out, meta["BLOCK_OC"]),
        )

    _conv2d_nchw_3x3_kernel[grid](
        x, weight_oihw, bias, y,
        N, C_in, H, W, C_out,
        sxn, sxc, sxh, sxw,
        sw_co, sw_ci, sw_kh, sw_kw,
        sy_n, sy_c, sy_h, sy_w,
        H_out, W_out,
    )
    return y


def baseline_eltwise_kernel(x, multiplier, negative_slope=0.01):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    stride_c = x.stride(1)
    ms0, _, _ = multiplier.stride()

    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _mul_leakyrelu_gelu_kernel[grid](
        x, multiplier, out,
        n_elements, C, stride_c, ms0, float(negative_slope),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def baseline_kernel_function(x, weight_oihw, bias, multiplier, negative_slope=0.01):
    x = _ensure_xpu_fp16_contig(x)
    weight_oihw = _ensure_xpu_fp16_contig(weight_oihw)
    bias = _ensure_xpu_fp16_contig(bias)
    multiplier = _ensure_xpu_fp16_contig(multiplier)

    conv_out = baseline_conv_kernel(x, weight_oihw, bias)
    return baseline_eltwise_kernel(conv_out, multiplier, negative_slope=negative_slope)


# ============================================================
# Optimized path: pre-prepare + fused kernel
# ============================================================

def prepare_optimized_inputs(x_nchw, w_hwio, conv_bias, multiplier):
    """Pre-convert all tensors ONCE outside the benchmark loop."""
    x = _ensure_xpu_fp16(x_nchw).contiguous(memory_format=torch.channels_last)
    x_nhwc = x.permute(0, 2, 3, 1)
    w = _ensure_xpu_fp16(w_hwio).contiguous()
    cb = _ensure_xpu_fp16(conv_bias.view(-1)).contiguous()
    m = _ensure_xpu_fp16(multiplier.view(-1)).contiguous()

    N, C_in, H, W = x.shape
    KH, KW, _, C_out = w.shape
    OH = H - KH + 1
    OW = W - KW + 1
    M_total = N * OH * OW

    # Pre-allocate output as channels_last
    y = torch.empty((N, C_out, OH, OW), device=x.device,
                     dtype=torch.float16, memory_format=torch.channels_last)
    y_nhwc = y.permute(0, 2, 3, 1)

    sy_i = C_out
    sy_k = 1

    def grid(meta):
        return (meta['NUM_SMS'],)

    return dict(
        x_nhwc=x_nhwc, w_hwio=w, conv_bias=cb, mult=m,
        y=y, y_nhwc=y_nhwc,
        N=N, C_in=C_in, C_out=C_out, OH=OH, OW=OW, KH=KH, KW=KW,
        M_total=M_total, sy_i=sy_i, sy_k=sy_k,
        sxn=x_nhwc.stride(0), sxh=x_nhwc.stride(1), sxw=x_nhwc.stride(2),
        swkh=w.stride(0), swkw=w.stride(1), swci=w.stride(2), swco=w.stride(3),
        grid=grid,
    )


def optimized_kernel_function(prep, negative_slope=0.01):
    """Hot-path: single fused kernel launch. Zero allocs, zero conversions."""
    _fused_conv_mul_leakyrelu_gelu[prep['grid']](
        prep['x_nhwc'], prep['w_hwio'], prep['conv_bias'], prep['mult'], prep['y_nhwc'],
        prep['N'], prep['C_out'], prep['OH'], prep['OW'],
        prep['M_total'],
        prep['sxn'], prep['sxh'], prep['sxw'],
        prep['swkh'], prep['swkw'], prep['swci'], prep['swco'],
        prep['sy_i'], prep['sy_k'],
        negative_slope,
        KH=prep['KH'], KW=prep['KW'], C_IN=prep['C_in'],
    )
    return prep['y']


# ============================================================
# Spatial-tiled path: ALL block_ptr (paper Listing 4 approach)
# ============================================================

def prepare_spatial_inputs(x_nchw, w_hwio, conv_bias, multiplier):
    """Pre-convert tensors for spatial-tiled kernel."""
    x = _ensure_xpu_fp16(x_nchw).contiguous(memory_format=torch.channels_last)
    x_nhwc = x.permute(0, 2, 3, 1)  # contiguous NHWC
    w = _ensure_xpu_fp16(w_hwio).contiguous()
    cb = _ensure_xpu_fp16(conv_bias.view(-1)).contiguous()
    m = _ensure_xpu_fp16(multiplier.view(-1)).contiguous()

    N, C_in, H, W = x.shape
    KH, KW, _, C_out = w.shape
    OH = H - KH + 1
    OW = W - KW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device,
                     dtype=torch.float16, memory_format=torch.channels_last)
    y_nhwc = y.permute(0, 2, 3, 1)

    def grid(meta):
        return (N, OH, triton.cdiv(OW, meta['BLOCK_OW']))

    return dict(
        x_nhwc=x_nhwc, w_hwio=w, conv_bias=cb, mult=m,
        y=y, y_nhwc=y_nhwc,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W, OH=OH, OW=OW, KH=KH, KW=KW,
        swkh=w.stride(0), swkw=w.stride(1), swci=w.stride(2), swco=w.stride(3),
        grid=grid,
    )


def spatial_kernel_function(prep, negative_slope=0.01):
    """Spatial-tiled kernel: ALL block_ptr, zero scatter loads."""
    _fused_conv_spatial_tiled[prep['grid']](
        prep['x_nhwc'], prep['w_hwio'], prep['conv_bias'], prep['mult'], prep['y_nhwc'],
        prep['N'], prep['H'], prep['W'], prep['C_out'], prep['OH'], prep['OW'],
        prep['swkh'], prep['swkw'], prep['swci'], prep['swco'],
        negative_slope,
        KH=prep['KH'], KW=prep['KW'], C_IN=prep['C_in'],
    )
    return prep['y']


# ============================================================
# im2col path: pre-prepare + im2col + GEMM with fused epilogue
# ============================================================

def prepare_im2col_inputs(x_nchw, w_hwio, conv_bias, multiplier):
    """Pre-convert tensors + pre-allocate im2col buffer."""
    x = _ensure_xpu_fp16(x_nchw).contiguous(memory_format=torch.channels_last)
    x_nhwc = x.permute(0, 2, 3, 1)
    w = _ensure_xpu_fp16(w_hwio).contiguous()
    cb = _ensure_xpu_fp16(conv_bias.view(-1)).contiguous()
    m = _ensure_xpu_fp16(multiplier.view(-1)).contiguous()

    N, C_in, H, W = x.shape
    KH, KW, _, C_out = w.shape
    OH = H - KH + 1
    OW = W - KW + 1
    M_total = N * OH * OW
    K_TOTAL = KH * KW * C_in

    # Weight as flat [K_TOTAL, C_out] — just a view, zero cost
    w_flat = w.reshape(K_TOTAL, C_out).contiguous()

    # im2col buffer — limit to ~1GB to leave room for other allocs + autotune
    max_rows = min(M_total, (1 * 1024**3) // (K_TOTAL * 2))
    max_rows = (max_rows // 256) * 256  # align
    torch.xpu.empty_cache()
    col = torch.empty((max_rows, K_TOTAL), device=x.device, dtype=torch.float16)

    # Pre-allocate output
    y = torch.empty((N, C_out, OH, OW), device=x.device,
                     dtype=torch.float16, memory_format=torch.channels_last)
    y_nhwc = y.permute(0, 2, 3, 1)

    def gemm_grid(meta):
        return (meta['NUM_SMS'],)

    return dict(
        x_nhwc=x_nhwc, w_flat=w_flat, conv_bias=cb, mult=m,
        col=col, y=y, y_nhwc=y_nhwc,
        N=N, C_in=C_in, C_out=C_out, OH=OH, OW=OW, KH=KH, KW=KW,
        M_total=M_total, K_TOTAL=K_TOTAL,
        sxn=x_nhwc.stride(0), sxh=x_nhwc.stride(1), sxw=x_nhwc.stride(2),
        gemm_grid=gemm_grid,
    )


def im2col_kernel_function(prep, negative_slope=0.01):
    """Chunked im2col + GEMM with fused epilogue."""
    M_total = prep['M_total']
    K_TOTAL = prep['K_TOTAL']
    C_out = prep['C_out']
    KH, KW = prep['KH'], prep['KW']
    N, C_in = prep['N'], prep['C_in']
    OH, OW = prep['OH'], prep['OW']
    chunk = prep['col'].shape[0]
    col = prep['col']

    for m0 in range(0, M_total, chunk):
        m_size = min(chunk, M_total - m0)

        # im2col: fill col[0:m_size] from x at global offset m0
        im2col_grid = (triton.cdiv(m_size, 128),)
        _im2col_nhwc_kernel[im2col_grid](
            prep['x_nhwc'], col,
            N, OH + KH - 1, OW + KW - 1, C_in,
            OH, OW,
            m0, m_size,
            prep['sxn'], prep['sxh'], prep['sxw'],
            KH=KH, KW=KW, C_IN=C_in,
            BLOCK_M=128, BLOCK_K=64,
        )

        # GEMM: col[0:m_size] @ w_flat -> y[m0:m0+m_size]
        _gemm_fused_epilogue[prep['gemm_grid']](
            col, prep['w_flat'], prep['conv_bias'], prep['mult'],
            prep['y_nhwc'],
            m_size, C_out,
            m0,  # y_m_offset
            K_TOTAL=K_TOTAL,
            sy_i=C_out, sy_k=1,
            negative_slope=negative_slope,
        )

    return prep['y']


# ============================================================
# Benchmark / correctness
# ============================================================

@dataclass
class CompareResult:
    name: str
    max_abs_diff: float
    mean_abs_diff: float
    allclose: bool
    ms: float


def benchmark(fn, warmup=10, iters=30):
    for _ in range(warmup):
        _ = fn()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    return out, (t1 - t0) * 1000.0 / iters


def verify_close(name, got, ref, atol=2e-2, rtol=2e-2):
    diff = (got.float() - ref.float()).abs()
    return CompareResult(
        name=name,
        max_abs_diff=diff.max().item(),
        mean_abs_diff=diff.mean().item(),
        allclose=torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol),
        ms=0.0,
    )


def main():
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU is required."

    torch.manual_seed(0)

    model = Model(in_channels, out_channels, kernel_size, multiplier_shape).eval()

    x = torch.rand(batch_size, in_channels, height, width)

    # Channels-last from the start
    model_xpu = model.to("xpu", dtype=torch.float16).to(memory_format=torch.channels_last)
    x_xpu = x.to("xpu", dtype=torch.float16).contiguous(memory_format=torch.channels_last)

    # Shared parameters
    weight_hwio = _pack_weight_oihw_to_hwio(model_xpu.conv.weight)
    bias = _ensure_xpu_fp16_contig(model_xpu.conv.bias)
    multiplier = _ensure_xpu_fp16_contig(model_xpu.multiplier)

    negative_slope = float(model_xpu.leaky_relu.negative_slope)

    # torch.compile
    compiled_model = torch.compile(model_xpu)

    # Pre-prepare spatial inputs
    spatial_prep = prepare_spatial_inputs(x_xpu, weight_hwio, bias, multiplier)

    with torch.no_grad():
        ref_fn = lambda: model_xpu(x_xpu)
        compiled_fn = lambda: compiled_model(x_xpu)
        spatial_fn = lambda: spatial_kernel_function(spatial_prep, negative_slope=negative_slope)

        ref_out, ref_ms = benchmark(ref_fn, warmup=5, iters=10)
        spatial_out, spatial_ms = benchmark(spatial_fn, warmup=10, iters=30)
        compiled_out, compiled_ms = benchmark(compiled_fn, warmup=5, iters=10)

    spatial_res = verify_close("spatial", spatial_out, ref_out)
    compiled_res = verify_close("compiled", compiled_out, ref_out)
    spatial_res.ms = spatial_ms
    compiled_res.ms = compiled_ms

    # FLOPs: conv2d = 2 * N * OH * OW * C_out * C_in * KH * KW
    OH = height - kernel_size + 1
    OW = width - kernel_size + 1
    flops = 2.0 * batch_size * OH * OW * out_channels * in_channels * kernel_size * kernel_size

    # Data moved: input + output + weights (fp16 = 2 bytes)
    data_bytes = (64*256*256*64*2) + (64*254*254*64*2) + (3*3*64*64*2)

    results = [
        ("Reference", ref_ms, None),
        ("Triton", spatial_ms, spatial_res),
        ("Compiled", compiled_ms, compiled_res),
    ]

    print(f"\n{'='*80}")
    print(f"  Roofline Analysis -- Intel Arc B580 (110 TF/s fp16 peak, 456 GB/s)")
    print(f"{'='*80}")
    print(f"  FLOPs:        {flops/1e9:.2f} GFLOP")
    print(f"  Data moved:   {data_bytes/1e6:.1f} MB")
    print(f"  Arith. intensity: {flops/data_bytes:.1f} FLOP/byte")
    print(f"{'='*80}")
    print(f"  {'Name':>12} {'Time':>8} {'TFLOPS':>8} {'BW(GB/s)':>10} {'vs eager':>10} {'vs compile':>10} {'correct':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for name, ms, res in results:
        tflops = flops / (ms * 1e-3) / 1e12
        bw = data_bytes / (ms * 1e-3) / 1e9
        vs_eager = ref_ms / ms
        vs_compile = compiled_ms / ms
        correct = "yes" if res is None or res.allclose else "no"
        print(f"  {name:>12} {ms:7.2f}ms {tflops:7.2f} {bw:9.1f} {vs_eager:9.2f}x {vs_compile:9.2f}x {correct:>8}")

    return {
        "reference_ms": ref_ms,
        "spatial": spatial_res,
        "compiled": compiled_res,
    }


if __name__ == "__main__":
    main()
