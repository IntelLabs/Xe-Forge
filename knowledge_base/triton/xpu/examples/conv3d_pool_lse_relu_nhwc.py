"""
Conv3d(32->64, 3x3x3, pad=1) + MaxPool3d(2) + LogSumExp(dim=1) + ReLU
FULL Triton on Intel Arc B580 — no oneDNN

Spatially-tiled Conv3d (all block_ptr) + fused Pool+LSE+ReLU.
Padding in d/h: conditional skip. Padding in w: block_ptr boundary_check.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Reference model
# ============================================================

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x


batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


# ============================================================
# Spatially-tiled Conv3d: all block_ptr
# Grid: (n, od*OH+oh, ow_tile * cout_tiles + cout_tile)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OW': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['D', 'H', 'W', 'C_IN', 'C_OUT', 'OD', 'OH', 'OW'],
)
@triton.jit
def _conv3d_spatial_tiled(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N_batch, D, H, W, OD, OH, OW,
    sx_n, sx_d, sx_h,
    sw_kd, sw_kh, sw_kw, sw_ci, sw_co,
    sy_n, sy_d, sy_h,
    BLOCK_OW: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
):
    """Spatially-tiled Conv3d + bias. All block_ptr.

    Padding in d/h handled by conditional skip.
    Padding in w handled by block_ptr boundary_check (negative offsets -> zero).
    """
    n = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_wn = tl.program_id(2)

    od = pid_dh // OH
    oh = pid_dh % OH

    num_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    pid_ow = pid_wn % num_ow_tiles
    pid_n = pid_wn // num_ow_tiles
    ow0 = pid_ow * BLOCK_OW

    acc = tl.zeros((BLOCK_OW, BLOCK_N), dtype=tl.float32)

    x_n_base = x_ptr + n * sx_n

    for kd in range(KD):
        d_in = od + kd - PAD
        d_ok = (d_in >= 0) & (d_in < D)
        if d_ok:
            for kh in range(KH):
                h_in = oh + kh - PAD
                h_ok = (h_in >= 0) & (h_in < H)
                if h_ok:
                    x_dh_base = x_n_base + d_in * sx_d + h_in * sx_h

                    for kw in range(KW):
                        w_start = ow0 + kw - PAD

                        x_bp = tl.make_block_ptr(
                            base=x_dh_base,
                            shape=(W, C_IN),
                            strides=(C_IN, 1),
                            offsets=(w_start, 0),
                            block_shape=(BLOCK_OW, BLOCK_K),
                            order=(1, 0),
                        )

                        w_bp = tl.make_block_ptr(
                            base=w_ptr + kd * sw_kd + kh * sw_kh + kw * sw_kw,
                            shape=(C_IN, C_OUT),
                            strides=(sw_ci, sw_co),
                            offsets=(0, pid_n * BLOCK_N),
                            block_shape=(BLOCK_K, BLOCK_N),
                            order=(1, 0),
                        )

                        for c0 in range(0, C_IN, BLOCK_K):
                            x_tile = tl.load(x_bp, boundary_check=(0, 1), padding_option="zero")
                            w_tile = tl.load(w_bp, boundary_check=(0, 1), padding_option="zero")
                            acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
                            x_bp = tl.advance(x_bp, (0, BLOCK_K))
                            w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    # Bias
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_vals = tl.load(b_ptr + offs_n, mask=offs_n < C_OUT, other=0.0)
    acc += bias_vals[None, :]

    # Store: boundary-limited to prevent oh-row wrapping
    y_dh_base = y_ptr + n * sy_n + od * sy_d + oh * sy_h
    y_valid = OW - ow0
    y_bp = tl.make_block_ptr(
        base=y_dh_base,
        shape=(y_valid, C_OUT),
        strides=(C_OUT, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_OW, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_bp, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================
# Fused MaxPool3d(2x2x2) + LogSumExp(dim=1) + ReLU
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64,  'BLOCK_C': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64,  'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_C': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W_pool', 'H_pool', 'D_pool'],
)
@triton.jit
def _fused_pool_lse_relu(
    conv_ptr, y_ptr,
    N, C, D_conv, H_conv, W_conv,
    D_pool, H_pool, W_pool,
    sc_n, sc_d, sc_h, sc_w, sc_c,
    sy_n, sy_d, sy_h, sy_w,
    BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nd = tl.program_id(2)

    n = pid_nd // D_pool
    d_pool = pid_nd % D_pool
    h_pool = pid_h

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_pool

    neg_inf = -float("inf")
    m = tl.full((BLOCK_W,), neg_inf, dtype=tl.float32)
    s = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        pooled = tl.full((BLOCK_C, BLOCK_W), neg_inf, dtype=tl.float32)

        for dd in range(2):
            d_in = d_pool * 2 + dd
            for hh in range(2):
                h_in = h_pool * 2 + hh
                for ww in range(2):
                    w_in = offs_w * 2 + ww
                    ptrs = (conv_ptr + n * sc_n + d_in * sc_d + h_in * sc_h
                            + w_in[None, :] * sc_w + offs_c[:, None] * sc_c)
                    vals = tl.load(ptrs, mask=mask_c[:, None] & mask_w[None, :],
                                   other=neg_inf).to(tl.float32)
                    pooled = tl.maximum(pooled, vals)

        tile_m = tl.max(pooled, axis=0)
        new_m = tl.maximum(m, tile_m)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(pooled - new_m[None, :]), axis=0)
        m = new_m

    out = tl.maximum(tl.log(s) + m, 0.0)
    y_base = y_ptr + n * sy_n + d_pool * sy_d + h_pool * sy_h
    tl.store(y_base + offs_w * sy_w, out.to(tl.float16), mask=mask_w)


# ============================================================
# Helpers + pipeline
# ============================================================

def _ensure_xpu_fp16(x):
    if x.device.type != "xpu" or x.dtype != torch.float16:
        return x.to("xpu", dtype=torch.float16)
    return x


def triton_kernel_function(x_ndhwc, w_dhwio, bias, conv_out, conv_ndhwc, y_buf):
    """Full Triton pipeline: Conv3d + fused Pool+LSE+ReLU."""
    N_b = x_ndhwc.shape[0]
    D_x = x_ndhwc.shape[1]
    H_x = x_ndhwc.shape[2]
    W_x = x_ndhwc.shape[3]
    C_in = x_ndhwc.shape[4]
    KD, KH_w, KW_w = w_dhwio.shape[0], w_dhwio.shape[1], w_dhwio.shape[2]
    C_out = w_dhwio.shape[4]
    OD, OH, OW = D_x, H_x, W_x  # stride=1, pad=1

    def conv_grid(meta):
        return (N_b, OD * OH, triton.cdiv(OW, meta['BLOCK_OW']) * triton.cdiv(C_out, meta['BLOCK_N']))

    _conv3d_spatial_tiled[conv_grid](
        x_ndhwc, w_dhwio, bias, conv_ndhwc,
        N_b, D_x, H_x, W_x, OD, OH, OW,
        x_ndhwc.stride(0), x_ndhwc.stride(1), x_ndhwc.stride(2),
        w_dhwio.stride(0), w_dhwio.stride(1), w_dhwio.stride(2), w_dhwio.stride(3), w_dhwio.stride(4),
        conv_ndhwc.stride(0), conv_ndhwc.stride(1), conv_ndhwc.stride(2),
        KD=KD, KH=KH_w, KW=KW_w, PAD=1,
        C_IN=C_in, C_OUT=C_out,
    )

    D_pool, H_pool, W_pool = OD // 2, OH // 2, OW // 2
    sc = conv_out.stride()
    sy = y_buf.stride()

    pool_grid = lambda META: (triton.cdiv(W_pool, META['BLOCK_W']), H_pool, N_b * D_pool)

    _fused_pool_lse_relu[pool_grid](
        conv_out, y_buf,
        N_b, C_out, OD, OH, OW,
        D_pool, H_pool, W_pool,
        sc[0], sc[2], sc[3], sc[4], sc[1],
        sy[0], sy[2], sy[3], sy[4],
    )
    return y_buf


# ============================================================
# Benchmark
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
    assert hasattr(torch, "xpu") and torch.xpu.is_available()
    torch.manual_seed(0)

    model = Model(in_channels, out_channels, kernel_size, stride, padding).eval()
    x = torch.rand(batch_size, in_channels, depth, height, width)

    model_xpu = model.to("xpu", dtype=torch.float16).to(memory_format=torch.channels_last_3d)
    x_xpu = x.to("xpu", dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)

    # Prepare
    w_dhwio = _ensure_xpu_fp16(model_xpu.conv.weight).permute(2, 3, 4, 1, 0).contiguous()
    bias_vec = _ensure_xpu_fp16(model_xpu.conv.bias.view(-1)).contiguous()

    x_ndhwc = x_xpu.permute(0, 2, 3, 4, 1)
    N_b, C_in_v, D_x, H_x, W_x = x_xpu.shape
    C_out_v = w_dhwio.shape[4]

    conv_out = torch.empty((N_b, C_out_v, D_x, H_x, W_x), device=x_xpu.device,
                            dtype=torch.float16, memory_format=torch.channels_last_3d)
    conv_ndhwc = conv_out.permute(0, 2, 3, 4, 1)

    D_pool, H_pool, W_pool = D_x // 2, H_x // 2, W_x // 2
    y_buf = torch.empty((N_b, 1, D_pool, H_pool, W_pool), device=x_xpu.device, dtype=torch.float16)

    compiled_model = torch.compile(model_xpu)

    with torch.no_grad():
        ref_fn = lambda: model_xpu(x_xpu)
        compiled_fn = lambda: compiled_model(x_xpu)
        triton_fn = lambda: triton_kernel_function(x_ndhwc, w_dhwio, bias_vec, conv_out, conv_ndhwc, y_buf)

        ref_out, ref_ms = benchmark(ref_fn, warmup=3, iters=10)
        triton_out, triton_ms = benchmark(triton_fn, warmup=5, iters=20)
        compiled_out, compiled_ms = benchmark(compiled_fn, warmup=3, iters=10)

    triton_res = verify_close("triton", triton_out, ref_out)
    compiled_res = verify_close("compiled", compiled_out, ref_out)
    triton_res.ms = triton_ms
    compiled_res.ms = compiled_ms

    flops = 2.0 * batch_size * out_channels * depth * height * width * in_channels * kernel_size**3
    data_bytes = (batch_size*in_channels*depth*height*width*2) + (batch_size*1*D_pool*H_pool*W_pool*2) + (out_channels*in_channels*kernel_size**3*2)

    results = [
        ("Reference", ref_ms, None),
        ("Triton", triton_ms, triton_res),
        ("Compiled", compiled_ms, compiled_res),
    ]

    print(f"\n{'='*80}")
    print(f"  Roofline -- Intel Arc B580 (110 TF/s fp16 peak, 456 GB/s)")
    print(f"{'='*80}")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOP  |  Data: {data_bytes/1e6:.1f} MB  |  AI: {flops/data_bytes:.1f} FLOP/byte")
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


if __name__ == "__main__":
    main()
