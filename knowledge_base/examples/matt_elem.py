import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------------------------------------------
# KernelBench Model (adapted): forward() calls kernel_function()
# -------------------------------------------------------------------
class Model(nn.Module):
    """
    KernelBench wrapper for fused:
        out = min(x @ W^T + b, constant) - constant

    Init signature matches YAML inits:
        (IN_FEAT, OUT_FEAT, CONSTANT)

    Important:
    - Store weight as [N, K] (float16) and bias as [N] (float32), matching kernel_function expectations.
    - Pre-pack W^T once on XPU as contiguous [K, N] to ensure coalesced RHS loads.
    - Keep a Python float copy of constant to avoid device->host sync in hot path.
    - forward() calls kernel_function() directly (this is what KernelBench should time).
    """

    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Keep constant as a parameter (float32), and cache a Python float for kernel use
        self.constant = nn.Parameter(torch.tensor(float(constant), dtype=torch.float32))
        self.constant_value = float(constant)

        # Params: weight [N,K] in fp16, bias [N] in fp32
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float16)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features, dtype=torch.float32))

        # Reasonable init (similar to nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Internal flags to avoid repeated device/dtype moves
        self._moved_params = False
        # Cached packed W^T [K, N] contiguous on XPU
        self._weight_t = None

    def _move_params_once(self):
        # Move parameters to XPU once and ensure desired dtypes/contiguity
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU is not available")
        if (
            self.weight.device.type != "xpu"
            or self.weight.dtype != torch.float16
            or not self.weight.is_contiguous()
        ):
            self.weight.data = self.weight.data.to("xpu", dtype=torch.float16).contiguous()
        if (
            self.bias.device.type != "xpu"
            or self.bias.dtype != torch.float32
            or not self.bias.is_contiguous()
        ):
            self.bias.data = self.bias.data.to("xpu", dtype=torch.float32).contiguous()
        # Pre-pack W^T once as [K, N] contiguous on XPU (avoid per-forward repack)
        self._weight_t = self.weight.data.t().contiguous()
        # Keep constant as Python float; do NOT move to XPU to avoid device-host sync
        self._moved_params = True

    def forward(self, x):
        # KernelBench may provide CPU tensors; ensure float16 + xpu and contiguous.
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU is not available")

        if x.device.type != "xpu" or x.dtype != torch.float16 or not x.is_contiguous():
            x = x.to("xpu", dtype=torch.float16).contiguous()

        # Move params once (KernelBench often instantiates model on CPU)
        if not self._moved_params:
            self._move_params_once()

        # Expect [M, K]
        if x.ndim != 2:
            raise ValueError(f"Expected X to be 2D [BATCH, IN_FEAT], got {tuple(x.shape)}")

        # Use cached Python float constant and pre-packed W^T
        return kernel_function(x, self._weight_t, self.bias, self.constant_value)


# -------------------------------------------------------------------
# KernelBench input generators (match YAML below)
# -------------------------------------------------------------------


def get_init_inputs():
    IN_FEAT = 16384
    OUT_FEAT = 16384
    CONSTANT = 2.0
    return [IN_FEAT, OUT_FEAT, CONSTANT]


def get_inputs():
    # KernelBench often generates inputs on CPU; Model moves them to xpu.
    BATCH = 128
    IN_FEAT, _, _ = get_init_inputs()
    # Generate in float16 as target requires fp16 inputs
    X = torch.rand((BATCH, IN_FEAT), dtype=torch.float16)
    return [X]


# -------------------------------------------------------------------
# Triton kernel for fused Linear + Min + Sub (fp16 inputs/outputs, fp32 accumulation)
# Weight is pre-packed as W^T [K, N] for coalesced RHS access.
# Switch to 2D grid over (M_tiles, N_tiles); autotune warps and grf_mode for XPU.
# -------------------------------------------------------------------
def _get_xpu_autotune_configs():
    # Build a sweep over tile shapes, num_warps, and grf_mode variants.
    configs = []
    tile_shapes = [
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
    ]
    warps = [8, 16, 32]
    grf_modes = ["256", "512", "default", None]  # None -> omit key to allow backend default
    for ts in tile_shapes:
        for nw in warps:
            for gm in grf_modes:
                meta = dict(ts)
                if gm is not None:
                    meta["grf_mode"] = gm
                # Keep num_stages identical to avoid confounding (per issue guidance)
                cfg = triton.Config(meta, num_warps=nw, num_stages=2)
                configs.append(cfg)
    return configs


@triton.autotune(configs=_get_xpu_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_min_sub_kernel(
    x_ptr,  # [M, K], fp16
    wt_ptr,  # [K, N], fp16 (pre-packed W^T)
    b_ptr,  # [N], fp32
    o_ptr,  # [M, N], fp16
    M,
    N,
    K,  # dims
    stride_xm,
    stride_xk,
    stride_wtk,
    stride_wtn,
    stride_b,
    stride_om,
    stride_on,
    constant,  # scalar float
    # Meta-parameters (from autotune, no defaults here)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  # kept for autotune symmetry; not used in 2D grid
):
    # 2D grid over (M_tiles, N_tiles)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers for X: (M, K) -> [BLOCK_M, BLOCK_K] tiles
    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # Block pointers for W^T: (K, N) -> [BLOCK_K, BLOCK_N] tiles
    wt_bp = tl.make_block_ptr(
        base=wt_ptr,
        shape=(K, N),
        strides=(stride_wtk, stride_wtn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for _ in range(0, K, BLOCK_K):
        x_blk = tl.load(x_bp, boundary_check=(0, 1))
        wt_blk = tl.load(wt_bp, boundary_check=(0, 1))
        acc = tl.dot(x_blk.to(tl.float16), wt_blk.to(tl.float16), acc=acc)
        x_bp = tl.advance(x_bp, (0, BLOCK_K))
        wt_bp = tl.advance(wt_bp, (BLOCK_K, 0))

    # Add bias once per tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.load(b_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :].to(tl.float32)

    # Epilogue: min then subtract constant
    acc = tl.minimum(acc, constant)
    acc = acc - constant

    # Store
    o_bp = tl.make_block_ptr(
        base=o_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(o_bp, acc.to(tl.float16), boundary_check=(0, 1))


def kernel_function(input_tensor, linear_weight_t, linear_bias, constant):
    """
    Wrapper for fused Linear‐Min‐Sub on Intel XPU. Computes
        out = min(input @ W^T + b, constant) - constant
    using Triton kernels only.

    Mixed precision:
    - input, weight_t: float16
    - bias, constant: float32
    - accumulation: float32
    - output: float16

    Note:
    - linear_weight_t must be pre-packed as [K, N] contiguous (W transposed once).
    """
    # sanity checks
    assert isinstance(input_tensor, torch.Tensor), "input must be a tensor"
    assert isinstance(linear_weight_t, torch.Tensor), "weight_t must be a tensor"
    assert isinstance(linear_bias, torch.Tensor), "bias must be a tensor"
    # must be on XPU
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU is not available"
    assert input_tensor.device.type == "xpu", "input_tensor must be on XPU"
    assert linear_weight_t.device.type == "xpu", "weight_t must be on XPU"
    assert linear_bias.device.type == "xpu", "bias must be on XPU"

    # Ensure dtypes and contiguity (prefer contiguous tensors for Triton kernels)
    if input_tensor.dtype != torch.float16 or not input_tensor.is_contiguous():
        input_tensor = input_tensor.to(torch.float16).contiguous()
    if linear_weight_t.dtype != torch.float16 or not linear_weight_t.is_contiguous():
        linear_weight_t = linear_weight_t.to(torch.float16).contiguous()
    if linear_bias.dtype != torch.float32 or not linear_bias.is_contiguous():
        linear_bias = linear_bias.to(torch.float32).contiguous()

    # constants should be Python floats already
    c_val = float(constant)

    # dimensions
    M, K = input_tensor.shape
    K_wt, N = linear_weight_t.shape
    assert K_wt == K, (
        f"Packed weight_t must be [K,N], got {tuple(linear_weight_t.shape)}, expected K={K}"
    )
    assert linear_bias.shape[0] == N, (
        f"Bias must be length N, got {tuple(linear_bias.shape)} and N={N}"
    )

    # allocate output in fp16
    output = torch.empty((M, N), device="xpu", dtype=torch.float16)

    # 2D grid over tiles (M, N)
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # launch the kernel (autotune provides num_warps/num_stages/grf_mode)
    _linear_min_sub_kernel[grid](
        # pointers
        input_tensor,
        linear_weight_t,
        linear_bias,
        output,
        # dimensions
        M,
        N,
        K,
        # strides for input [M, K]
        input_tensor.stride(0),
        input_tensor.stride(1),
        # strides for packed W^T [K, N] (CRITICAL: use packed strides)
        linear_weight_t.stride(0),
        linear_weight_t.stride(1),
        # stride for bias [N]
        linear_bias.stride(0),
        # strides for output [M, N]
        output.stride(0),
        output.stride(1),
        # constant clamp value (Python float)
        c_val,
    )
    # synchronize to ensure completion
    return output
