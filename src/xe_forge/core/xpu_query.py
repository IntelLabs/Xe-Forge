"""
Intel XPU Configuration Query Utility

Queries Intel GPU/XPU hardware information to pass optimal parameters to the optimizer.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class XPUDeviceInfo:
    """Intel XPU device information."""

    # Basic info
    name: str = "Unknown"
    device_id: int = 0
    vendor: str = "Intel"
    driver_version: str = "Unknown"

    # Compute capabilities
    max_compute_units: int = 0  # Execution Units (EUs)
    max_subgroup_size: int = 32  # Similar to warp size
    max_work_group_size: int = 1024
    max_num_sub_groups: int = 0

    # Memory
    global_mem_size_gb: float = 0.0
    local_mem_size_kb: int = 0
    max_mem_alloc_size_gb: float = 0.0

    # Architecture specific
    gpu_eu_count: int = 0
    gpu_subslice_count: int = 0
    gpu_slice_count: int = 0
    has_fp64: bool = False
    has_fp16: bool = True
    has_bf16: bool = False

    # Recommended kernel parameters
    recommended_num_warps: int = 32
    recommended_grf_mode: str = "large"  # "large" (256) or "small" (128)
    recommended_tile_m: int = 256
    recommended_tile_n: int = 256
    recommended_tile_k: int = 32
    recommended_group_size_m: int = 4

    # Raw properties dict
    raw_properties: dict[str, Any] = field(default_factory=dict)


def query_xpu_via_torch() -> XPUDeviceInfo | None:
    """Query XPU info using PyTorch XPU backend."""
    try:
        import torch

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            logger.warning("torch.xpu not available")
            return None

        device_count = torch.xpu.device_count()
        if device_count == 0:
            return None

        info = XPUDeviceInfo()
        info.device_id = torch.xpu.current_device()

        # Get device name
        if hasattr(torch.xpu, "get_device_name"):
            info.name = torch.xpu.get_device_name(info.device_id)

        # Get device properties if available
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(info.device_id)

            if hasattr(props, "name"):
                info.name = props.name
            if hasattr(props, "total_memory"):
                info.global_mem_size_gb = props.total_memory / (1024**3)
            if hasattr(props, "max_compute_units"):
                info.max_compute_units = props.max_compute_units
            if hasattr(props, "gpu_eu_count"):
                info.gpu_eu_count = props.gpu_eu_count
            if hasattr(props, "gpu_subslice_count"):
                info.gpu_subslice_count = props.gpu_subslice_count
            if hasattr(props, "max_work_group_size"):
                info.max_work_group_size = props.max_work_group_size
            if hasattr(props, "max_num_sub_groups"):
                info.max_num_sub_groups = props.max_num_sub_groups
            if hasattr(props, "sub_group_sizes"):
                # Usually returns list like [8, 16, 32]
                info.max_subgroup_size = max(props.sub_group_sizes) if props.sub_group_sizes else 32
            if hasattr(props, "has_fp64"):
                info.has_fp64 = props.has_fp64
            if hasattr(props, "has_fp16"):
                info.has_fp16 = props.has_fp16

            # Store raw properties
            for attr in dir(props):
                if not attr.startswith("_"):
                    try:
                        info.raw_properties[attr] = getattr(props, attr)
                    except Exception:
                        pass

        # Get memory info
        if hasattr(torch.xpu, "get_device_capability"):
            cap = torch.xpu.get_device_capability(info.device_id)
            info.raw_properties["capability"] = cap

        # Infer architecture-specific recommendations
        info = _set_recommendations(info)

        return info

    except ImportError:
        logger.warning("PyTorch not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to query XPU via torch: {e}")
        return None


def query_xpu_via_xpu_smi() -> XPUDeviceInfo | None:
    """Query XPU info using xpu-smi tool."""
    try:
        # xpu-smi discovery
        result = subprocess.run(
            ["xpu-smi", "discovery", "--json"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            # Try without --json
            result = subprocess.run(
                ["xpu-smi", "discovery"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

        info = XPUDeviceInfo()

        # Try to parse JSON output
        try:
            data = json.loads(result.stdout)
            if isinstance(data, list) and len(data) > 0:
                device = data[0]
                info.name = device.get("device_name", "Unknown")
                info.device_id = device.get("device_id", 0)

                if "memory" in device:
                    mem = device["memory"]
                    if "physical_size" in mem:
                        info.global_mem_size_gb = mem["physical_size"] / (1024**3)

                info.raw_properties = device
        except json.JSONDecodeError:
            # Parse text output
            output = result.stdout
            for line in output.split("\n"):
                if "Device Name" in line:
                    info.name = line.split(":")[-1].strip()
                elif "Device ID" in line:
                    try:
                        info.device_id = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
            info.raw_properties["xpu_smi_output"] = output

        # Get additional stats
        try:
            stats_result = subprocess.run(
                ["xpu-smi", "stats", "-d", "0"], capture_output=True, text=True, timeout=10
            )
            if stats_result.returncode == 0:
                info.raw_properties["xpu_smi_stats"] = stats_result.stdout
        except Exception:
            pass

        info = _set_recommendations(info)
        return info

    except FileNotFoundError:
        logger.debug("xpu-smi not found")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("xpu-smi timed out")
        return None
    except Exception as e:
        logger.warning(f"Failed to query XPU via xpu-smi: {e}")
        return None


def _set_recommendations(info: XPUDeviceInfo) -> XPUDeviceInfo:
    """Set recommended kernel parameters based on detected hardware."""

    name_lower = info.name.lower()

    # Detect Intel GPU architecture
    if any(x in name_lower for x in ["arc", "a770", "a750", "a580", "a380", "a310"]):
        # Intel Arc (Alchemist) - Xe-HPG
        info.recommended_num_warps = 32
        info.recommended_grf_mode = "large"  # 256 registers
        info.recommended_tile_m = 256
        info.recommended_tile_n = 256
        info.recommended_tile_k = 32
        info.recommended_group_size_m = 4
        info.has_bf16 = True
        info.has_fp16 = True

    elif any(x in name_lower for x in ["max", "pvc", "ponte vecchio", "data center gpu"]):
        # Intel Data Center GPU Max (Ponte Vecchio) - Xe-HPC
        info.recommended_num_warps = 32
        info.recommended_grf_mode = "large"
        info.recommended_tile_m = 256
        info.recommended_tile_n = 256
        info.recommended_tile_k = 32
        info.recommended_group_size_m = 8  # More SMs, can use larger group
        info.has_bf16 = True
        info.has_fp16 = True
        info.has_fp64 = True  # PVC has good FP64

    elif any(x in name_lower for x in ["flex", "170", "140"]):
        # Intel Data Center GPU Flex (Arctic Sound-M)
        info.recommended_num_warps = 32
        info.recommended_grf_mode = "large"
        info.recommended_tile_m = 128
        info.recommended_tile_n = 128
        info.recommended_tile_k = 32
        info.recommended_group_size_m = 4
        info.has_bf16 = True

    elif any(x in name_lower for x in ["iris", "uhd", "integrated"]):
        # Integrated graphics - more conservative
        info.recommended_num_warps = 16
        info.recommended_grf_mode = "small"  # 128 registers
        info.recommended_tile_m = 64
        info.recommended_tile_n = 64
        info.recommended_tile_k = 32
        info.recommended_group_size_m = 2

    else:
        # Default for unknown Intel GPU
        info.recommended_num_warps = 32
        info.recommended_grf_mode = "large"
        info.recommended_tile_m = 128
        info.recommended_tile_n = 128
        info.recommended_tile_k = 32
        info.recommended_group_size_m = 4

    # Adjust based on memory size
    if info.global_mem_size_gb > 0:
        if info.global_mem_size_gb >= 16:
            # High-end GPU, can use larger tiles
            info.recommended_tile_m = max(info.recommended_tile_m, 256)
            info.recommended_tile_n = max(info.recommended_tile_n, 256)
        elif info.global_mem_size_gb < 4:
            # Limited memory, use smaller tiles
            info.recommended_tile_m = min(info.recommended_tile_m, 64)
            info.recommended_tile_n = min(info.recommended_tile_n, 64)

    # Adjust based on compute units
    if info.max_compute_units > 0:
        if info.max_compute_units >= 512:
            # Many EUs, can parallelize more
            info.recommended_group_size_m = 8
        elif info.max_compute_units < 128:
            # Fewer EUs
            info.recommended_group_size_m = 2

    return info


def get_xpu_config(device_id: int = 0) -> XPUDeviceInfo:
    """
    Query Intel XPU configuration.

    Tries in order:
    1. PyTorch XPU backend
    2. xpu-smi tool

    Returns best available info, or defaults if nothing works.
    """
    info = None

    # Try PyTorch first (preferred)
    result = query_xpu_via_torch()
    if result and result.name != "Unknown":
        logger.info(f"Got XPU info via torch.xpu: {result.name}")
        result.raw_properties["query_method"] = "torch.xpu"
        return result
    elif result:
        info = result
        info.raw_properties["query_method"] = "torch.xpu"

    # Try xpu-smi
    result = query_xpu_via_xpu_smi()
    if result and result.name != "Unknown":
        logger.info(f"Got XPU info via xpu-smi: {result.name}")
        result.raw_properties["query_method"] = "xpu-smi"
        return result
    elif result and info is None:
        info = result
        info.raw_properties["query_method"] = "xpu-smi"

    if info is None:
        logger.warning("Could not query XPU info, using defaults")
        info = XPUDeviceInfo()
        info.raw_properties["query_method"] = "defaults"

    return info


def get_xpu_config_dict(device_id: int = 0) -> dict[str, Any]:
    """
    Get XPU configuration as a dictionary for the optimizer.

    Returns dict compatible with xpu_config parameter.
    NOTE: These are hardware defaults. Use get_optimal_params() for shape-aware tuning.
    """
    info = get_xpu_config(device_id)

    return {
        "device": "xpu",
        "device_name": info.name,
        "grf_mode": info.recommended_grf_mode,
        "num_warps": info.recommended_num_warps,
        "num_stages": 2,  # Default for Intel
        "BLOCK_SIZE_M": info.recommended_tile_m,
        "BLOCK_SIZE_N": info.recommended_tile_n,
        "BLOCK_SIZE_K": info.recommended_tile_k,
        "GROUP_SIZE_M": info.recommended_group_size_m,
        # Hardware info
        "max_compute_units": info.max_compute_units,
        "global_mem_gb": info.global_mem_size_gb,
        "has_fp16": info.has_fp16,
        "has_bf16": info.has_bf16,
        "has_fp64": info.has_fp64,
    }


def get_optimal_params(
    M: int,
    N: int,
    K: int,
    device_id: int = 0,
    dtype: str = "float16",
) -> dict[str, Any]:
    """
    Get optimal kernel parameters based on problem shape AND hardware.

    This provides shape-aware recommendations, not just hardware defaults.

    Args:
        M: First matrix dimension (rows of A, rows of C)
        N: Second matrix dimension (cols of B, cols of C)
        K: Inner dimension (cols of A, rows of B)
        device_id: XPU device ID
        dtype: Data type (float16, bfloat16, float32)

    Returns:
        Dict with recommended kernel parameters
    """
    info = get_xpu_config(device_id)

    # Start with hardware defaults
    params = {
        "device": "xpu",
        "device_name": info.name,
        "grf_mode": info.recommended_grf_mode,
        "num_warps": info.recommended_num_warps,
        "num_stages": 2,
        "BLOCK_SIZE_M": info.recommended_tile_m,
        "BLOCK_SIZE_N": info.recommended_tile_n,
        "BLOCK_SIZE_K": info.recommended_tile_k,
        "GROUP_SIZE_M": info.recommended_group_size_m,
    }

    # === SHAPE-AWARE TUNING ===

    # 1. Tile sizes should not exceed problem dimensions
    #    (wastes threads and causes boundary overhead)
    params["BLOCK_SIZE_M"] = _clamp_to_power_of_2(min(params["BLOCK_SIZE_M"], M))
    params["BLOCK_SIZE_N"] = _clamp_to_power_of_2(min(params["BLOCK_SIZE_N"], N))
    params["BLOCK_SIZE_K"] = _clamp_to_power_of_2(min(params["BLOCK_SIZE_K"], K))

    # 2. For very small dimensions, use smaller tiles
    if M <= 32:
        params["BLOCK_SIZE_M"] = min(32, M)
    if N <= 32:
        params["BLOCK_SIZE_N"] = min(32, N)
    if K <= 32:
        params["BLOCK_SIZE_K"] = min(32, K)

    # 3. For skinny matrices, adjust tile ratios
    #    e.g., M=16384, N=64 -> BLOCK_M should be larger, BLOCK_N smaller
    if M > 4 * N and N < 128:
        # Tall-skinny: favor M tiles
        params["BLOCK_SIZE_M"] = min(256, _clamp_to_power_of_2(M // 4))
        params["BLOCK_SIZE_N"] = max(16, _clamp_to_power_of_2(N))
    elif N > 4 * M and M < 128:
        # Short-wide: favor N tiles
        params["BLOCK_SIZE_M"] = max(16, _clamp_to_power_of_2(M))
        params["BLOCK_SIZE_N"] = min(256, _clamp_to_power_of_2(N // 4))

    # 4. Adjust K tile based on memory pressure
    #    Larger BLOCK_K = fewer iterations but more register pressure
    bytes_per_elem = 2 if dtype in ["float16", "bfloat16"] else 4
    tile_memory_kb = (
        params["BLOCK_SIZE_M"] * params["BLOCK_SIZE_K"] * bytes_per_elem
        + params["BLOCK_SIZE_K"] * params["BLOCK_SIZE_N"] * bytes_per_elem
        + params["BLOCK_SIZE_M"] * params["BLOCK_SIZE_N"] * 4  # Accumulator always fp32
    ) / 1024

    # If tile needs too much memory, reduce K
    max_tile_memory_kb = 64 if info.recommended_grf_mode == "large" else 32
    while tile_memory_kb > max_tile_memory_kb and params["BLOCK_SIZE_K"] > 16:
        params["BLOCK_SIZE_K"] //= 2
        tile_memory_kb = (
            params["BLOCK_SIZE_M"] * params["BLOCK_SIZE_K"] * bytes_per_elem
            + params["BLOCK_SIZE_K"] * params["BLOCK_SIZE_N"] * bytes_per_elem
            + params["BLOCK_SIZE_M"] * params["BLOCK_SIZE_N"] * 4
        ) / 1024

    # 5. GROUP_SIZE_M for L2 cache locality (swizzling)
    #    Should create enough groups to saturate SMs but not too many
    num_m_tiles = (M + params["BLOCK_SIZE_M"] - 1) // params["BLOCK_SIZE_M"]
    num_n_tiles = (N + params["BLOCK_SIZE_N"] - 1) // params["BLOCK_SIZE_N"]
    total_tiles = num_m_tiles * num_n_tiles

    if total_tiles < 16:
        # Very few tiles, no swizzling benefit
        params["GROUP_SIZE_M"] = 1
    elif num_m_tiles < 4:
        # Few M tiles, small groups
        params["GROUP_SIZE_M"] = max(1, num_m_tiles)
    elif info.max_compute_units > 0:
        # Try to have ~4 groups per SM for good occupancy
        target_groups = info.max_compute_units * 4
        params["GROUP_SIZE_M"] = max(
            1, min(16, num_m_tiles // max(1, target_groups // num_n_tiles))
        )
    else:
        # Default heuristic
        params["GROUP_SIZE_M"] = min(8, max(1, num_m_tiles // 4))

    # 6. num_warps based on tile size
    #    More threads for larger tiles, but diminishing returns
    tile_elements = params["BLOCK_SIZE_M"] * params["BLOCK_SIZE_N"]
    if tile_elements >= 256 * 256:
        params["num_warps"] = 32
    elif tile_elements >= 128 * 128:
        params["num_warps"] = 16
    elif tile_elements >= 64 * 64:
        params["num_warps"] = 8
    else:
        params["num_warps"] = 4

    # Cap by hardware max
    params["num_warps"] = min(params["num_warps"], info.recommended_num_warps)

    # 7. num_stages (software pipelining)
    #    More stages = more memory but hides latency
    if K >= 1024:
        params["num_stages"] = 3  # Long K, benefit from pipelining
    elif K >= 256:
        params["num_stages"] = 2
    else:
        params["num_stages"] = 1  # Short K, pipelining overhead not worth it

    # Add problem info to config
    params["problem_shape"] = {"M": M, "N": N, "K": K}
    params["total_tiles"] = total_tiles
    params["estimated_tile_memory_kb"] = round(tile_memory_kb, 2)

    return params


def _clamp_to_power_of_2(x: int) -> int:
    """Round down to nearest power of 2, minimum 16."""
    if x <= 16:
        return 16
    # Find largest power of 2 <= x
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def get_autotune_configs(
    M: int,
    N: int,
    K: int,
    device_id: int = 0,
) -> list[dict[str, Any]]:
    """
    Generate multiple configs for Triton autotuning.

    Returns a list of configurations to try, ordered by expected performance.

    Args:
        M, N, K: Problem dimensions
        device_id: XPU device ID

    Returns:
        List of config dicts suitable for @triton.autotune
    """
    info = get_xpu_config(device_id)
    configs = []

    # Base config from shape analysis
    base = get_optimal_params(M, N, K, device_id)
    configs.append(
        {
            "BLOCK_SIZE_M": base["BLOCK_SIZE_M"],
            "BLOCK_SIZE_N": base["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": base["BLOCK_SIZE_K"],
            "GROUP_SIZE_M": base["GROUP_SIZE_M"],
            "num_warps": base["num_warps"],
            "num_stages": base["num_stages"],
        }
    )

    # Generate variations
    block_m_options = [b for b in [64, 128, 256] if b <= M]
    block_n_options = [b for b in [64, 128, 256] if b <= N]
    block_k_options = [b for b in [32, 64] if b <= K]

    for bm in block_m_options:
        for bn in block_n_options:
            for bk in block_k_options:
                # Skip base config (already added)
                if (
                    bm == base["BLOCK_SIZE_M"]
                    and bn == base["BLOCK_SIZE_N"]
                    and bk == base["BLOCK_SIZE_K"]
                ):
                    continue

                # Calculate appropriate GROUP_SIZE_M
                num_m_tiles = (M + bm - 1) // bm
                gsm = min(8, max(1, num_m_tiles // 4))

                # Calculate appropriate num_warps
                tile_elem = bm * bn
                if tile_elem >= 256 * 256:
                    nw = 32
                elif tile_elem >= 128 * 128:
                    nw = 16
                elif tile_elem >= 64 * 64:
                    nw = 8
                else:
                    nw = 4
                nw = min(nw, info.recommended_num_warps)

                config = {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "BLOCK_SIZE_K": bk,
                    "GROUP_SIZE_M": gsm,
                    "num_warps": nw,
                    "num_stages": 2,
                }

                if config not in configs:
                    configs.append(config)

    return configs[:12]  # Limit to reasonable number for autotuning


# =============================================================================
# Pipeline Integration Functions
# =============================================================================


def extract_mnk_from_shapes(
    input_shapes: list[tuple[int, ...]],
    kernel_type: str = "gemm",
) -> tuple[int | None, int | None, int | None]:
    """
    Extract M, N, K dimensions from input shapes.

    Supports common patterns:
    - GEMM: A[M,K], B[K,N] -> C[M,N]
    - GEMM with batch: A[B,M,K], B[B,K,N]
    - MatMul variants
    - Attention: Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D] -> M=S, N=S, K=D
    - Single input: infer from largest dimensions
    - 4D tensors: [B,H,M,K] style

    Args:
        input_shapes: List of input tensor shapes
        kernel_type: Type of kernel (gemm, matmul, attention, etc.)

    Returns:
        Tuple of (M, N, K) or (None, None, None) if cannot determine
    """
    if not input_shapes:
        logger.debug("No input shapes provided for M,N,K extraction")
        return None, None, None

    # --- Single input shape ---
    if len(input_shapes) == 1:
        shape = input_shapes[0]
        if len(shape) == 4:
            # Likely [B, H, SeqLen, HeadDim] (attention) or [B, C, H, W] (conv)
            # Use last two dims as M, K; M=N for self-attention
            _, _, M, K = shape
            logger.info(f"Single 4D input {shape}: inferring M={M}, N={M}, K={K}")
            return M, M, K
        elif len(shape) == 3:
            # Likely [B, SeqLen, Dim] or [B, M, K]
            _, M, K = shape
            logger.info(f"Single 3D input {shape}: inferring M={M}, N={M}, K={K}")
            return M, M, K
        elif len(shape) == 2:
            M, K = shape
            logger.info(f"Single 2D input {shape}: inferring M={M}, N={M}, K={K}")
            return M, M, K
        else:
            # 1D or 5D+ — use largest dim
            largest = max(shape)
            logger.info(f"Single input {shape}: using largest dim={largest}")
            return largest, largest, largest
        return None, None, None

    # --- 3+ inputs: check for attention pattern Q, K, V ---
    if len(input_shapes) >= 3:
        s0, s1, s2 = input_shapes[0], input_shapes[1], input_shapes[2]

        # Attention: Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]
        if len(s0) == 4 and len(s1) == 4 and len(s2) == 4:
            _, _, seq_q, head_dim = s0
            _, _, seq_k, _ = s1
            logger.info(
                f"Attention pattern detected: Q{s0}, K{s1}, V{s2} "
                f"-> M={seq_q}, N={seq_k}, K={head_dim}"
            )
            return seq_q, seq_k, head_dim

        # Attention: Q[B,S,D], K[B,S,D], V[B,S,D]
        if len(s0) == 3 and len(s1) == 3 and len(s2) == 3:
            _, seq_q, head_dim = s0
            _, seq_k, _ = s1
            logger.info(
                f"Attention pattern (3D): Q{s0}, K{s1}, V{s2} -> M={seq_q}, N={seq_k}, K={head_dim}"
            )
            return seq_q, seq_k, head_dim

    # --- 2 inputs: standard GEMM ---
    shape_a = input_shapes[0]
    shape_b = input_shapes[1]

    # Handle different shape formats
    if len(shape_a) == 2 and len(shape_b) == 2:
        # Standard 2D GEMM: A[M,K] @ B[K,N]
        M, K1 = shape_a
        K2, N = shape_b

        if K1 == K2:
            return M, N, K1
        else:
            # Maybe B is transposed: A[M,K] @ B[N,K].T
            if K1 == N:
                return M, K2, K1
            logger.warning(f"Shape mismatch: A{shape_a} B{shape_b}, K dimensions don't match")
            return M, shape_b[1], K1  # Best guess

    elif len(shape_a) == 3 and len(shape_b) == 3:
        # Batched GEMM: A[B,M,K] @ B[B,K,N]
        _, M, K1 = shape_a
        _, K2, N = shape_b

        if K1 == K2:
            return M, N, K1
        else:
            return M, shape_b[2], K1

    elif len(shape_a) == 4 and len(shape_b) == 4:
        # 4D batched: A[B,H,M,K] @ B[B,H,K,N]
        _, _, M, K1 = shape_a
        _, _, K2, N = shape_b

        if K1 == K2:
            return M, N, K1
        else:
            # Could be attention: Q[B,H,S,D] @ K[B,H,S,D].T
            # In this case K dims won't match; use last two of first tensor
            logger.info(f"4D pattern A{shape_a} B{shape_b}: using M={M}, N={shape_b[2]}, K={K1}")
            return M, shape_b[2], K1

    elif len(shape_a) == 2 and len(shape_b) == 1:
        # Matrix-vector: A[M,K] @ x[K]
        M, K = shape_a
        return M, 1, K

    elif len(shape_a) == 1 and len(shape_b) == 2:
        # Vector-matrix: x[K] @ B[K,N]
        K1 = shape_a[0]
        K2, N = shape_b
        return 1, N, K1

    else:
        # Unknown pattern - try to extract reasonable values
        logger.info(f"Unknown shape pattern: A{shape_a} B{shape_b}, using heuristic")

        # Use largest dimensions as M, N, K
        all_dims = list(shape_a) + list(shape_b)
        all_dims.sort(reverse=True)

        if len(all_dims) >= 3:
            return all_dims[0], all_dims[1], all_dims[2]
        elif len(all_dims) == 2:
            return all_dims[0], all_dims[1], all_dims[0]
        else:
            return None, None, None


def get_xpu_config_for_pipeline(
    input_shapes: list[tuple[int, ...]] | None = None,
    config: Any | None = None,  # xe_forge.config.Config
    dtype: str = "float16",
    device_id: int = 0,
) -> dict[str, Any]:
    """
    Get XPU configuration for the optimization pipeline.

    If input_shapes are provided, uses shape-aware tuning.
    Otherwise falls back to hardware defaults or config values.

    Args:
        input_shapes: Input tensor shapes (from spec file)
        config: Config object with fallback values
        dtype: Data type string
        device_id: XPU device ID

    Returns:
        Dict with XPU configuration for optimizer

    Usage in pipeline.py:
        xpu_config = get_xpu_config_for_pipeline(
            input_shapes=input_shapes,
            config=self.config,
            dtype=effective_target_dtype or "float16",
        )
    """
    # Try shape-aware params first
    if input_shapes and len(input_shapes) >= 1:
        M, N, K = extract_mnk_from_shapes(input_shapes)

        if M is not None and N is not None and K is not None:
            logger.info(f"Using shape-aware XPU params for M={M}, N={N}, K={K}")
            params = get_optimal_params(M, N, K, device_id, dtype)

            # Add hardware info
            hw_info = get_xpu_config(device_id)
            params["max_compute_units"] = hw_info.max_compute_units
            params["global_mem_gb"] = hw_info.global_mem_size_gb
            params["has_fp16"] = hw_info.has_fp16
            params["has_bf16"] = hw_info.has_bf16
            params["has_fp64"] = hw_info.has_fp64

            return params

    # Fall back to hardware defaults
    logger.info("Using hardware default XPU params (no shape info)")
    hw_config = get_xpu_config_dict(device_id)

    # Override with config values if provided
    if config and hasattr(config, "xpu"):
        xpu_cfg = config.xpu
        hw_config.update(
            {
                "device": getattr(xpu_cfg, "device", hw_config.get("device", "xpu")),
                "grf_mode": getattr(xpu_cfg, "grf_mode", hw_config.get("grf_mode", "large")),
                "num_warps": getattr(xpu_cfg, "default_num_warps", hw_config.get("num_warps", 32)),
                "num_stages": getattr(
                    xpu_cfg, "default_num_stages", hw_config.get("num_stages", 2)
                ),
                "BLOCK_SIZE_M": getattr(
                    xpu_cfg, "preferred_tile_m", hw_config.get("BLOCK_SIZE_M", 256)
                ),
                "BLOCK_SIZE_N": getattr(
                    xpu_cfg, "preferred_tile_n", hw_config.get("BLOCK_SIZE_N", 256)
                ),
                "BLOCK_SIZE_K": getattr(
                    xpu_cfg, "preferred_tile_k", hw_config.get("BLOCK_SIZE_K", 32)
                ),
                "GROUP_SIZE_M": getattr(xpu_cfg, "group_size_m", hw_config.get("GROUP_SIZE_M", 4)),
            }
        )

    return hw_config


def format_xpu_config_for_llm(xpu_config: dict[str, Any]) -> str:
    """
    Format XPU config as a string for the LLM prompt.

    Includes both the recommended values and the reasoning.
    """
    lines = [
        "Intel XPU Configuration:",
        "=" * 40,
    ]

    # Problem shape if available
    if "problem_shape" in xpu_config:
        shape = xpu_config["problem_shape"]
        lines.append(f"Problem Shape: M={shape['M']}, N={shape['N']}, K={shape['K']}")
        lines.append("")

    # Recommended parameters
    lines.append("RECOMMENDED KERNEL PARAMETERS:")
    lines.append(f"  BLOCK_SIZE_M: {xpu_config.get('BLOCK_SIZE_M', 256)}")
    lines.append(f"  BLOCK_SIZE_N: {xpu_config.get('BLOCK_SIZE_N', 256)}")
    lines.append(f"  BLOCK_SIZE_K: {xpu_config.get('BLOCK_SIZE_K', 32)}")
    lines.append(f"  GROUP_SIZE_M: {xpu_config.get('GROUP_SIZE_M', 4)}")
    lines.append(f"  num_warps: {xpu_config.get('num_warps', 32)}")
    lines.append(f"  num_stages: {xpu_config.get('num_stages', 2)}")
    lines.append(f"  grf_mode: {xpu_config.get('grf_mode', 'large')}")

    # Extra info if available
    if "total_tiles" in xpu_config:
        lines.append("")
        lines.append("ANALYSIS:")
        lines.append(f"  Total tiles: {xpu_config['total_tiles']}")
        lines.append(f"  Tile memory: {xpu_config.get('estimated_tile_memory_kb', 'N/A')} KB")

    # Hardware info
    if xpu_config.get("device_name"):
        lines.append("")
        lines.append("HARDWARE:")
        lines.append(f"  Device: {xpu_config.get('device_name', 'Unknown')}")
        if xpu_config.get("max_compute_units"):
            lines.append(f"  Compute Units: {xpu_config['max_compute_units']}")
        if xpu_config.get("global_mem_gb"):
            lines.append(f"  Memory: {xpu_config['global_mem_gb']:.1f} GB")

    return "\n".join(lines)


def print_xpu_info(device_id: int = 0):
    """Print XPU information in human-readable format."""
    info = get_xpu_config(device_id)

    print("=" * 60)
    print("INTEL XPU CONFIGURATION")
    print("=" * 60)
    print(f"Device Name:         {info.name}")
    print(f"Device ID:           {info.device_id}")
    print(f"Driver Version:      {info.driver_version}")
    print()
    print("HARDWARE:")
    print(f"  Compute Units/EUs: {info.max_compute_units or 'Unknown'}")
    print(f"  EU Count:          {info.gpu_eu_count or 'Unknown'}")
    print(f"  Subslices:         {info.gpu_subslice_count or 'Unknown'}")
    print(f"  Slices:            {info.gpu_slice_count or 'Unknown'}")
    print(f"  Max Subgroup Size: {info.max_subgroup_size}")
    print(f"  Max Work Group:    {info.max_work_group_size}")
    print()
    print("MEMORY:")
    print(f"  Global Memory:     {info.global_mem_size_gb:.2f} GB")
    print(f"  Local Memory:      {info.local_mem_size_kb} KB")
    print()
    print("CAPABILITIES:")
    print(f"  FP16:              {'Yes' if info.has_fp16 else 'No'}")
    print(f"  BF16:              {'Yes' if info.has_bf16 else 'No'}")
    print(f"  FP64:              {'Yes' if info.has_fp64 else 'No'}")
    print()
    print("HARDWARE DEFAULTS (use get_optimal_params for shape-aware tuning):")
    print(f"  num_warps:         {info.recommended_num_warps}")
    print(f"  grf_mode:          {info.recommended_grf_mode}")
    print(f"  BLOCK_SIZE_M:      {info.recommended_tile_m}")
    print(f"  BLOCK_SIZE_N:      {info.recommended_tile_n}")
    print(f"  BLOCK_SIZE_K:      {info.recommended_tile_k}")
    print(f"  GROUP_SIZE_M:      {info.recommended_group_size_m}")
    print("=" * 60)

    if info.raw_properties.get("query_method"):
        print(f"\n(Info obtained via: {info.raw_properties['query_method']})")

    # Show example shape-aware params
    print("\n" + "=" * 60)
    print("EXAMPLE: Shape-aware parameters for different GEMM sizes")
    print("=" * 60)

    examples = [
        (4096, 4096, 4096, "Large square"),
        (64, 64, 64, "Small square"),
        (8192, 128, 4096, "Tall-skinny"),
        (128, 8192, 4096, "Short-wide"),
    ]

    for M, N, K, desc in examples:
        params = get_optimal_params(M, N, K, device_id)
        print(f"\n{desc} ({M}x{N}x{K}):")
        print(
            f"  BLOCK: {params['BLOCK_SIZE_M']}x{params['BLOCK_SIZE_N']}x{params['BLOCK_SIZE_K']}"
        )
        print(
            f"  num_warps={params['num_warps']}, GROUP_SIZE_M={params['GROUP_SIZE_M']}, stages={params['num_stages']}"
        )
        print(f"  tiles={params['total_tiles']}, tile_mem={params['estimated_tile_memory_kb']}KB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_xpu_info()
