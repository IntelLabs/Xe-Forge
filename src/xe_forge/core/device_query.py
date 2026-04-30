"""
Device Configuration Query Utility

Provides a generic device query interface that dispatches to device-specific
implementations (Intel XPU, NVIDIA CUDA, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Base device information shared across all device types."""

    name: str = "Unknown"
    device_id: int = 0
    device_type: str = "xpu"
    vendor: str = "Unknown"
    driver_version: str = "Unknown"

    max_compute_units: int = 0
    global_mem_size_gb: float = 0.0
    has_fp16: bool = True
    has_bf16: bool = False
    has_fp64: bool = False

    recommended_num_warps: int = 32
    recommended_tile_m: int = 128
    recommended_tile_n: int = 128
    recommended_tile_k: int = 32

    raw_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDADeviceInfo(DeviceInfo):
    """NVIDIA CUDA device information."""

    device_type: str = "cuda"
    vendor: str = "NVIDIA"

    sm_count: int = 0
    compute_capability: tuple[int, int] = (0, 0)
    max_threads_per_sm: int = 0
    max_shared_memory_per_block_kb: int = 0
    warp_size: int = 32

    recommended_num_warps: int = 4
    recommended_tile_m: int = 128
    recommended_tile_n: int = 128
    recommended_tile_k: int = 32


def query_cuda_via_torch() -> CUDADeviceInfo | None:
    """Query CUDA device info using PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return None

        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None

        info = CUDADeviceInfo()
        info.device_id = torch.cuda.current_device()
        info.name = torch.cuda.get_device_name(info.device_id)

        props = torch.cuda.get_device_properties(info.device_id)
        info.sm_count = props.multi_processor_count
        info.max_compute_units = props.multi_processor_count
        info.global_mem_size_gb = props.total_memory / (1024**3)
        info.compute_capability = (props.major, props.minor)
        info.max_shared_memory_per_block_kb = props.max_shared_memory_per_block // 1024

        info.has_fp16 = True
        info.has_bf16 = props.major >= 8
        info.has_fp64 = True

        for attr in dir(props):
            if not attr.startswith("_"):
                try:
                    info.raw_properties[attr] = getattr(props, attr)
                except Exception:
                    pass
        info.raw_properties["query_method"] = "torch.cuda"

        info = _set_cuda_recommendations(info)
        return info

    except ImportError:
        logger.warning("PyTorch not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to query CUDA via torch: {e}")
        return None


def _set_cuda_recommendations(info: CUDADeviceInfo) -> CUDADeviceInfo:
    """Set recommended kernel parameters based on CUDA hardware."""
    major, _minor = info.compute_capability

    if major >= 9:
        # Hopper (H100, etc.)
        info.recommended_num_warps = 8
        info.recommended_tile_m = 128
        info.recommended_tile_n = 256
        info.recommended_tile_k = 64
    elif major >= 8:
        # Ampere (A100, etc.)
        info.recommended_num_warps = 8
        info.recommended_tile_m = 128
        info.recommended_tile_n = 128
        info.recommended_tile_k = 32
    elif major >= 7:
        # Volta/Turing (V100, T4, etc.)
        info.recommended_num_warps = 4
        info.recommended_tile_m = 128
        info.recommended_tile_n = 128
        info.recommended_tile_k = 32
    else:
        info.recommended_num_warps = 4
        info.recommended_tile_m = 64
        info.recommended_tile_n = 64
        info.recommended_tile_k = 32

    if info.global_mem_size_gb >= 40:
        info.recommended_tile_m = max(info.recommended_tile_m, 128)
        info.recommended_tile_n = max(info.recommended_tile_n, 128)

    return info


def query_device(device_type: str, device_id: int = 0) -> DeviceInfo:
    """Query device info, dispatching to the appropriate backend.

    Args:
        device_type: "xpu", "cuda", or "cpu"
        device_id: Device index

    Returns:
        DeviceInfo (or a subclass) with hardware details and recommendations.
    """
    if device_type == "cuda":
        result = query_cuda_via_torch()
        if result:
            return result
        logger.warning("CUDA query failed, using defaults")
        return CUDADeviceInfo(raw_properties={"query_method": "defaults"})

    if device_type == "xpu":
        from xe_forge.core.xpu_query import get_xpu_config

        xpu_info = get_xpu_config(device_id)
        return xpu_info

    # CPU fallback
    return DeviceInfo(
        name="CPU",
        device_type="cpu",
        raw_properties={"query_method": "defaults"},
    )


def get_device_config_for_pipeline(
    device_type: str = "xpu",
    input_shapes: list[tuple[int, ...]] | None = None,
    config: Any | None = None,
    dtype: str = "float16",
    device_id: int = 0,
) -> dict[str, Any]:
    """Get device configuration for the optimization pipeline.

    Dispatches to XPU or CUDA-specific logic based on device_type.
    """
    if device_type == "xpu":
        from xe_forge.core.xpu_query import get_xpu_config_for_pipeline

        return get_xpu_config_for_pipeline(
            input_shapes=input_shapes,
            config=config,
            dtype=dtype,
            device_id=device_id,
        )

    if device_type == "cuda":
        return _get_cuda_config_for_pipeline(
            input_shapes=input_shapes,
            config=config,
            dtype=dtype,
            device_id=device_id,
        )

    # CPU fallback
    return {
        "device": "cpu",
        "device_name": "CPU",
        "num_warps": 1,
        "num_stages": 1,
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
    }


def _get_cuda_config_for_pipeline(
    input_shapes: list[tuple[int, ...]] | None = None,
    config: Any | None = None,
    dtype: str = "float16",
    device_id: int = 0,
) -> dict[str, Any]:
    """Get CUDA configuration for the optimization pipeline."""
    from xe_forge.core.xpu_query import extract_mnk_from_shapes

    info = query_cuda_via_torch()
    if info is None:
        info = CUDADeviceInfo()

    result = {
        "device": "cuda",
        "device_name": info.name,
        "num_warps": info.recommended_num_warps,
        "num_stages": 3,
        "BLOCK_SIZE_M": info.recommended_tile_m,
        "BLOCK_SIZE_N": info.recommended_tile_n,
        "BLOCK_SIZE_K": info.recommended_tile_k,
        "max_compute_units": info.sm_count,
        "global_mem_gb": info.global_mem_size_gb,
        "has_fp16": info.has_fp16,
        "has_bf16": info.has_bf16,
        "has_fp64": info.has_fp64,
        "compute_capability": info.compute_capability,
    }

    if input_shapes and len(input_shapes) >= 1:
        M, N, K = extract_mnk_from_shapes(input_shapes)
        if M is not None and N is not None and K is not None:
            result["problem_shape"] = {"M": M, "N": N, "K": K}
            result["BLOCK_SIZE_M"] = min(result["BLOCK_SIZE_M"], M)
            result["BLOCK_SIZE_N"] = min(result["BLOCK_SIZE_N"], N)
            result["BLOCK_SIZE_K"] = min(result["BLOCK_SIZE_K"], K)

    if config and hasattr(config, "device_config"):
        dc = config.device_config
        result["num_warps"] = getattr(dc, "default_num_warps", result["num_warps"])
        result["num_stages"] = getattr(dc, "default_num_stages", result["num_stages"])

    return result


def format_device_config_for_llm(device_config: dict[str, Any], device_type: str = "xpu") -> str:
    """Format device config as a string for LLM prompts.

    Dispatches to device-specific formatting.
    """
    if device_type == "xpu":
        from xe_forge.core.xpu_query import format_xpu_config_for_llm

        return format_xpu_config_for_llm(device_config)

    return _format_cuda_config_for_llm(device_config)


def _format_cuda_config_for_llm(config: dict[str, Any]) -> str:
    """Format CUDA config as a string for LLM prompts."""
    lines = [
        "NVIDIA CUDA Configuration:",
        "=" * 40,
    ]

    if "problem_shape" in config:
        shape = config["problem_shape"]
        lines.append(f"Problem Shape: M={shape['M']}, N={shape['N']}, K={shape['K']}")
        lines.append("")

    lines.append("RECOMMENDED KERNEL PARAMETERS:")
    lines.append(f"  BLOCK_SIZE_M: {config.get('BLOCK_SIZE_M', 128)}")
    lines.append(f"  BLOCK_SIZE_N: {config.get('BLOCK_SIZE_N', 128)}")
    lines.append(f"  BLOCK_SIZE_K: {config.get('BLOCK_SIZE_K', 32)}")
    lines.append(f"  num_warps: {config.get('num_warps', 4)}")
    lines.append(f"  num_stages: {config.get('num_stages', 3)}")

    if config.get("device_name"):
        lines.append("")
        lines.append("HARDWARE:")
        lines.append(f"  Device: {config.get('device_name', 'Unknown')}")
        if config.get("compute_capability"):
            cc = config["compute_capability"]
            lines.append(f"  Compute Capability: {cc[0]}.{cc[1]}")
        if config.get("max_compute_units"):
            lines.append(f"  SM Count: {config['max_compute_units']}")
        if config.get("global_mem_gb"):
            lines.append(f"  Memory: {config['global_mem_gb']:.1f} GB")

    return "\n".join(lines)
