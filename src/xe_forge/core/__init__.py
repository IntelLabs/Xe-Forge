"""
Core components for kernel execution and validation

Provides KernelBench-style testing with accurate GPU timing including:
- L2 cache flushing between runs
- Hardware event-based timing
- Proper warmup and synchronization
- Comparison tools for CoVeR agent feedback
- YAML spec loading for test configurations
- Device hardware query for optimal kernel parameters (XPU, CUDA)
- Configurable correctness validation (via REQUIRE_CORRECTNESS, CORRECTNESS_RTOL, CORRECTNESS_ATOL)
"""

from xe_forge.core.device_query import (
    CUDADeviceInfo,
    DeviceInfo,
    format_device_config_for_llm,
    get_device_config_for_pipeline,
    query_cuda_via_torch,
    query_device,
)
from xe_forge.core.executor import (
    ComparisonResult,
    KernelBenchExecutor,
    KernelExecutor,
    create_executor_tool,
)
from xe_forge.core.spec_loader import (
    InputSpec,
    KernelSpec,
    VariantSpec,
    get_test_config_from_spec,
    load_spec,
    load_spec_from_string,
    parse_spec,
)

# Backward-compatible XPU-specific exports
from xe_forge.core.xpu_query import (
    XPUDeviceInfo,
    extract_mnk_from_shapes,
    format_xpu_config_for_llm,
    get_autotune_configs,
    get_optimal_params,
    get_xpu_config,
    get_xpu_config_dict,
    get_xpu_config_for_pipeline,
    print_xpu_info,
)

__all__ = [
    "CUDADeviceInfo",
    "ComparisonResult",
    "DeviceInfo",
    "InputSpec",
    "KernelBenchExecutor",
    "KernelExecutor",
    "KernelSpec",
    "VariantSpec",
    "XPUDeviceInfo",
    "create_executor_from_config",
    "create_executor_tool",
    "extract_mnk_from_shapes",
    "format_device_config_for_llm",
    "format_xpu_config_for_llm",
    "get_autotune_configs",
    "get_device_config_for_pipeline",
    "get_optimal_params",
    "get_test_config_from_spec",
    "get_xpu_config",
    "get_xpu_config_dict",
    "get_xpu_config_for_pipeline",
    "load_spec",
    "load_spec_from_string",
    "parse_spec",
    "print_xpu_info",
    "query_cuda_via_torch",
    "query_device",
]


def create_executor_from_config(config) -> KernelBenchExecutor:
    """
    Create a KernelBenchExecutor with settings from Config.

    Args:
        config: Config object (from xe_forge.config)

    Returns:
        KernelBenchExecutor configured with correctness settings

    Example:
        from xe_forge.config import get_config
        from xe_forge.core import create_executor_from_config

        config = get_config()
        executor = create_executor_from_config(config)
    """
    return KernelBenchExecutor(
        device=config.device_config.device,
        require_correctness=config.optimization.require_correctness,
        rtol=config.optimization.correctness_rtol,
        atol=config.optimization.correctness_atol,
    )
