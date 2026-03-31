"""
Core components for kernel execution and validation

Provides KernelBench-style testing with accurate XPU timing including:
- L2 cache flushing between runs
- Hardware event-based timing
- Proper warmup and synchronization
- Comparison tools for CoVeR agent feedback
- YAML spec loading for test configurations
- XPU hardware query for optimal kernel parameters
- Configurable correctness validation (via REQUIRE_CORRECTNESS, CORRECTNESS_RTOL, CORRECTNESS_ATOL)
"""

from xpu_forge.core.executor import (
    ComparisonResult,
    KernelBenchExecutor,
    KernelExecutor,
    create_executor_tool,
)
from xpu_forge.core.spec_loader import (
    InputSpec,
    KernelSpec,
    VariantSpec,
    get_test_config_from_spec,
    load_spec,
    load_spec_from_string,
    parse_spec,
)

# validator module not included — correctness checking is handled by executor
# from xpu_forge.core.validator import KernelValidator
from xpu_forge.core.xpu_query import (
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
    # Executor
    "ComparisonResult",
    # Spec loader
    "InputSpec",
    "KernelBenchExecutor",
    "KernelExecutor",
    "KernelSpec",
    "VariantSpec",
    # XPU query & config
    "XPUDeviceInfo",
    "create_executor_from_config",
    "create_executor_tool",
    "extract_mnk_from_shapes",
    "format_xpu_config_for_llm",
    "get_autotune_configs",
    "get_optimal_params",
    "get_test_config_from_spec",
    "get_xpu_config",
    "get_xpu_config_dict",
    "get_xpu_config_for_pipeline",
    "load_spec",
    "load_spec_from_string",
    "parse_spec",
    "print_xpu_info",
]


def create_executor_from_config(config) -> KernelBenchExecutor:
    """
    Create a KernelBenchExecutor with settings from Config.

    Args:
        config: Config object (from xpu_forge.config)

    Returns:
        KernelBenchExecutor configured with correctness settings

    Example:
        from xpu_forge.config import get_config
        from xpu_forge.core import create_executor_from_config

        config = get_config()
        executor = create_executor_from_config(config)
    """
    return KernelBenchExecutor(
        device=config.xpu.device,
        require_correctness=config.optimization.require_correctness,
        rtol=config.optimization.correctness_rtol,
        atol=config.optimization.correctness_atol,
    )
