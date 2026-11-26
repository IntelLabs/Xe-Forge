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

from __future__ import annotations

import importlib

# name -> the submodule that defines it, imported when the name is first read.
_EXPORTS = {
    "CUDADeviceInfo": "device_query",
    "DeviceInfo": "device_query",
    "format_device_config_for_llm": "device_query",
    "get_device_config_for_pipeline": "device_query",
    "query_cuda_via_torch": "device_query",
    "query_device": "device_query",
    "ComparisonResult": "executor",
    "KernelBenchExecutor": "executor",
    "KernelExecutor": "executor",
    "create_executor_tool": "executor",
    "AnalysisResult": "kernel_analyzer",
    "KernelAnalyzer": "kernel_analyzer",
    "format_analysis": "kernel_analyzer",
    "ProfileMetrics": "profiler",
    "ProfileResult": "profiler",
    "Recommendation": "profiler",
    "XPUProfiler": "profiler",
    "InputSpec": "spec_loader",
    "KernelSpec": "spec_loader",
    "VariantSpec": "spec_loader",
    "get_test_config_from_spec": "spec_loader",
    "load_spec": "spec_loader",
    "load_spec_from_string": "spec_loader",
    "parse_spec": "spec_loader",
    "KernelType": "sycl_executor",
    "SyclComparisonResult": "sycl_executor",
    "SyclExecutor": "sycl_executor",
    "TrialManager": "trial_manager",
    "KernelValidator": "validator",
    "ValidationIssue": "validator",
    "format_issues": "validator",
    "XPUDeviceInfo": "xpu_query",
    "extract_mnk_from_shapes": "xpu_query",
    "format_xpu_config_for_llm": "xpu_query",
    "get_autotune_configs": "xpu_query",
    "get_optimal_params": "xpu_query",
    "get_xpu_config": "xpu_query",
    "get_xpu_config_dict": "xpu_query",
    "get_xpu_config_for_pipeline": "xpu_query",
    "print_xpu_info": "xpu_query",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f"x_forge.core.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'x_forge.core' has no attribute {name!r}")


__all__ = [
    "AnalysisResult",
    "CUDADeviceInfo",
    "ComparisonResult",
    "DeviceInfo",
    "InputSpec",
    "KernelAnalyzer",
    "KernelBenchExecutor",
    "KernelExecutor",
    "KernelSpec",
    "KernelType",
    "KernelValidator",
    "ProfileMetrics",
    "ProfileResult",
    "Recommendation",
    "SyclComparisonResult",
    "SyclExecutor",
    "TrialManager",
    "ValidationIssue",
    "VariantSpec",
    "XPUDeviceInfo",
    "XPUProfiler",
    "create_executor_from_config",
    "create_executor_tool",
    "extract_mnk_from_shapes",
    "format_analysis",
    "format_device_config_for_llm",
    "format_issues",
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


def create_executor_from_config(
    config,
    kernel_type: KernelType | str | None = None,
) -> KernelBenchExecutor | SyclExecutor:
    """
    Create an executor with settings from Config.

    Returns SyclExecutor when dsl=sycl, KernelBenchExecutor otherwise; ``kernel_type``
    defaults to GEMM.
    """
    from x_forge.core.executor import KernelBenchExecutor
    from x_forge.core.sycl_executor import KernelType, SyclExecutor
    from x_forge.models import DSL

    kernel_type = KernelType.GEMM if kernel_type is None else kernel_type

    if config.device_config.dsl == DSL.SYCL:
        return SyclExecutor(
            verify=config.optimization.require_correctness,
            kernel_type=kernel_type,
            build_backend=getattr(config.engine, "build_backend", None),
            device=config.device_config.device,
        )
    return KernelBenchExecutor(
        device=config.device_config.device,
        require_correctness=config.optimization.require_correctness,
        rtol=config.optimization.correctness_rtol,
        atol=config.optimization.correctness_atol,
    )
