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

Exports resolve lazily (PEP 562), the same pattern `xe_forge.orbit` uses: importing a
light submodule such as `spec_loader` or `validator` must not drag in `executor`,
whose module-level `ai_bench` import is a heavyweight git dependency that CPU-only CI
deliberately does not install. `from xe_forge.core import X` behaves exactly as
before — the submodule loads on first attribute access.
"""

from __future__ import annotations

# Public name -> defining submodule. This is the same export surface the previous
# eager import block declared; only the load time changed.
_EXPORTS = {
    # device_query
    "CUDADeviceInfo": "device_query",
    "DeviceInfo": "device_query",
    "format_device_config_for_llm": "device_query",
    "get_device_config_for_pipeline": "device_query",
    "query_cuda_via_torch": "device_query",
    "query_device": "device_query",
    # executor (imports ai_bench)
    "ComparisonResult": "executor",
    "KernelBenchExecutor": "executor",
    "KernelExecutor": "executor",
    "create_executor_tool": "executor",
    # kernel_analyzer
    "AnalysisResult": "kernel_analyzer",
    "KernelAnalyzer": "kernel_analyzer",
    "format_analysis": "kernel_analyzer",
    # profiler
    "ProfileMetrics": "profiler",
    "ProfileResult": "profiler",
    "Recommendation": "profiler",
    "XPUProfiler": "profiler",
    # spec_loader
    "InputSpec": "spec_loader",
    "KernelSpec": "spec_loader",
    "VariantSpec": "spec_loader",
    "get_test_config_from_spec": "spec_loader",
    "load_spec": "spec_loader",
    "load_spec_from_string": "spec_loader",
    "parse_spec": "spec_loader",
    # sycl_executor (imports ai_bench)
    "KernelType": "sycl_executor",
    "SyclComparisonResult": "sycl_executor",
    "SyclExecutor": "sycl_executor",
    # trial_manager
    "TrialManager": "trial_manager",
    # validator
    "KernelValidator": "validator",
    "ValidationIssue": "validator",
    "format_issues": "validator",
    # xpu_query (backward-compatible XPU-specific exports)
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

__all__ = [*sorted(_EXPORTS), "create_executor_from_config"]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache so the lookup runs once
    return value


def __dir__() -> list[str]:
    return list(__all__)


def create_executor_from_config(config, kernel_type="gemm"):
    """
    Create an executor with settings from Config.

    Returns SyclExecutor when dsl=sycl, KernelBenchExecutor otherwise.
    `kernel_type` accepts a `KernelType` or its string value.
    """
    from xe_forge.models import DSL

    if config.device_config.dsl == DSL.SYCL:
        from xe_forge.core.sycl_executor import KernelType, SyclExecutor

        if isinstance(kernel_type, str):
            kernel_type = KernelType(kernel_type)
        return SyclExecutor(
            verify=config.optimization.require_correctness,
            kernel_type=kernel_type,
        )
    from xe_forge.core.executor import KernelBenchExecutor

    return KernelBenchExecutor(
        device=config.device_config.device,
        require_correctness=config.optimization.require_correctness,
        rtol=config.optimization.correctness_rtol,
        atol=config.optimization.correctness_atol,
    )
