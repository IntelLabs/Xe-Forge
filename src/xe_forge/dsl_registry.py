"""DSL-stage compatibility registry.

Maps each DSL to the set of optimization stages it supports.
"""

from xe_forge.models import DSL, OptimizationStage

DSL_SUPPORTED_STAGES: dict[DSL, set[OptimizationStage]] = {
    DSL.TRITON: {
        OptimizationStage.ANALYSIS,
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.BLOCK_POINTERS,
        OptimizationStage.PERSISTENT_KERNEL,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.AUTOTUNING,
        OptimizationStage.DISCOVERY,
    },
    DSL.GLUON: {
        OptimizationStage.ANALYSIS,
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.AUTOTUNING,
        OptimizationStage.DISCOVERY,
    },
    DSL.SYCL: {
        OptimizationStage.ANALYSIS,
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.DISCOVERY,
    },
    DSL.CUDA: {
        OptimizationStage.ANALYSIS,
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.PERSISTENT_KERNEL,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.AUTOTUNING,
        OptimizationStage.DISCOVERY,
    },
}


def get_stages_for_dsl(dsl: DSL | str) -> list[OptimizationStage]:
    """Return the ordered list of supported stages for a given DSL."""
    dsl = DSL(dsl) if isinstance(dsl, str) else dsl
    supported = DSL_SUPPORTED_STAGES.get(dsl, DSL_SUPPORTED_STAGES[DSL.TRITON])
    stage_order = list(OptimizationStage)
    return [s for s in stage_order if s in supported]
