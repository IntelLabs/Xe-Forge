"""
Core data models for Triton optimization pipeline
"""

from datetime import datetime
from enum import Enum, StrEnum

from pydantic import BaseModel, Field


class OptimizationStage(str, Enum):
    ANALYSIS = "analysis"
    ALGORITHMIC = "algorithmic"
    DTYPE_FIX = "dtype_fix"
    FUSION = "fusion"
    MEMORY_ACCESS = "memory_access"
    BLOCK_POINTERS = "block_pointers"
    PERSISTENT_KERNEL = "persistent_kernel"
    XPU_SPECIFIC = "xpu_specific"
    AUTOTUNING = "autotuning"


class IssueType(StrEnum):
    # ALGORITHMIC
    REDUNDANT_COMPUTATION = "redundant_computation"
    SUBOPTIMAL_ALGORITHM = "suboptimal_algorithm"
    ASSOCIATIVITY_REORDER = "associativity_reorder"
    COMMON_SUBEXPRESSION = "common_subexpression"
    ALGEBRAIC_SIMPLIFICATION = "algebraic_simplification"
    CACHEABLE_INTERMEDIATE = "cacheable_intermediate"
    LOOP_INVARIANT_CODE = "loop_invariant_code"
    UNNECESSARY_MATERIALIZATION = "unnecessary_materialization"
    GEMM_SIMPLIFICATION = "gemm_simplification"
    REDUCTION_TREE_SUBOPTIMAL = "reduction_tree_suboptimal"
    # DTYPE
    DTYPE_FLOAT64 = "dtype_float64"
    DTYPE_PRECISION = "dtype_precision"
    DTYPE_INPUT_CONVERSION = "dtype_input_conversion"
    # FUSION
    UNFUSED_KERNELS = "unfused_kernels"
    UNFUSED_ELEMENTWISE = "unfused_elementwise"
    UNFUSED_REDUCTION = "unfused_reduction"
    FUSION_REGISTER_PRESSURE = "fusion_register_pressure"
    FUSION_REPLACES_VENDOR = "fusion_replaces_vendor"
    FUSION_NOOP = "fusion_noop"
    # MEMORY ACCESS
    MANUAL_POINTER_ARITHMETIC = "manual_pointer_arithmetic"
    MISSING_BOUNDARY_CHECK = "missing_boundary_check"
    TRANSPOSE_IN_LOOP = "transpose_in_loop"
    MISSING_TMA = "missing_tma"
    UNCOALESCED_ACCESS = "uncoalesced_access"
    DEVICE_HOST_SYNC = "device_host_sync"
    NON_CONTIGUOUS_INPUT = "non_contiguous_input"
    CACHE_EVICTION_RISK = "cache_eviction_risk"
    LONG_LIVENESS = "long_liveness"
    HIGH_REGISTER_PRESSURE = "high_register_pressure"
    # BLOCK POINTERS
    BLOCK_PTR_BOUNDARY_WRONG = "block_ptr_boundary_wrong"
    BLOCK_PTR_MULTIPLE_OF_MISUSE = "block_ptr_multiple_of_misuse"
    MISSING_BLOCK_POINTERS = "missing_block_pointers"
    # XPU SPECIFIC
    SUBOPTIMAL_TILE_SIZE = "suboptimal_tile_size"
    SUBOPTIMAL_WARPS = "suboptimal_warps"
    MISSING_GRF_MODE = "missing_grf_mode"
    NO_SWIZZLING = "no_swizzling"
    REPACK_IN_FORWARD = "repack_in_forward"
    MISSING_PACKED_TRANSPOSE = "missing_packed_transpose"
    SERIALIZED_N_TILES = "serialized_n_tiles"
    AUTOTUNE_DUPLICATE_PARAMS = "autotune_duplicate_params"
    SIGMOID_SLOW_EXP = "sigmoid_slow_exp"
    # PERSISTENT KERNEL
    MISSING_PERSISTENT = "missing_persistent"
    PERSISTENT_NUM_PROGS_HARDCODED = "persistent_num_progs_hardcoded"
    # AUTOTUNING
    MISSING_AUTOTUNE = "missing_autotune"
    SUBOPTIMAL_AUTOTUNE_CONFIGS = "suboptimal_autotune_configs"
    AUTOTUNE_KEY_MISSING = "autotune_key_missing"


class DetectedIssue(BaseModel):
    issue_type: IssueType
    severity: int = Field(ge=1, le=5, description="1=minor, 5=critical")
    location: str | None = None
    description: str
    suggested_fix: str
    estimated_speedup: str | None = None


class KernelAnalysis(BaseModel):
    kernel_name: str
    detected_issues: list[DetectedIssue] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    memory_accesses: list[str] = Field(default_factory=list)
    has_fusion_opportunity: bool = False
    has_algorithmic_opportunity: bool = False
    current_dtype: str = "unknown"
    tile_sizes: dict[str, int] = Field(default_factory=dict)
    num_warps: int | None = None
    num_stages: int | None = None
    uses_block_pointers: bool = False
    uses_tma: bool = False
    is_persistent: bool = False


class StageResult(BaseModel):
    stage: OptimizationStage
    success: bool
    input_code: str
    output_code: str | None = None
    changes_made: list[str] = Field(default_factory=list)
    reasoning: str | None = None
    error_message: str | None = None
    metrics_before: dict[str, float] | None = None
    metrics_after: dict[str, float] | None = None
    speedup: float | None = None


class OptimizationResult(BaseModel):
    kernel_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    original_code: str
    optimized_code: str | None = None
    stages_applied: list[StageResult] = Field(default_factory=list)
    total_speedup: float | None = None
    analysis: KernelAnalysis | None = None
    success: bool = False
    error_message: str | None = None
    original_ms: float | None = None
    optimized_ms: float | None = None
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_memory_bw: float | None = None
    optimized_memory_bw: float | None = None


class KnowledgeEntry(BaseModel):
    id: str
    name: str
    stage: OptimizationStage
    pattern_before: str
    pattern_after: str
    description: str
    rationale: str
    expected_speedup: str | None = None
    applies_to: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    examples: list[dict[str, str]] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    success: bool
    execution_time_ms: float | None = None
    tflops: float | None = None
    memory_bandwidth_gb: float | None = None
    output_correct: bool | None = None
    error_message: str | None = None
    error_traceback: str | None = None
