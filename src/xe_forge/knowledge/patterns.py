"""
Pattern matching helpers — maps issue types to optimization stages.

Resolution order:
1. Explicit dict  (O(1), authoritative for all known IssueType values)
2. Keyword-based stage inference  (handles LLM variants / future additions)
3. Prefix matching  (dtype_* → DTYPE_FIX, etc.)
4. Default: ANALYSIS  (logged as a warning so nothing is silently dropped)

To add a new IssueType:
  Option A (preferred): Add it to the explicit _MAPPING dict below.
  Option B (dynamic):   Ensure its name contains a keyword in _KEYWORD_RULES
                        and it will be routed automatically.

To add a new Stage with new keywords:
  Add entries to _KEYWORD_RULES — no other changes needed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from xe_forge.models import IssueType, OptimizationStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1: Explicit mapping  (canonical, covers all current IssueType values)
# ---------------------------------------------------------------------------

_MAPPING: dict[IssueType, OptimizationStage] = {
    # ALGORITHMIC
    IssueType.REDUNDANT_COMPUTATION: OptimizationStage.ALGORITHMIC,
    IssueType.SUBOPTIMAL_ALGORITHM: OptimizationStage.ALGORITHMIC,
    IssueType.ASSOCIATIVITY_REORDER: OptimizationStage.ALGORITHMIC,
    IssueType.COMMON_SUBEXPRESSION: OptimizationStage.ALGORITHMIC,
    IssueType.ALGEBRAIC_SIMPLIFICATION: OptimizationStage.ALGORITHMIC,
    IssueType.CACHEABLE_INTERMEDIATE: OptimizationStage.ALGORITHMIC,
    IssueType.LOOP_INVARIANT_CODE: OptimizationStage.ALGORITHMIC,
    IssueType.UNNECESSARY_MATERIALIZATION: OptimizationStage.ALGORITHMIC,
    IssueType.GEMM_SIMPLIFICATION: OptimizationStage.ALGORITHMIC,
    IssueType.REDUCTION_TREE_SUBOPTIMAL: OptimizationStage.ALGORITHMIC,
    # DTYPE
    IssueType.DTYPE_FLOAT64: OptimizationStage.DTYPE_FIX,
    IssueType.DTYPE_PRECISION: OptimizationStage.DTYPE_FIX,
    IssueType.DTYPE_INPUT_CONVERSION: OptimizationStage.DTYPE_FIX,
    # FUSION
    IssueType.UNFUSED_KERNELS: OptimizationStage.FUSION,
    IssueType.UNFUSED_ELEMENTWISE: OptimizationStage.FUSION,
    IssueType.UNFUSED_REDUCTION: OptimizationStage.FUSION,
    IssueType.FUSION_REGISTER_PRESSURE: OptimizationStage.FUSION,
    IssueType.FUSION_REPLACES_VENDOR: OptimizationStage.FUSION,
    IssueType.FUSION_NOOP: OptimizationStage.FUSION,
    # MEMORY ACCESS
    IssueType.MISSING_BOUNDARY_CHECK: OptimizationStage.MEMORY_ACCESS,
    IssueType.TRANSPOSE_IN_LOOP: OptimizationStage.MEMORY_ACCESS,
    IssueType.MISSING_TMA: OptimizationStage.MEMORY_ACCESS,
    IssueType.UNCOALESCED_ACCESS: OptimizationStage.MEMORY_ACCESS,
    IssueType.DEVICE_HOST_SYNC: OptimizationStage.MEMORY_ACCESS,
    IssueType.NON_CONTIGUOUS_INPUT: OptimizationStage.MEMORY_ACCESS,
    IssueType.CACHE_EVICTION_RISK: OptimizationStage.MEMORY_ACCESS,
    IssueType.LONG_LIVENESS: OptimizationStage.MEMORY_ACCESS,
    IssueType.HIGH_REGISTER_PRESSURE: OptimizationStage.MEMORY_ACCESS,
    # BLOCK POINTERS
    IssueType.MANUAL_POINTER_ARITHMETIC: OptimizationStage.BLOCK_POINTERS,
    IssueType.BLOCK_PTR_BOUNDARY_WRONG: OptimizationStage.BLOCK_POINTERS,
    IssueType.BLOCK_PTR_MULTIPLE_OF_MISUSE: OptimizationStage.BLOCK_POINTERS,
    IssueType.MISSING_BLOCK_POINTERS: OptimizationStage.BLOCK_POINTERS,
    # XPU SPECIFIC
    IssueType.SUBOPTIMAL_TILE_SIZE: OptimizationStage.XPU_SPECIFIC,
    IssueType.SUBOPTIMAL_WARPS: OptimizationStage.XPU_SPECIFIC,
    IssueType.MISSING_GRF_MODE: OptimizationStage.XPU_SPECIFIC,
    IssueType.NO_SWIZZLING: OptimizationStage.XPU_SPECIFIC,
    IssueType.REPACK_IN_FORWARD: OptimizationStage.XPU_SPECIFIC,
    IssueType.MISSING_PACKED_TRANSPOSE: OptimizationStage.XPU_SPECIFIC,
    IssueType.SERIALIZED_N_TILES: OptimizationStage.XPU_SPECIFIC,
    IssueType.SIGMOID_SLOW_EXP: OptimizationStage.XPU_SPECIFIC,
    IssueType.AUTOTUNE_DUPLICATE_PARAMS: OptimizationStage.AUTOTUNING,
    # PERSISTENT KERNEL
    IssueType.MISSING_PERSISTENT: OptimizationStage.PERSISTENT_KERNEL,
    IssueType.PERSISTENT_NUM_PROGS_HARDCODED: OptimizationStage.PERSISTENT_KERNEL,
    # AUTOTUNING
    IssueType.MISSING_AUTOTUNE: OptimizationStage.AUTOTUNING,
    IssueType.SUBOPTIMAL_AUTOTUNE_CONFIGS: OptimizationStage.AUTOTUNING,
    IssueType.AUTOTUNE_KEY_MISSING: OptimizationStage.AUTOTUNING,
    # DISCOVERY
    IssueType.OPEN_ENDED: OptimizationStage.DISCOVERY,
}

# ---------------------------------------------------------------------------
# Layer 2: Keyword-based inference
# Ordered: more specific keywords first, more general ones last.
# Each entry: (keyword_substring, OptimizationStage)
# The first match wins.
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[str, OptimizationStage]] = [
    # ALGORITHMIC
    ("redundant", OptimizationStage.ALGORITHMIC),
    ("suboptimal_algo", OptimizationStage.ALGORITHMIC),
    ("associativ", OptimizationStage.ALGORITHMIC),
    ("common_subexpr", OptimizationStage.ALGORITHMIC),
    ("algebraic", OptimizationStage.ALGORITHMIC),
    ("cacheable", OptimizationStage.ALGORITHMIC),
    ("loop_invariant", OptimizationStage.ALGORITHMIC),
    ("materialization", OptimizationStage.ALGORITHMIC),
    ("gemm_simplif", OptimizationStage.ALGORITHMIC),
    ("reduction_tree", OptimizationStage.ALGORITHMIC),
    # DTYPE  (check before generic "precision" which could mean other things)
    ("float64", OptimizationStage.DTYPE_FIX),
    ("dtype", OptimizationStage.DTYPE_FIX),
    ("precision", OptimizationStage.DTYPE_FIX),
    ("input_conversion", OptimizationStage.DTYPE_FIX),
    # FUSION
    ("unfused", OptimizationStage.FUSION),
    ("fusion", OptimizationStage.FUSION),
    ("fuse", OptimizationStage.FUSION),
    # AUTOTUNING  (before "autotune_duplicate" which is also XPU; check specific first)
    ("open_ended", OptimizationStage.DISCOVERY),
    ("discovery", OptimizationStage.DISCOVERY),
    ("autotune_key", OptimizationStage.AUTOTUNING),
    ("missing_autotune", OptimizationStage.AUTOTUNING),
    ("suboptimal_autotune", OptimizationStage.AUTOTUNING),
    ("autotune_config", OptimizationStage.AUTOTUNING),
    # BLOCK POINTERS
    ("block_ptr", OptimizationStage.BLOCK_POINTERS),
    ("block_pointer", OptimizationStage.BLOCK_POINTERS),
    ("manual_pointer", OptimizationStage.BLOCK_POINTERS),
    ("pointer_arithmetic", OptimizationStage.BLOCK_POINTERS),
    ("tensor_descriptor", OptimizationStage.BLOCK_POINTERS),
    ("tma", OptimizationStage.MEMORY_ACCESS),
    # MEMORY ACCESS
    ("boundary_check", OptimizationStage.MEMORY_ACCESS),
    ("transpose_in_loop", OptimizationStage.MEMORY_ACCESS),
    ("uncoalesced", OptimizationStage.MEMORY_ACCESS),
    ("coalesce", OptimizationStage.MEMORY_ACCESS),
    ("device_host", OptimizationStage.MEMORY_ACCESS),
    ("host_sync", OptimizationStage.MEMORY_ACCESS),
    ("contiguous", OptimizationStage.MEMORY_ACCESS),
    ("cache_eviction", OptimizationStage.MEMORY_ACCESS),
    ("liveness", OptimizationStage.MEMORY_ACCESS),
    ("register_pressure", OptimizationStage.MEMORY_ACCESS),
    ("memory_access", OptimizationStage.MEMORY_ACCESS),
    ("memory_layout", OptimizationStage.MEMORY_ACCESS),
    ("memory_coalesce", OptimizationStage.MEMORY_ACCESS),
    ("poor_memory", OptimizationStage.MEMORY_ACCESS),
    ("bandwidth", OptimizationStage.MEMORY_ACCESS),
    # PERSISTENT KERNEL
    ("persistent", OptimizationStage.PERSISTENT_KERNEL),
    ("stream_k", OptimizationStage.PERSISTENT_KERNEL),
    ("num_progs", OptimizationStage.PERSISTENT_KERNEL),
    # XPU SPECIFIC  (after more specific rules above)
    ("tile_size", OptimizationStage.XPU_SPECIFIC),
    ("suboptimal_tile", OptimizationStage.XPU_SPECIFIC),
    ("suboptimal_warp", OptimizationStage.XPU_SPECIFIC),
    ("num_warps", OptimizationStage.XPU_SPECIFIC),
    ("grf_mode", OptimizationStage.XPU_SPECIFIC),
    ("grf", OptimizationStage.XPU_SPECIFIC),
    ("swizzl", OptimizationStage.XPU_SPECIFIC),
    ("repack", OptimizationStage.XPU_SPECIFIC),
    ("packed_transpose", OptimizationStage.XPU_SPECIFIC),
    ("serialized_n", OptimizationStage.XPU_SPECIFIC),
    ("sigmoid", OptimizationStage.XPU_SPECIFIC),
    ("exp2", OptimizationStage.XPU_SPECIFIC),
    ("autotune_duplicate", OptimizationStage.AUTOTUNING),
    ("autotune", OptimizationStage.AUTOTUNING),
    ("warp", OptimizationStage.XPU_SPECIFIC),
    ("xpu", OptimizationStage.XPU_SPECIFIC),
    ("intel", OptimizationStage.XPU_SPECIFIC),
]

# ---------------------------------------------------------------------------
# Layer 3: Prefix rules  (last resort before ANALYSIS fallback)
# ---------------------------------------------------------------------------

_PREFIX_RULES: list[tuple[str, OptimizationStage]] = [
    ("dtype_", OptimizationStage.DTYPE_FIX),
    ("missing_block", OptimizationStage.BLOCK_POINTERS),
    ("block_ptr_", OptimizationStage.BLOCK_POINTERS),
    ("missing_", OptimizationStage.XPU_SPECIFIC),  # missing_grf, missing_warp, etc.
    ("suboptimal_", OptimizationStage.XPU_SPECIFIC),
    ("unfused_", OptimizationStage.FUSION),
    ("fusion_", OptimizationStage.FUSION),
    ("persistent_", OptimizationStage.PERSISTENT_KERNEL),
    ("open_ended", OptimizationStage.DISCOVERY),
]

# ---------------------------------------------------------------------------
# Dynamic registration — allows new issue→stage mappings without editing _MAPPING
# ---------------------------------------------------------------------------

_dynamic_registry: dict[str, OptimizationStage] = {}


def register_stage(issue_type_value: str, stage: OptimizationStage) -> None:
    """
    Register a custom issue_type string → stage mapping at runtime.

    Useful for plugins or experiments that introduce new IssueType values
    without modifying the core enum. Registered mappings take priority over
    keyword and prefix inference, but yield to the explicit _MAPPING dict.

    Example:
        register_stage("my_custom_issue", OptimizationStage.XPU_SPECIFIC)
    """
    _dynamic_registry[issue_type_value.lower()] = stage
    # Clear lru_cache so the new mapping is picked up immediately
    _infer_from_string.cache_clear()
    logger.debug("Registered dynamic stage mapping: %s → %s", issue_type_value, stage.value)


# ---------------------------------------------------------------------------
# Core inference — keyword + prefix fallback
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _infer_from_string(type_str: str) -> OptimizationStage:
    """
    Infer a stage from a raw issue type string using keyword and prefix rules.
    Results are cached so repeated calls for the same string are O(1).
    """
    s = type_str.lower()

    # Dynamic registry first
    if s in _dynamic_registry:
        return _dynamic_registry[s]

    # Keyword scan
    for keyword, stage in _KEYWORD_RULES:
        if keyword in s:
            return stage

    # Prefix scan
    for prefix, stage in _PREFIX_RULES:
        if s.startswith(prefix):
            return stage

    return OptimizationStage.ANALYSIS  # will be logged by caller


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_stage_for_issue(issue_type: IssueType) -> OptimizationStage:
    """
    Map an IssueType to an OptimizationStage.

    Resolution order:
    1. Explicit _MAPPING dict       (all current enum values)
    2. Dynamic registry             (runtime additions via register_stage())
    3. Keyword inference            (handles LLM variants and future types)
    4. Prefix inference             (dtype_*, missing_*, etc.)
    5. ANALYSIS fallback            (logged as WARNING so nothing is silently lost)
    """
    # Layer 1: explicit
    stage = _MAPPING.get(issue_type)
    if stage is not None:
        return stage

    # Layers 2-4: string-based inference
    type_str = issue_type.value if hasattr(issue_type, "value") else str(issue_type)
    stage = _infer_from_string(type_str)

    if stage == OptimizationStage.ANALYSIS:
        logger.warning(
            "Issue type %r has no explicit stage mapping and could not be inferred — "
            "it will be skipped by the pipeline. "
            "Add it to _MAPPING in xe_forge/knowledge/patterns.py or call register_stage().",
            type_str,
        )
    else:
        logger.debug("Issue type %r mapped via keyword inference → %s", type_str, stage.value)

    return stage


def get_stage_for_issue_str(issue_type_str: str) -> OptimizationStage:
    """
    Variant that accepts a raw string — useful when the LLM returns a value
    that did not parse into a valid IssueType enum member.

    Tries to coerce to IssueType first; falls back to string inference.
    """
    try:
        return get_stage_for_issue(IssueType(issue_type_str))
    except (ValueError, KeyError):
        stage = _infer_from_string(issue_type_str)
        if stage == OptimizationStage.ANALYSIS:
            logger.warning(
                "Unknown issue type string %r — not in IssueType enum and no keyword match. "
                "Returning ANALYSIS (will be skipped). "
                "Consider adding to IssueType enum and _MAPPING.",
                issue_type_str,
            )
        else:
            logger.info(
                "Unknown issue type string %r inferred as %s via keyword matching.",
                issue_type_str,
                stage.value,
            )
        return stage


def all_mapped_stages() -> dict[str, str]:
    """Return all explicit issue→stage mappings as {issue_value: stage_value} dict.
    Useful for debugging and documentation."""
    return {k.value: v.value for k, v in _MAPPING.items()}
