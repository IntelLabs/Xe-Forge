"""
Analyzer Agent - LLM-based analysis of Triton kernels for optimization opportunities.
"""

import logging

import dspy

from xe_forge.models import DetectedIssue, KernelAnalysis

logger = logging.getLogger(__name__)


class AnalysisSignature(dspy.Signature):
    """Analyze Triton kernel for optimization opportunities.

    You are a world-class expert in Triton GPU/XPU kernel optimization,
    numerical linear algebra, and high-performance computing.

    Analyze the given Triton kernel code and, if available, the original PyTorch
    implementation for higher-level algorithmic context.

    You must identify ALL applicable optimizations across every category below.
    Use your deep knowledge of GPU programming, Triton internals, Intel XPU
    architecture, and mathematical optimization.

    === ISSUE CATEGORIES (issue_type values) ===

    ALGORITHMIC / MATHEMATICAL (run BEFORE low-level optimizations):
      - redundant_computation: repeated work that can be factored out
      - suboptimal_algorithm: naive algorithm when a better one exists
      - associativity_reorder: reorder associative ops to reduce FLOPs
      - common_subexpression: same sub-expression computed multiple times
      - algebraic_simplification: identity ops, distributive law simplifications
      - cacheable_intermediate: reusable intermediate that is recomputed
      - loop_invariant_code: computation inside loop that can be hoisted
      - unnecessary_materialization: writes intermediate to global mem needlessly
      - gemm_simplification: GEMM has exploitable structure (symmetric, triangular, etc.)
      - reduction_tree_suboptimal: naive serial reduction vs tree reduction

    DTYPE:
      - dtype_float64, dtype_precision, dtype_input_conversion

    FUSION:
      - unfused_kernels, unfused_elementwise, unfused_reduction
      - fusion_register_pressure, fusion_replaces_vendor, fusion_noop

    MEMORY ACCESS:
      - missing_boundary_check, transpose_in_loop, missing_tma
      - uncoalesced_access, device_host_sync, non_contiguous_input
      - cache_eviction_risk, long_liveness, high_register_pressure

    BLOCK POINTERS:
      - manual_pointer_arithmetic, block_ptr_boundary_wrong
      - block_ptr_multiple_of_misuse, missing_block_pointers

    XPU SPECIFIC (Intel):
      - suboptimal_tile_size, suboptimal_warps, missing_grf_mode
      - no_swizzling, repack_in_forward, missing_packed_transpose
      - serialized_n_tiles, sigmoid_slow_exp

    PERSISTENT KERNEL:
      - missing_persistent, persistent_num_progs_hardcoded

    AUTOTUNING:
      - missing_autotune: kernel uses hardcoded params instead of @triton.autotune
      - suboptimal_autotune_configs: autotune configs exist but are incomplete
          or missing important param combinations for the target hardware
      - autotune_key_missing: @triton.autotune is missing key= argument so
          configs are not re-evaluated when problem shape changes
      - autotune_duplicate_params: meta-parameter redefined in autotune config

    IMPORTANT:
    - Return issues as a JSON array of DetectedIssue objects.
    - Each issue MUST have: issue_type, severity (1-5), description,
      suggested_fix, estimated_speedup.
    - For fused kernels, pay special attention to ALGORITHMIC issues.
    - Return empty array [] ONLY if the kernel is already optimal.
    """

    pytorch_code: dspy.Code["python"] = dspy.InputField(
        desc="Original PyTorch implementation. May be empty if not available."
    )
    triton_code: dspy.Code["python"] = dspy.InputField(desc="Triton kernel source code to analyze.")
    problem_context: str = dspy.InputField(
        desc="Problem size, FLOP count, target device, and other context."
    )

    issues_found: list[DetectedIssue] = dspy.OutputField(
        desc="List of detected issues. Return empty array [] only if kernel is already optimal."
    )


class AnalyzerAgent:
    """LLM-based analyzer for Triton kernels. No local knowledge base required."""

    def __init__(self, knowledge_base=None):
        # knowledge_base accepted for backward compat but not used
        self.predictor = dspy.Predict(AnalysisSignature)

    def analyze(
        self,
        triton_code: str,
        pytorch_code: str = "",
        kernel_name: str = "kernel",
        input_shapes: list[tuple] | None = None,
        flop: float | None = None,
        target_dtype: str | None = None,
    ) -> KernelAnalysis:
        problem_context = self._build_problem_context(input_shapes, flop, target_dtype)

        issues: list[DetectedIssue] = []
        try:
            result = self.predictor(
                triton_code=triton_code,
                pytorch_code=pytorch_code or "",
                problem_context=problem_context,
            )
            logger.debug(f"LLM issues_found raw: {result.issues_found}")
            issues = result.issues_found
            logger.info(f"Parsed {len(issues)} issues from LLM")
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")

        from xe_forge.knowledge.patterns import get_stage_for_issue
        from xe_forge.models import OptimizationStage

        has_fusion = any(
            get_stage_for_issue(i.issue_type) == OptimizationStage.FUSION for i in issues
        )
        has_algo = any(
            get_stage_for_issue(i.issue_type) == OptimizationStage.ALGORITHMIC for i in issues
        )

        return KernelAnalysis(
            kernel_name=kernel_name,
            detected_issues=issues,
            has_fusion_opportunity=has_fusion,
            has_algorithmic_opportunity=has_algo,
        )

    def _build_problem_context(self, input_shapes, flop, target_dtype=None):
        lines = ["TARGET DEVICE: Intel XPU (Data Center GPU Max / Ponte Vecchio)", ""]
        if target_dtype:
            lines.append(f"TARGET DTYPE: {target_dtype}")
            lines.append(
                f"Kernel should use {target_dtype} for inputs/outputs and accumulate in float32"
            )
            lines.append("")
        if input_shapes:
            lines.append(f"INPUT SHAPES: {input_shapes}")
            total = sum((s[0] * s[1] if len(s) >= 2 else s[0]) for s in input_shapes if s)
            lines.append(f"Total elements: {total:,}")
            if total > 1_000_000:
                lines.append("Problem size: LARGE (>1M elements)")
            elif total > 10_000:
                lines.append("Problem size: MEDIUM (10K-1M elements)")
            else:
                lines.append("Problem size: SMALL (<10K elements)")
        if flop:
            lines.append(f"FLOP COUNT: {flop:,.0f}")
            if flop > 1e12:
                lines.append("Compute intensity: VERY HIGH (>1 TFLOP)")
            elif flop > 1e9:
                lines.append("Compute intensity: HIGH (>1 GFLOP)")
            else:
                lines.append("Compute intensity: LOW (<1 GFLOP)")
        return "\n".join(lines) if lines else "No problem context provided"
