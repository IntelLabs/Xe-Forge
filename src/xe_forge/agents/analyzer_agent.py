"""
Analyzer Agent - LLM-based analysis of Triton kernels for optimization opportunities.
"""

import logging

import dspy

from xe_forge.knowledge.patterns import get_stage_for_issue, get_stage_for_issue_str
from xe_forge.models import DSL, DetectedIssue, IssueType, KernelAnalysis, OptimizationStage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the issue category section of the prompt dynamically from the enum
# so it never drifts from the actual IssueType values.
# ---------------------------------------------------------------------------


_SYCL_DESCRIPTIONS: dict[IssueType, str] = {
    IssueType.SUBOPTIMAL_TILE_SIZE: "TileShape dimensions suboptimal for Intel BMG — try Shape<_256,_256,_32> or Shape<_128,_128,_64>",
    IssueType.SUBOPTIMAL_WARPS: "SubgroupSize in MMA Atom suboptimal — try XE_DPAS_TT<8, float, bfloat16_t>",
    IssueType.HIGH_REGISTER_PRESSURE: "Too many PipelineStages or large TileShape causing register spill",
    IssueType.CACHE_EVICTION_RISK: "PipelineStages too high or TileShape too large for L1/L2",
    IssueType.UNCOALESCED_ACCESS: "Memory layout mismatch (RowMajor vs ColumnMajor) causing non-coalesced access",
    IssueType.DTYPE_PRECISION: "Using float accumulators where bfloat16 suffices, or float64 anywhere",
    IssueType.DTYPE_FLOAT64: "float64 in computation — extremely slow on Intel XPU",
    IssueType.UNFUSED_ELEMENTWISE: "Elementwise op not fused into CUTLASS epilogue callbacks",
    IssueType.UNFUSED_KERNELS: "Multiple kernel launches that could be fused",
    IssueType.SUBOPTIMAL_ALGORITHM: "Naive algorithm when a more efficient form exists",
    IssueType.REDUNDANT_COMPUTATION: "Repeated work that can be factored out",
    IssueType.OPEN_ENDED: (
        "A novel optimization not covered by any existing issue_type. "
        "Use ONLY when you have found a concrete, high-value, implementable "
        "optimization with no matching type above. "
        "You MUST populate open_ended_proposal with: "
        "(a) exactly what changes, "
        "(b) why it is valid, "
        "(c) a before/after C++ code sketch, "
        "(d) estimated speedup with reasoning."
    ),
}

_SYCL_SKIP_ISSUES: set[IssueType] = {
    IssueType.MANUAL_POINTER_ARITHMETIC,
    IssueType.BLOCK_PTR_BOUNDARY_WRONG,
    IssueType.BLOCK_PTR_MULTIPLE_OF_MISUSE,
    IssueType.MISSING_BLOCK_POINTERS,
    IssueType.MISSING_AUTOTUNE,
    IssueType.SUBOPTIMAL_AUTOTUNE_CONFIGS,
    IssueType.AUTOTUNE_KEY_MISSING,
    IssueType.AUTOTUNE_DUPLICATE_PARAMS,
    IssueType.MISSING_GRF_MODE,
    IssueType.NO_SWIZZLING,
    IssueType.SIGMOID_SLOW_EXP,
    IssueType.REPACK_IN_FORWARD,
    IssueType.MISSING_PACKED_TRANSPOSE,
    IssueType.MISSING_PERSISTENT,
    IssueType.PERSISTENT_NUM_PROGS_HARDCODED,
    IssueType.SERIALIZED_N_TILES,
    IssueType.MISSING_TMA,
    IssueType.MISSING_BOUNDARY_CHECK,
    IssueType.DEVICE_HOST_SYNC,
    IssueType.NON_CONTIGUOUS_INPUT,
    IssueType.TRANSPOSE_IN_LOOP,
}


def _build_issue_categories(dsl: DSL = DSL.TRITON) -> str:
    """
    Generate the === ISSUE CATEGORIES === block from the IssueType enum + stage mapping.
    Groups by stage in pipeline order so the LLM sees a clean, current list.
    """

    stage_order = [
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.DTYPE_FIX,
        OptimizationStage.FUSION,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.BLOCK_POINTERS,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.PERSISTENT_KERNEL,
        OptimizationStage.AUTOTUNING,
    ]

    # Human-readable stage labels
    stage_labels = {
        OptimizationStage.ALGORITHMIC: "ALGORITHMIC / MATHEMATICAL (run BEFORE low-level optimizations)",
        OptimizationStage.DTYPE_FIX: "DTYPE",
        OptimizationStage.FUSION: "FUSION",
        OptimizationStage.MEMORY_ACCESS: "MEMORY ACCESS",
        OptimizationStage.BLOCK_POINTERS: "BLOCK POINTERS",
        OptimizationStage.DEVICE_SPECIFIC: "DEVICE SPECIFIC",
        OptimizationStage.PERSISTENT_KERNEL: "PERSISTENT KERNEL",
        OptimizationStage.AUTOTUNING: "AUTOTUNING",
    }

    # Descriptions for each issue type (LLM guidance)
    _descriptions: dict[IssueType, str] = {
        # ALGORITHMIC
        IssueType.REDUNDANT_COMPUTATION: "repeated work that can be factored out",
        IssueType.SUBOPTIMAL_ALGORITHM: "naive algorithm when a mathematically equivalent but faster form exists — look for: sum(A@B, dim) rewritten as A @ B.sum(dim), reduction over one axis followed by contraction, any case where operation order reduces total work via associativity/distributivity",
        IssueType.ASSOCIATIVITY_REORDER: "reorder associative ops to reduce FLOPs",
        IssueType.COMMON_SUBEXPRESSION: "same sub-expression computed multiple times",
        IssueType.ALGEBRAIC_SIMPLIFICATION: "identity ops, distributive law simplifications",
        IssueType.CACHEABLE_INTERMEDIATE: "reusable value that is recomputed unnecessarily — includes in-kernel intermediates AND model-level weight-derived statistics (column sums, row norms, packed transposes) that depend only on frozen parameters and should be computed once in __init__ and cached, not recomputed every forward() call",
        IssueType.LOOP_INVARIANT_CODE: "computation that does not depend on the loop variable or the per-call input — includes weight statistics (column sums, norms) recomputed every forward() call when weights are frozen, and any kernel launch overhead that could be amortized",
        IssueType.UNNECESSARY_MATERIALIZATION: "intermediate tensor written to HBM between kernels when algebraic reordering could eliminate it, or when the result could stay in registers — e.g. a [K] column-sum written to HBM only to be read once by the next kernel",
        IssueType.GEMM_SIMPLIFICATION: "GEMM has exploitable structure (symmetric, triangular, etc.)",
        IssueType.REDUCTION_TREE_SUBOPTIMAL: "naive serial reduction vs tree reduction",
        # DTYPE
        IssueType.DTYPE_FLOAT64: "float64 accumulator or I/O — 16-32x slower than float32 on XPU",
        IssueType.DTYPE_PRECISION: "unnecessary precision (e.g. float32 where float16 suffices)",
        IssueType.DTYPE_INPUT_CONVERSION: "redundant or wrong dtype conversion in hot path",
        # FUSION
        IssueType.UNFUSED_KERNELS: "multiple kernels that should be fused into one",
        IssueType.UNFUSED_ELEMENTWISE: "elementwise ops that could be epilogue-fused with GEMM",
        IssueType.UNFUSED_REDUCTION: "reduction that could be fused into preceding kernel",
        IssueType.FUSION_REGISTER_PRESSURE: "fusion would cause register spill — keep separate",
        # fusion_replaces_vendor intentionally excluded — it is a guard, not an issue.
        # If fusion would replace a vendor GEMM, return no fusion issue at all.
        IssueType.FUSION_NOOP: "intermediate is not materialized so fusion provides no benefit. NEVER return fusion_replaces_vendor as an issue: if fusion would replace a faster vendor GEMM, return no fusion issues at all",
        # MEMORY ACCESS
        IssueType.MISSING_BOUNDARY_CHECK: "tl.load/store missing mask for non-tile-divisible shapes",
        IssueType.TRANSPOSE_IN_LOOP: "matrix transposed inside K-loop — should be pre-packed",
        IssueType.MISSING_TMA: "could use tensor memory accelerator / descriptor for this access",
        IssueType.UNCOALESCED_ACCESS: "non-sequential memory access pattern — kills bandwidth",
        IssueType.DEVICE_HOST_SYNC: ".item() / float(tensor) / print in hot path — forces sync",
        IssueType.NON_CONTIGUOUS_INPUT: "strided/non-contiguous tensor passed to kernel without .contiguous()",
        IssueType.CACHE_EVICTION_RISK: "large tile or long liveness evicts L2 cache lines",
        IssueType.LONG_LIVENESS: "tensor live across many ops — occupancy/register pressure risk",
        IssueType.HIGH_REGISTER_PRESSURE: "too many live values — reduces occupancy",
        # BLOCK POINTERS
        IssueType.MANUAL_POINTER_ARITHMETIC: "manual offset arithmetic — replace with tl.make_block_ptr",
        IssueType.BLOCK_PTR_BOUNDARY_WRONG: "boundary_check uses booleans instead of dimension indices (0,1)",
        IssueType.BLOCK_PTR_MULTIPLE_OF_MISUSE: "tl.multiple_of() applied to Python scalar — only valid on tensors",
        IssueType.MISSING_BLOCK_POINTERS: "kernel could use block pointers for automatic boundary handling",
        # XPU SPECIFIC
        IssueType.SUBOPTIMAL_TILE_SIZE: "BLOCK_M/N/K too small — XPU prefers 256x256x32",
        IssueType.SUBOPTIMAL_WARPS: "num_warps too low — XPU large GEMMs prefer 32",
        IssueType.MISSING_GRF_MODE: "grf_mode not set — add grf_mode='256' for large tiles",
        IssueType.NO_SWIZZLING: "missing GROUP_SIZE_M swizzling for L2 cache reuse",
        IssueType.REPACK_IN_FORWARD: "weight transposed every forward() call — pack once in __init__",
        IssueType.MISSING_PACKED_TRANSPOSE: "RHS passed as [N,K] — should pre-pack as [K,N] contiguous",
        IssueType.SERIALIZED_N_TILES: "GEMM2 or reduction iterates over N serially — needs 2D parallel grid",
        IssueType.SIGMOID_SLOW_EXP: "sigmoid uses tl.exp — replace with tl.math.exp2 (faster on XPU)",
        IssueType.AUTOTUNE_DUPLICATE_PARAMS: "autotune Config param also has default value in kernel signature",
        # PERSISTENT KERNEL
        IssueType.MISSING_PERSISTENT: "large 2D GEMM grid (M>=1024, N>=1024) with tail quantization — consider persistent kernel. DO NOT flag for: 1D reduction kernels (output [M] or [M,1]), small grids (total tiles < 512), or batch_size < 256",
        IssueType.PERSISTENT_NUM_PROGS_HARDCODED: "NUM_PROGS hardcoded — query gpu_subslice_count() at runtime",
        # AUTOTUNING
        IssueType.MISSING_AUTOTUNE: "hardcoded tile/warp params — add @triton.autotune",
        IssueType.SUBOPTIMAL_AUTOTUNE_CONFIGS: "autotune configs missing important combinations (grf_mode, 32 warps, large tiles)",
        IssueType.AUTOTUNE_KEY_MISSING: "@triton.autotune missing key= — configs not re-evaluated on shape change",
        # DISCOVERY
        IssueType.OPEN_ENDED: (
            "A novel optimization not covered by any existing issue_type. "
            "Use ONLY when you have found a concrete, high-value, implementable "
            "optimization with no matching type above. "
            "You MUST populate open_ended_proposal with: "
            "(a) exactly what changes, "
            "(b) why it is mathematically/logically valid, "
            "(c) a before/after code sketch, "
            "(d) estimated speedup with reasoning. "
            "DO NOT use for vague suggestions. "
            "Good examples: sum(A@B,dim=1) rewritten as A@B.sum(0) eliminating "
            "O(M*N*K) GEMM; weight column-sum cached in __init__ eliminating "
            "134MB W read per call; two kernels whose HBM intermediate is "
            "eliminable by algebraic reordering."
        ),
    }

    # When building for SYCL, apply overrides and skip Triton-only issues
    if dsl == DSL.SYCL:
        skip = _SYCL_SKIP_ISSUES
        desc_overrides = _SYCL_DESCRIPTIONS
    else:
        skip = set()
        desc_overrides = {}

    # Group by stage
    by_stage: dict[OptimizationStage, list[IssueType]] = {s: [] for s in stage_order}
    for issue in IssueType:
        if issue in skip:
            continue
        stage = get_stage_for_issue(issue)
        if stage in by_stage:
            by_stage[stage].append(issue)

    lines = ["=== ISSUE CATEGORIES (issue_type values) ===", ""]
    for stage in stage_order:
        issues = by_stage[stage]
        if not issues:
            continue
        lines.append(f"{stage_labels[stage]}:")
        for issue in issues:
            desc = desc_overrides.get(issue, _descriptions.get(issue, ""))
            if desc:
                lines.append(f"  - {issue.value}: {desc}")
            else:
                lines.append(f"  - {issue.value}")
        lines.append("")

    return "\n".join(lines)


# Build once at import time — cheap, just string formatting
_ISSUE_CATEGORIES_BLOCK = _build_issue_categories(DSL.TRITON)
_SYCL_ISSUE_CATEGORIES_BLOCK = _build_issue_categories(DSL.SYCL)


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class AnalysisSignature(dspy.Signature):
    # The docstring is dynamically constructed so the issue list
    # always matches the live IssueType enum.
    __doc__ = f"""Analyze Triton kernel for optimization opportunities.

You are a world-class expert in Triton GPU/XPU kernel optimization,
numerical linear algebra, and high-performance computing.

Analyze the given Triton kernel code and, if available, the original PyTorch
implementation for higher-level algorithmic context.

You must identify ALL applicable optimizations across every category below.
Use your deep knowledge of GPU programming, Triton internals, Intel XPU
architecture, and mathematical optimization.

{_ISSUE_CATEGORIES_BLOCK}
IMPORTANT:
- Return issues as a JSON array of DetectedIssue objects.
- Each issue MUST have: issue_type (exact string from the list above),
  severity (1-5), description, suggested_fix, estimated_speedup.
- issue_type MUST be one of the exact strings listed above (e.g. "dtype_float64",
  "missing_grf_mode"). Do NOT invent new type names.
- For fused kernels, pay special attention to ALGORITHMIC issues.
- Return empty array [] ONLY if the kernel is already optimal.

OPEN-ENDED DISCOVERY (issue_type="open_ended"):
After checking all categories above, ask yourself: is there a high-value
optimization that does not fit any existing type? If yes, use issue_type="open_ended"
and populate open_ended_proposal with the full proposal. Requirements:
  - Concrete and implementable — not a vague observation
  - Mathematically or logically justified
  - Includes a before/after code sketch in open_ended_proposal
  - Includes estimated speedup with reasoning
Examples that qualify as open_ended:
  * sum(x @ W.T, dim=1) rewritten as x @ W.sum(dim=0) — eliminates O(M*N*K) GEMM
  * Weight statistic (colsum, norm) recomputed every forward() — cache in __init__
  * Two-kernel pipeline where the HBM intermediate can be eliminated algebraically
Examples that do NOT qualify (use the named type instead):
  * "use better tile sizes" → use suboptimal_tile_size
  * "add autotuning" → use missing_autotune
  * "fuse these kernels" → use unfused_kernels
"""

    kernel_code: dspy.Code["python"] = dspy.InputField(desc="Triton kernel source code to analyze.")
    reference_code: str = dspy.InputField(
        desc="Original PyTorch implementation. May be empty if not available."
    )
    problem_context: str = dspy.InputField(
        desc="Problem size, FLOP count, target device, and other context."
    )

    knowledge_base_context: str = dspy.InputField(
        desc="Critical constraints from the knowledge base that commonly manifest "
        "as bugs in Triton/XPU kernels. Use these to guide detection: if a "
        "constraint is violated in the code, flag the corresponding issue. "
        "Empty string if KB is disabled."
    )
    issues_found: list[DetectedIssue] = dspy.OutputField(
        desc="List of detected issues. Return empty array [] only if kernel is already optimal."
    )


class SyclAnalysisSignature(dspy.Signature):
    __doc__ = f"""Analyze SYCL/CUTLASS C++ kernel for optimization opportunities on Intel XPU.

You are a world-class expert in SYCL, CUTLASS/XeTLA, and Intel XPU GPU kernel optimization.

Analyze the given SYCL C++ kernel code and identify ALL applicable optimizations.

=== SYCL/CUTLASS OPTIMIZATION KNOBS ===
- TileShape: Shape<_M, _N, _K> (e.g. 256x256x32, 128x128x64, 128x256x32)
- PipelineStages: 2, 3, or 4 (more = more prefetching, but more register pressure)
- MMA Atom: XE_DPAS_TT<SubgroupSize, AccumType, InputType> — SubgroupSize 4 or 8
- Dispatch Policy: MainloopXeL1Staged (L1 cached), MainloopXeL0Staged
- Data types: bfloat16_t/half_t inputs, float/bfloat16_t accumulators
- Memory layout: RowMajor vs ColumnMajor for A, B, C, D matrices
- Epilogue fusion: LinearCombination, bias addition, activation functions via FusionCallbacks
- GmemTiledCopy: void (auto) or explicit copy atoms for fine-grained control

{_SYCL_ISSUE_CATEGORIES_BLOCK}
IMPORTANT:
- Return issues as a JSON array of DetectedIssue objects.
- Each issue MUST have: issue_type (exact string from the list above),
  severity (1-5), description, suggested_fix, estimated_speedup.
- issue_type MUST be one of the exact strings listed above.
- Return empty array [] ONLY if the kernel is already optimal.
"""

    kernel_code: dspy.Code["cpp"] = dspy.InputField(
        desc="SYCL/CUTLASS C++ kernel source code to analyze."
    )
    reference_code: str = dspy.InputField(
        desc="Reference implementation or description. May be empty."
    )
    problem_context: str = dspy.InputField(
        desc="Problem size, FLOP count, target device, and other context."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Constraints from the knowledge base for SYCL/XPU kernels. "
        "Empty string if KB is disabled."
    )
    issues_found: list[DetectedIssue] = dspy.OutputField(
        desc="List of detected issues. Return empty array [] only if kernel is already optimal."
    )


# ---------------------------------------------------------------------------
# Analyzer Agent
# ---------------------------------------------------------------------------


class AnalyzerAgent:
    """LLM-based analyzer for Triton kernels."""

    def __init__(self, knowledge_base=None, dsl: DSL | str = DSL.TRITON):
        self.knowledge_base = knowledge_base
        self.dsl = DSL(dsl) if isinstance(dsl, str) else dsl
        sig = SyclAnalysisSignature if self.dsl == DSL.SYCL else AnalysisSignature
        self.predictor = dspy.Predict(sig)

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
        raw_issues = None

        kb_context = self._get_kb_context()
        if kb_context:
            logger.info(
                "Analyzer KB context: %d chars, %d constraints",
                len(kb_context),
                kb_context.count("[CRITICAL]") + kb_context.count("[WARNING]"),
            )
        try:
            result = self.predictor(
                kernel_code=triton_code,
                reference_code=pytorch_code or "",
                problem_context=problem_context,
                knowledge_base_context=kb_context,
            )
            raw_issues = result.issues_found
            logger.debug("LLM issues_found raw: %s", raw_issues)

        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            # Don't return early — process whatever partial result we have

        # --- Robust issue parsing ---
        # DSPy may return fully validated DetectedIssue objects, raw dicts,
        # or a mix. Handle all cases and log anything that couldn't be parsed.
        if raw_issues:
            for raw in raw_issues:
                issue = self._coerce_issue(raw)
                if issue is not None:
                    issues.append(issue)

        logger.info("Parsed %d/%d issues from LLM", len(issues), len(raw_issues or []))

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

    # ------------------------------------------------------------------

    def _coerce_issue(self, raw) -> DetectedIssue | None:
        """
        Convert a raw LLM output item to a DetectedIssue.

        Handles:
          - Already-validated DetectedIssue objects (DSPy parsed correctly)
          - Plain dicts with string issue_type
          - issue_type strings that are LLM variants (e.g. "slow_sigmoid")
            — routed via get_stage_for_issue_str so they still run the right stage
        """
        # Already the right type
        if isinstance(raw, DetectedIssue):
            return raw

        # Dict from LLM output that DSPy didn't fully validate
        if isinstance(raw, dict):
            issue_type_raw = raw.get("issue_type", "")

            # Try exact IssueType match first
            try:
                issue_type = IssueType(issue_type_raw)
            except (ValueError, KeyError):
                # Try case-insensitive match
                try:
                    issue_type = IssueType(issue_type_raw.lower())
                except (ValueError, KeyError):
                    # Unknown type — check if we can still route it to a stage
                    stage = get_stage_for_issue_str(issue_type_raw)
                    if stage == OptimizationStage.ANALYSIS:
                        logger.warning(
                            "Dropping unknown issue_type %r — not in IssueType enum "
                            "and no keyword match. Raw: %s",
                            issue_type_raw,
                            raw,
                        )
                        return None
                    else:
                        # Known stage but unknown exact type — log and skip
                        # (pipeline needs a valid IssueType to route correctly)
                        logger.warning(
                            "Unknown issue_type %r inferred as stage=%s via keyword, "
                            "but cannot create DetectedIssue without a valid IssueType. "
                            "Add %r to the IssueType enum to enable this optimization.",
                            issue_type_raw,
                            stage.value,
                            issue_type_raw,
                        )
                        return None

            # For open_ended issues, log the proposal so it's visible in output
            if issue_type == IssueType.OPEN_ENDED:
                proposal = (
                    raw.get("open_ended_proposal")
                    or raw.get("suggested_fix", "")
                    or raw.get("description", "")
                )
                logger.info(
                    "OPEN_ENDED optimization proposed (will run DISCOVERY stage):\n%s",
                    proposal[:500] if proposal else "(no proposal text)",
                )

            try:
                return DetectedIssue(
                    issue_type=issue_type,
                    severity=int(raw.get("severity", 3)),
                    location=raw.get("location"),
                    description=raw.get("description", ""),
                    suggested_fix=raw.get("suggested_fix", ""),
                    estimated_speedup=raw.get("estimated_speedup"),
                    open_ended_proposal=(
                        raw.get("open_ended_proposal")
                        if issue_type == IssueType.OPEN_ENDED
                        else None
                    ),
                )
            except Exception as e:
                logger.warning("Failed to construct DetectedIssue from dict %s: %s", raw, e)
                return None

        logger.warning("Unexpected issue type %s: %r — skipping", type(raw).__name__, raw)
        return None

    # ------------------------------------------------------------------

    def _get_kb_context(self) -> str:
        """Return critical constraints from KB for the analyzer to check against."""
        if self.knowledge_base is None:
            return ""
        try:
            from xe_forge.models import OptimizationStage

            lines_out = ["=== KB Constraints (check these against the code) ==="]
            seen = set()
            for stage in OptimizationStage:
                if stage == OptimizationStage.ANALYSIS:
                    continue
                for c in self.knowledge_base.constraints_for_stage(stage):
                    if c.id in seen or c.severity not in ("critical", "warning"):
                        continue
                    seen.add(c.id)
                    desc = c.description.strip()[:300]
                    lines_out.append(f"[{c.severity.upper()}] {c.name}: {desc}")
            result = "\n".join(lines_out)
            return result[:6000] + "\n...[truncated]" if len(result) > 6000 else result
        except Exception as e:
            logger.debug("KB context failed: %s", e)
            return ""

    def _build_problem_context(
        self,
        input_shapes: list[tuple] | None,
        flop: float | None,
        target_dtype: str | None = None,
    ) -> str:
        from xe_forge.config import get_config
        from xe_forge.prompts import PromptLibrary

        cfg = get_config()
        prompts = PromptLibrary(dsl=cfg.device_config.dsl, device_type=cfg.device_config.device)
        lines = [prompts.target_device_line(), ""]

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
