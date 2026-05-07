"""
PlannerAgent - LLM-based stage ordering for the optimization pipeline.

Given the detected issues and available stages, produces an optimal ordered
list of stages to apply, with reasoning.  Falls back to DEFAULT_STAGE_ORDER
(filtered to needed stages) if the LLM call fails or produces invalid output.
"""

from __future__ import annotations

import logging

import dspy

from xe_forge.models import IssueType, KernelAnalysis, OptimizationStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default ordering — used as fallback and as the "safe baseline" description
# shown to the planner so it knows what it is overriding.
# ---------------------------------------------------------------------------

DEFAULT_STAGE_ORDER: list[OptimizationStage] = [
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.DISCOVERY,
    OptimizationStage.DTYPE_FIX,
    OptimizationStage.FUSION,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.BLOCK_POINTERS,
    OptimizationStage.PERSISTENT_KERNEL,
    OptimizationStage.DEVICE_SPECIFIC,
    OptimizationStage.AUTOTUNING,
]

# ---------------------------------------------------------------------------
# Stage dependency rules — hard constraints the planner must respect.
# These are enforced in _validate_and_fix() regardless of what the LLM returns.
# ---------------------------------------------------------------------------

# (before, after): before must appear before after if both are present
_HARD_DEPENDENCIES: list[tuple[OptimizationStage, OptimizationStage]] = [
    (OptimizationStage.ALGORITHMIC, OptimizationStage.FUSION),
    (OptimizationStage.ALGORITHMIC, OptimizationStage.DTYPE_FIX),
    (OptimizationStage.DISCOVERY, OptimizationStage.DTYPE_FIX),
    (OptimizationStage.DISCOVERY, OptimizationStage.FUSION),
    (OptimizationStage.DTYPE_FIX, OptimizationStage.FUSION),
    (OptimizationStage.MEMORY_ACCESS, OptimizationStage.BLOCK_POINTERS),
    (OptimizationStage.FUSION, OptimizationStage.DEVICE_SPECIFIC),
    (OptimizationStage.BLOCK_POINTERS, OptimizationStage.DEVICE_SPECIFIC),
    (OptimizationStage.DEVICE_SPECIFIC, OptimizationStage.AUTOTUNING),
]


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class PlanningSignature(dspy.Signature):
    """Determine the optimal order to apply optimization stages to a GPU kernel.

    You are an expert in GPU kernel optimization.
    You have analyzed a kernel and found issues in specific optimization categories.
    Your task: decide the OPTIMAL ORDER to apply the available stages.

    === ORDERING PRINCIPLES ===

    Mathematical correctness first:
      - ALGORITHMIC and DISCOVERY before everything else — structural rewrites may
        eliminate entire categories of lower-level issues. A rewrite that removes
        a kernel makes BLOCK_POINTERS for that kernel irrelevant.

    Dtype early, but after structure:
      - DTYPE_FIX after ALGORITHMIC/DISCOVERY — convert to fp16 after the algorithm
        is settled, not before (avoids converting code that gets rewritten anyway).

    Fusion after dtype:
      - FUSION after DTYPE_FIX — fuse already-converted kernels. Fusing then
        converting can produce suboptimal mixed-precision boundaries.
      - Exception: if FUSION is the dominant issue and DTYPE is minor, move FUSION
        earlier to reduce the number of kernels before low-level tuning.

    Memory before block pointers:
      - MEMORY_ACCESS before BLOCK_POINTERS — fix coalescing and layout issues
        before converting to block pointer API (block pointers assume good layout).

    Low-level tuning last:
      - DEVICE_SPECIFIC and AUTOTUNING last — tile sizes, warp counts, and autotune
        configs should be applied to the final kernel structure, not intermediate forms.

    Persistent kernel placement:
      - PERSISTENT_KERNEL before DEVICE_SPECIFIC — persistence changes the kernel
        structure; device tuning should happen on the persistent form.
      - Skip PERSISTENT_KERNEL entirely if grid size is small (< 512 tiles) or
        if the kernel produces a scalar/[M] output — persistence won't help.

    === RULES ===
    - Only include stages from the available_stages list (stages with detected issues).
    - Do NOT add stages that have no detected issues.
    - Do NOT include ANALYSIS.
    - Your ordered_stages output must be a JSON array of stage value strings,
      e.g. ["algorithmic", "dtype_fix", "device_specific"].
    - Every stage in ordered_stages must appear in available_stages.
    - Provide clear rationale explaining why you chose this specific order.
    """

    available_stages: str = dspy.InputField(
        desc="JSON object mapping stage_name → [issue_type, ...] for stages with detected issues."
    )
    issue_summary: str = dspy.InputField(
        desc="Summary of detected issues with severities and descriptions."
    )
    kernel_context: str = dspy.InputField(
        desc="Brief description of what the kernel does, input shapes, and compute characteristics."
    )

    ordered_stages: list[str] = dspy.OutputField(
        desc='Ordered list of stage names to apply, e.g. ["algorithmic", "dtype_fix"]. '
        "Must be a subset of available_stages keys. Most impactful stages first."
    )
    rationale: str = dspy.OutputField(
        desc="Explanation of why this order was chosen, referencing the specific issues found."
    )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class PlannerAgent:
    """LLM-based stage ordering. Falls back to default order on any failure."""

    def __init__(self) -> None:
        self.predictor = dspy.Predict(PlanningSignature)

    def plan(
        self,
        stages_needed: dict[OptimizationStage, list[str]],
        analysis: KernelAnalysis,
        input_shapes: list | None = None,
        flop: float | None = None,
    ) -> list[OptimizationStage]:
        """
        Return an ordered list of stages to apply.

        stages_needed: {stage: [issue_type_values]} — only stages with issues.
        Falls back to default ordering (filtered to needed stages) on failure.
        """
        if not stages_needed:
            return []

        # Always fall back to default if only one stage needed — no ordering decision
        if len(stages_needed) == 1:
            return list(stages_needed.keys())

        fallback = _default_filtered(stages_needed)

        try:
            available_str = _format_available(stages_needed)
            issue_summary = _format_issue_summary(analysis)
            kernel_ctx = _format_kernel_context(analysis, input_shapes, flop)

            result = self.predictor(
                available_stages=available_str,
                issue_summary=issue_summary,
                kernel_context=kernel_ctx,
            )

            ordered = _parse_and_validate(result.ordered_stages, stages_needed, fallback)

            logger.info("Stage order (agent): %s", [s.value for s in ordered])
            logger.info("Planner rationale: %s", result.rationale[:300] if result.rationale else "")

            return ordered

        except Exception as e:
            logger.warning("PlannerAgent failed (%s) — using default stage order", e)
            return fallback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_filtered(stages_needed: dict[OptimizationStage, list[str]]) -> list[OptimizationStage]:
    """Default order filtered to only stages that have detected issues."""
    return [s for s in DEFAULT_STAGE_ORDER if s in stages_needed]


def _format_available(stages_needed: dict[OptimizationStage, list[str]]) -> str:
    import json

    return json.dumps({s.value: issues for s, issues in stages_needed.items()}, indent=2)


def _format_issue_summary(analysis: KernelAnalysis) -> str:
    lines = []
    for iss in analysis.detected_issues:
        proposal = ""
        if iss.issue_type == IssueType.OPEN_ENDED and iss.open_ended_proposal:
            proposal = f" | proposal: {iss.open_ended_proposal[:100]}"
        lines.append(
            f"  [{iss.severity}/5] {iss.issue_type.value}: {iss.description[:80]}{proposal}"
        )
    return "\n".join(lines) if lines else "No issues"


def _format_kernel_context(
    analysis: KernelAnalysis,
    input_shapes: list | None,
    flop: float | None,
) -> str:
    parts = [f"kernel: {analysis.kernel_name}"]
    if input_shapes:
        parts.append(f"input_shapes: {input_shapes}")
    if flop:
        parts.append(f"flop: {flop:.2e}")
    parts.append(f"uses_block_pointers: {analysis.uses_block_pointers}")
    parts.append(f"is_persistent: {analysis.is_persistent}")
    parts.append(f"has_fusion_opportunity: {analysis.has_fusion_opportunity}")
    parts.append(f"has_algorithmic_opportunity: {analysis.has_algorithmic_opportunity}")
    return ", ".join(parts)


def _parse_and_validate(
    raw_ordered: list,
    stages_needed: dict[OptimizationStage, list[str]],
    fallback: list[OptimizationStage],
) -> list[OptimizationStage]:
    """
    Parse the LLM output, validate each stage name, enforce hard dependencies,
    and fill in any missing needed stages at the end.
    """
    # Build stage value → enum map for fast lookup
    valid = {s.value: s for s in OptimizationStage}

    parsed: list[OptimizationStage] = []
    seen: set[OptimizationStage] = set()

    for item in raw_ordered or []:
        s_str = str(item).strip().lower()
        stage = valid.get(s_str)
        if stage is None:
            logger.debug("Planner returned unknown stage %r — skipping", s_str)
            continue
        if stage == OptimizationStage.ANALYSIS:
            continue
        if stage not in stages_needed:
            logger.debug("Planner returned stage %r with no issues — skipping", s_str)
            continue
        if stage in seen:
            continue
        parsed.append(stage)
        seen.add(stage)

    # Any needed stage the LLM omitted gets appended in default order
    for s in fallback:
        if s not in seen:
            logger.debug("Planner omitted needed stage %r — appending at end", s.value)
            parsed.append(s)
            seen.add(s)

    # Enforce hard dependency constraints via topological correction
    parsed = _enforce_dependencies(parsed)

    if not parsed:
        logger.warning("Planner produced empty stage list — using fallback")
        return fallback

    return parsed


def _enforce_dependencies(stages: list[OptimizationStage]) -> list[OptimizationStage]:
    """
    Enforce _HARD_DEPENDENCIES by moving stages that violate ordering constraints.
    """
    stages = list(stages)

    for before, after in _HARD_DEPENDENCIES:
        if before not in stages or after not in stages:
            continue
        i_before = stages.index(before)
        i_after = stages.index(after)
        if i_before > i_after:
            stages.remove(before)
            i_after = stages.index(after)
            stages.insert(i_after, before)
            logger.debug("Dependency fix: moved %s before %s", before.value, after.value)

    return stages
