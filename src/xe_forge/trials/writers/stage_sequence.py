"""Stage-sequence writer: runs the existing analyzer + planner + per-stage
optimizer pipeline on the parent kernel and returns the final code.

This is the migration glue: the body of today's XeForgePipeline.optimize()
becomes one writer choice. Children build on parents (not on the root), so
even with tree_walk search this is a genuine upgrade over the flat best_k loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xe_forge.knowledge.patterns import get_stage_for_issue
from xe_forge.models import OptimizationStage
from xe_forge.planner import DEFAULT_STAGE_ORDER as PLANNER_DEFAULT_STAGE_ORDER
from xe_forge.trials.writers.base import TrialWriter

if TYPE_CHECKING:
    from xe_forge.agents import AnalyzerAgent, Optimizer
    from xe_forge.planner import PlannerAgent
    from xe_forge.trials.context import TrialContext
    from xe_forge.trials.search import TrialProposal

logger = logging.getLogger(__name__)


class StageSequenceWriter(TrialWriter):
    """One trial = one full analyze -> plan -> per-stage pass over the parent."""

    def __init__(
        self,
        analyzer: "AnalyzerAgent",
        planner: "PlannerAgent",
        optimizer: "Optimizer",
        display_name: str = "Model",
    ):
        self.analyzer = analyzer
        self.planner = planner
        self.optimizer = optimizer
        self.display_name = display_name

    def write(
        self,
        parent_code: str,
        proposal: "TrialProposal",
        ctx: "TrialContext",
        vtune_report: str = "",
    ) -> str:
        input_shapes = ctx.input_shapes
        flop = ctx.flop
        dtype = ctx.dtype
        pytorch_code = ctx.pytorch_code
        init_args = ctx.init_args
        xpu_config = ctx.xpu_config or {}
        target_dtype = _dtype_str(dtype)

        logger.info("[stage_sequence] analyzing parent kernel")
        analysis = self.analyzer.analyze(
            parent_code,
            pytorch_code,
            self.display_name,
            input_shapes,
            flop,
            target_dtype=target_dtype,
        )

        if not analysis.detected_issues:
            logger.info("[stage_sequence] no issues detected — returning parent unchanged")
            return parent_code

        stages_needed: dict[OptimizationStage, list[str]] = {}
        for iss in analysis.detected_issues:
            st = get_stage_for_issue(iss.issue_type)
            stages_needed.setdefault(st, []).append(iss.issue_type.value)

        stages_to_apply = self.planner.plan(
            stages_needed=stages_needed,
            analysis=analysis,
            input_shapes=input_shapes,
            flop=flop,
        )
        logger.info(
            "[stage_sequence] plan: %s",
            [s.value for s in stages_to_apply],
        )
        if not stages_to_apply:
            return parent_code

        current_code = parent_code
        for stage in stages_to_apply:
            if stage == OptimizationStage.ANALYSIS:
                continue

            logger.info("[stage_sequence] stage=%s", stage.value)
            stage_result = self.optimizer.optimize_stage(
                code=current_code,
                stage=stage,
                analysis=analysis,
                xpu_config=xpu_config,
                kernel_name=ctx.kernel_name,
                input_shapes=input_shapes,
                flop=flop,
                dtype=dtype,
                pytorch_code=pytorch_code,
                init_args=init_args,
                vtune_report=vtune_report,
                perf_context={"speedup_so_far": None},
            )

            if (
                stage_result.success
                and stage_result.output_code
                and stage_result.output_code != current_code
            ):
                current_code = stage_result.output_code
                # Re-analyze so the next stage sees the updated code.
                analysis = self.analyzer.analyze(
                    current_code,
                    pytorch_code,
                    self.display_name,
                    input_shapes,
                    flop,
                    target_dtype=target_dtype,
                )
            elif not stage_result.success:
                logger.warning(
                    "[stage_sequence] stage %s failed: %s",
                    stage.value,
                    stage_result.error_message,
                )

        return current_code


def _dtype_str(dtype) -> str | None:
    try:
        import torch

        if dtype is None:
            return None
        mapping = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        return mapping.get(dtype)
    except Exception:
        return None


# Silence unused import warning — kept for IDE jump-to-definition.
_ = PLANNER_DEFAULT_STAGE_ORDER
