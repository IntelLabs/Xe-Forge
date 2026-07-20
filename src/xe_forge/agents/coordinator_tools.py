"""
CoordinatorState and tool factory functions for CoordinatorAgent.

Code never flows through tool arguments — all tools are closures over CoordinatorState
so the coordinator LLM operates at semantic level ("apply block_pointers stage")
without ever handling raw kernel code.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from xe_forge.models import KernelAnalysis, OptimizationStage, StageResult

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorState:
    """Mutable kernel optimization state shared across all coordinator tools."""

    original_code: str
    current_code: str
    best_code: str
    best_speedup: float = 1.0
    stages_tried: list[str] = field(default_factory=list)
    stages_succeeded: list[str] = field(default_factory=list)
    analysis: KernelAnalysis | None = None
    profile_text: str = ""
    attempt_log: list[str] = field(default_factory=list)
    stage_results: list[StageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_analysis_for_coordinator(analysis: KernelAnalysis) -> str:
    """Compact analysis summary for the coordinator — no code."""
    if not analysis.detected_issues:
        return "No optimization opportunities detected. Kernel appears optimal."

    from xe_forge.knowledge.patterns import get_stage_for_issue

    by_stage: dict[OptimizationStage, list] = {}
    for iss in analysis.detected_issues:
        stage = get_stage_for_issue(iss.issue_type)
        by_stage.setdefault(stage, []).append(iss)

    lines = [f"Detected {len(analysis.detected_issues)} issue(s):"]
    for stage, issues in sorted(by_stage.items(), key=lambda kv: kv[0].value):
        lines.append(f"\n  Stage: {stage.value}")
        for iss in sorted(issues, key=lambda i: -i.severity):
            lines.append(f"    [{iss.severity}/5] {iss.issue_type.value}: {iss.description[:120]}")
            if iss.suggested_fix:
                lines.append(f"      Fix: {iss.suggested_fix[:100]}")

    if analysis.has_algorithmic_opportunity:
        lines.append(
            "\nNote: algorithmic opportunities detected — apply ALGORITHMIC/DISCOVERY first."
        )
    return "\n".join(lines)


def _parse_stage(stage_str: str) -> OptimizationStage:
    """Tolerant string → OptimizationStage parser."""
    from xe_forge.knowledge.patterns import get_stage_for_issue_str

    s = stage_str.strip().lower()
    # Direct enum value match
    try:
        return OptimizationStage(s)
    except ValueError:
        pass
    # Keyword inference via patterns module
    inferred = get_stage_for_issue_str(s)
    if inferred != OptimizationStage.ANALYSIS:
        return inferred
    raise ValueError(
        f"Unknown stage {stage_str!r}. Valid values: "
        + ", ".join(s.value for s in OptimizationStage if s != OptimizationStage.ANALYSIS)
    )


def _summarize_stage_result(stage: str, result: StageResult, prev_speedup: float) -> str:
    """Single-line summary for coordinator tool return value."""
    if result.success and result.speedup and result.speedup > 1.0:
        delta = result.speedup - prev_speedup
        return f"IMPROVED: {stage} → {result.speedup:.3f}x speedup" + (
            f" (+{delta:.3f}x vs previous best)" if prev_speedup > 0 else ""
        )
    if result.success:
        return f"APPLIED: {stage} — no measurable speedup (kernel unchanged or within noise)"
    return f"FAILED: {stage} — {result.error_message or 'no details'}"


# ---------------------------------------------------------------------------
# Tool factory functions
# ---------------------------------------------------------------------------


def make_analyze_tool(
    state: CoordinatorState,
    analyzer,
    pytorch_code: str,
    kernel_name: str | None,
    input_shapes,
    flop,
    dtype,
) -> Callable:
    def analyze_kernel() -> str:
        """Analyze the current kernel for optimization opportunities.

        Returns a structured list of detected issues grouped by stage with severities
        and suggested fixes. Call this first and again after significant changes.
        """
        try:
            analysis = analyzer.analyze(
                state.current_code,
                pytorch_code or "",
                kernel_name or "kernel",
                input_shapes,
                flop,
                target_dtype=str(dtype) if dtype is not None else None,
            )
            state.analysis = analysis
            summary = _format_analysis_for_coordinator(analysis)
            logger.info(
                "Coordinator: analyze_kernel returned %d issues", len(analysis.detected_issues)
            )
            return summary
        except Exception as e:
            logger.warning("analyze_kernel failed: %s", e)
            return f"Analysis failed: {e}"

    return analyze_kernel


def make_retrieve_patterns_tool(state: CoordinatorState, knowledge_base) -> Callable:
    def retrieve_patterns(stage: str) -> str:
        """Retrieve knowledge base patterns and constraints for a given optimization stage.

        Use before applying a stage to understand best practices and known gotchas.
        stage: e.g. "block_pointers", "device_specific", "algorithmic"
        """
        if knowledge_base is None:
            return f"No knowledge base available for stage {stage!r}. Rely on LLM knowledge."
        try:
            stage_enum = _parse_stage(stage)
            result = knowledge_base.format_for_stage(stage_enum)
            return result or f"No patterns found for stage {stage!r}."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.warning("retrieve_patterns failed: %s", e)
            return f"Pattern retrieval failed: {e}"

    return retrieve_patterns


def make_apply_stage_tool(
    state: CoordinatorState,
    optimizer,
    xpu_config: dict,
    kernel_name: str | None,
    input_shapes,
    spec_dims,
    flop,
    dtype,
    pytorch_code: str,
    init_args,
    input_dtypes,
    vtune_report_ref: list[str],
) -> Callable:
    def apply_stage(stage: str, hints: str = "") -> str:
        """Apply a single optimization stage to the current kernel.

        If the optimization improves performance, the new kernel automatically becomes
        the current version. Returns speedup achieved or a failure reason.
        stage: e.g. "block_pointers", "device_specific", "dtype_fix"
        hints: optional free-text guidance to pass as extra context (e.g. "focus on reduction loop")
        """
        try:
            stage_enum = _parse_stage(stage)
            stage = stage_enum.value  # normalize to canonical stage name
        except ValueError as e:
            return f"Error: {e}"

        if stage in state.stages_tried:
            return f"Stage {stage!r} already tried. Use get_status() to see what's been attempted."

        if state.analysis is None:
            return "Must call analyze_kernel() first to identify issues before applying stages."

        state.stages_tried.append(stage)
        logger.info("Coordinator: applying stage %s", stage)

        # Build a minimal analysis that includes only this stage's issues
        prev_speedup = state.best_speedup

        try:
            vtune = vtune_report_ref[0] if vtune_report_ref else ""
            perf_ctx = {
                "original_ms": None,
                "speedup_so_far": prev_speedup if prev_speedup > 0 else None,
            }

            stage_result = optimizer.optimize_stage(
                code=state.current_code,
                stage=stage_enum,
                analysis=state.analysis,
                xpu_config=xpu_config,
                kernel_name=kernel_name,
                input_shapes=input_shapes,
                spec_dims=spec_dims,
                flop=flop,
                dtype=dtype,
                pytorch_code=pytorch_code or "",
                init_args=init_args,
                vtune_report=vtune,
                perf_context=perf_ctx,
                input_dtypes=input_dtypes,
            )
        except Exception as e:
            logger.warning("apply_stage %s failed with exception: %s", stage, e)
            state.stage_results.append(
                StageResult(
                    stage=stage_enum,
                    success=False,
                    input_code=state.current_code,
                    output_code=state.current_code,
                    error_message=str(e),
                )
            )
            return f"FAILED: {stage} — exception: {e}"

        state.stage_results.append(stage_result)
        summary = _summarize_stage_result(stage, stage_result, prev_speedup)

        if (
            stage_result.success
            and stage_result.output_code
            and stage_result.output_code != state.current_code
        ):
            state.current_code = stage_result.output_code
            state.stages_succeeded.append(stage)
            spd = stage_result.speedup or 0.0
            if spd > state.best_speedup:
                state.best_code = stage_result.output_code
                state.best_speedup = spd

        state.attempt_log.append(summary)
        logger.info("Coordinator: %s", summary)
        return summary

    return apply_stage


def make_profile_tool(
    state: CoordinatorState,
    profiler,
    kernel_name: str | None,
    spec_path: str | None,
    variant_type: str,
    vtune_report_ref: list[str],
) -> Callable:
    def profile_kernel() -> str:
        """Profile the current kernel to identify hardware-level bottlenecks.

        Use when you suspect cache misses, low XVE utilization, or memory bandwidth issues
        that static analysis cannot reveal. Requires VTune to be configured.
        """
        if profiler is None or not profiler.available():
            return (
                "VTune profiler not available. Use static analysis from analyze_kernel() instead."
            )
        if spec_path is None:
            return "Cannot profile: no spec_path provided (required for VTune benchmark config)."
        try:
            tmp = Path(tempfile.mkdtemp()) / f"{kernel_name or 'kernel'}_coordinator_profile.py"
            tmp.write_text(state.current_code)
            profile_result = profiler.profile(str(tmp), spec_path=spec_path, variant=variant_type)
            if profile_result.error:
                return f"Profile failed: {profile_result.error}"
            report = profile_result.format_for_llm()
            state.profile_text = report
            vtune_report_ref[0] = report
            logger.info("Coordinator: profile_kernel completed")
            return report
        except Exception as e:
            logger.warning("profile_kernel failed: %s", e)
            return f"Profile failed: {e}"

    return profile_kernel


def make_benchmark_tool(
    state: CoordinatorState,
    executor,
    kernel_name: str | None,
    input_shapes,
    flop,
    dtype,
    spec_dims,
    init_args,
    input_dtypes,
) -> Callable:
    def benchmark_current() -> str:
        """Benchmark the current kernel against the original baseline.

        Returns speedup and correctness verdict. Use to verify cumulative improvement.
        """
        if executor is None:
            return "No executor available — cannot benchmark."
        if not input_shapes and spec_dims is None:
            return "No input shapes or spec_dims — cannot benchmark."
        try:
            is_sycl = spec_dims is not None and not input_shapes

            if is_sycl:
                comparison = executor.compare_kernels(
                    original_code=state.original_code,
                    optimized_code=state.current_code,
                    dims=spec_dims,
                )
            else:
                comparison = executor.compare_kernels(
                    original_code=state.original_code,
                    optimized_code=state.current_code,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    flop=flop,
                    dtype=dtype,
                    init_args=init_args,
                    input_dtypes=input_dtypes,
                )

            if not comparison.optimized_correct:
                return (
                    f"INCORRECT: {comparison.feedback_message or 'kernel produces wrong results'}"
                )

            speedup = comparison.speedup or 1.0
            orig_ms = getattr(comparison, "original_time_us", None)
            opt_ms = getattr(comparison, "optimized_time_us", None)

            lines = [f"Speedup: {speedup:.3f}x"]
            if orig_ms and opt_ms:
                lines.append(f"Original: {orig_ms:.1f} µs  →  Optimized: {opt_ms:.1f} µs")
            if comparison.is_slower:
                lines.append("WARNING: current code is SLOWER than original")
            elif speedup > 1.0:
                lines.append("Improvement confirmed vs original baseline.")

            if speedup > state.best_speedup:
                state.best_speedup = speedup
                state.best_code = state.current_code

            return "\n".join(lines)
        except Exception as e:
            logger.warning("benchmark_current failed: %s", e)
            return f"Benchmark failed: {e}"

    return benchmark_current


def make_status_tool(state: CoordinatorState) -> Callable:
    def get_status() -> str:
        """Return a summary of the optimization state: speedup achieved, stages tried/succeeded, pending issues.

        Use to review progress and decide next steps.
        """
        lines = ["=== Coordinator Status ==="]
        lines.append(f"Best speedup so far: {state.best_speedup:.3f}x")
        lines.append(
            f"Stages tried ({len(state.stages_tried)}): {', '.join(state.stages_tried) or 'none'}"
        )
        lines.append(
            f"Stages succeeded ({len(state.stages_succeeded)}): {', '.join(state.stages_succeeded) or 'none'}"
        )
        if state.analysis:
            from xe_forge.knowledge.patterns import get_stage_for_issue

            tried = {s.strip().lower() for s in state.stages_tried}
            remaining = sorted(
                {
                    get_stage_for_issue(iss.issue_type).value
                    for iss in state.analysis.detected_issues
                }
                - tried
            )
            if remaining:
                lines.append(f"Remaining stages to attempt: {', '.join(remaining[:8])}")
            else:
                lines.append("All detected stages have been attempted.")
        if state.attempt_log:
            lines.append("\nAttempt history:")
            for entry in state.attempt_log[-5:]:
                lines.append(f"  {entry}")
        return "\n".join(lines)

    return get_status
