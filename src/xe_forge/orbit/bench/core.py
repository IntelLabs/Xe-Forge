"""
The measurement backbone: runs a workload, times it properly, and emits structured
JSON. Enforces two rules — no point values (samples plus an interval, always) and
interleaved comparisons. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from xe_forge.orbit import stats
from xe_forge.orbit.executor import Executor, LocalExecutor, RunResult
from xe_forge.orbit.models import (
    MetricEstimate,
    WorkloadMeasurement,
    WorkloadSpec,
)

# Extracts named metrics from one process run. Adapters supply these; the generic
# path has none and reports wall time only.
MetricExtractor = Callable[[RunResult], dict[str, float]]


@dataclass
class BenchResult:
    """Raw samples from a benchmarking session, before they become an artifact."""

    wall_times: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warmup_discarded: int = 0
    # The last failed run, kept whole so a workload that never produced a sample can
    # be diagnosed by the enablement classifier instead of merely reported dead.
    last_failure: RunResult | None = None

    def add(self, result: RunResult, extracted: dict[str, float]) -> None:
        self.wall_times.append(result.duration_s)
        for key, value in extracted.items():
            self.metrics.setdefault(key, []).append(value)


class BenchRunner:
    """Runs workloads and turns them into measurements with intervals."""

    def __init__(
        self,
        executor: Executor | None = None,
        metric_extractor: MetricExtractor | None = None,
    ) -> None:
        self.executor = executor or LocalExecutor()
        self.metric_extractor = metric_extractor

    def _run_once(self, spec: WorkloadSpec) -> tuple[RunResult, dict[str, float]]:
        result = self.executor.run(spec.command, env=spec.env, cwd=spec.cwd, timeout=spec.timeout_s)
        extracted: dict[str, float] = {}
        if result.ok and self.metric_extractor is not None:
            try:
                extracted = self.metric_extractor(result) or {}
            except Exception as exc:  # an adapter's parser must never kill a run
                extracted = {}
                result = result.model_copy(
                    update={"stderr": f"{result.stderr}\n[metric extraction failed: {exc}]"}
                )
        return result, extracted

    def run(self, spec: WorkloadSpec, repetitions: int | None = None) -> BenchResult:
        """Warm up, then measure `repetitions` times. Warmup samples are discarded."""
        out = BenchResult()

        for _ in range(max(0, spec.warmup_iterations)):
            result, _ = self._run_once(spec)
            out.warmup_discarded += 1
            if not result.ok:
                out.failures.append(_describe_failure("warmup", result))
                out.last_failure = result

        n = repetitions if repetitions is not None else spec.repetitions
        for i in range(max(1, n)):
            result, extracted = self._run_once(spec)
            if not result.ok:
                out.failures.append(_describe_failure(f"rep {i}", result))
                out.last_failure = result
                continue
            out.add(result, extracted)

        return out

    def measure(
        self,
        spec: WorkloadSpec,
        repetitions: int | None = None,
        profile_id: str | None = None,
        clock_samples: list[float] | None = None,
    ) -> WorkloadMeasurement:
        """Run a workload and produce the measurement artifact for it."""
        raw = self.run(spec, repetitions=repetitions)
        if not raw.wall_times:
            detail = "; ".join(raw.failures) or "no successful runs"
            # A workload that never ran is an enablement finding: classify the
            # capability gap so the error says what to do next.
            if raw.last_failure is not None:
                from xe_forge.orbit.enablement import diagnose

                gaps = diagnose(
                    raw.last_failure.returncode,
                    raw.last_failure.stdout,
                    raw.last_failure.stderr,
                )
                if gaps:
                    diagnosis = "\n".join(gap.format() for gap in gaps)
                    raise RuntimeError(
                        f"workload produced no usable samples: {detail}\n"
                        f"enablement diagnosis:\n{diagnosis}"
                    )
            raise RuntimeError(f"workload produced no usable samples: {detail}")

        wall = stats.estimate(raw.wall_times, unit="s")
        metrics: dict[str, MetricEstimate] = {
            name: stats.estimate(values, unit="") for name, values in raw.metrics.items() if values
        }

        available = ["wall_time", *sorted(metrics)]
        return WorkloadMeasurement(
            wall_time=wall,
            throughput=metrics.get("throughput"),
            ttft_ms=metrics.get("ttft_ms"),
            tpot_ms=metrics.get("tpot_ms"),
            minimum_detectable_effect=stats.minimum_detectable_effect(raw.wall_times),
            frequency_locked=False,
            clock_samples=clock_samples or [],
            profile_id=profile_id,
            metrics_available=available,
        )

    def interleaved(
        self,
        baseline: WorkloadSpec,
        candidate: WorkloadSpec,
        repetitions: int = 5,
        metric: str = "wall_time",
    ) -> tuple[list[float], list[float]]:
        """Interleave baseline and candidate runs, returning paired sample lists.

        Both arms are warmed up before the first pair. The ordering is ABBA rather
        than strict ABAB: counterbalancing cancels first-position effects and linear
        drift, which is what lets the null test come back INCONCLUSIVE on a workload
        compared against itself.
        """
        for spec in (baseline, candidate):
            for _ in range(max(0, spec.warmup_iterations)):
                self._run_once(spec)

        base_samples: list[float] = []
        cand_samples: list[float] = []

        for index in range(max(1, repetitions)):
            pair = (
                ((baseline, base_samples), (candidate, cand_samples))
                if index % 2 == 0
                else ((candidate, cand_samples), (baseline, base_samples))
            )
            for spec, sink in pair:
                result, extracted = self._run_once(spec)
                if not result.ok:
                    continue
                if metric == "wall_time":
                    sink.append(result.duration_s)
                elif metric in extracted:
                    sink.append(extracted[metric])

        return base_samples, cand_samples


def _describe_failure(label: str, result: RunResult) -> str:
    if result.timed_out:
        return f"{label}: timed out after {result.duration_s:.1f}s"
    tail = (result.stderr or "").strip().splitlines()
    hint = tail[-1] if tail else "no stderr"
    return f"{label}: exit {result.returncode} ({hint})"
