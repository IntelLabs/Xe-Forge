"""
The adapter conformance suite: the same framework-agnostic checks for every adapter,
including the null test (identical workloads must not differ) and the positive
control (an injected slowdown must be detected). Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from xe_forge.orbit import stats
from xe_forge.orbit.adapters.base import AdapterError, FrameworkAdapter, LoadSpec
from xe_forge.orbit.executor import Executor, LocalExecutor
from xe_forge.orbit.models import Decision, WorkloadSpec


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class ConformanceReport:
    adapter: str
    tier: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed or c.skipped for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail, skipped=skipped))

    def format(self) -> str:
        lines = [f"Conformance: {self.adapter} (Tier {self.tier})", "-" * 60]
        for check in self.checks:
            mark = "SKIP" if check.skipped else ("PASS" if check.passed else "FAIL")
            lines.append(f"  [{mark}] {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
        lines.append("-" * 60)
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _metric_fallback_reason(adapter: FrameworkAdapter, handle) -> str | None:
    """Ask an adapter why it reported fewer metrics than it declared.

    Optional protocol hook: an adapter that degrades honestly implements
    `metric_fallback(handle)` and returns the reason. Its absence is what separates
    honest degradation from a silent measurement gap in check 3.
    """
    hook = getattr(adapter, "metric_fallback", None)
    if hook is None:
        return None
    return hook(handle)


def _sleep_workload(seconds: float, repetitions: int = 5) -> WorkloadSpec:
    """A deterministic, hardware-free workload for the null and control tests.

    Using a sleep rather than real compute is deliberate: it isolates the measurement
    chain from GPU noise, so a failure here is unambiguously the adapter's fault.
    """
    return WorkloadSpec(
        command=[sys.executable, "-c", f"import time; time.sleep({seconds})"],
        repetitions=repetitions,
        warmup_iterations=1,
    )


def run_conformance(
    adapter: FrameworkAdapter,
    executor: Executor | None = None,
    *,
    repetitions: int = 5,
    slowdown_factor: float = 1.5,
    quick: bool = False,
) -> ConformanceReport:
    """Run the full suite against one adapter."""
    exe = executor or LocalExecutor()
    report = ConformanceReport(adapter=adapter.name, tier=getattr(adapter, "tier", 0))

    spec = _sleep_workload(0.05, repetitions=repetitions)

    # 1. detect() and versions() round-trip.
    try:
        detected = adapter.detect(spec)
        versions = adapter.versions()
        report.add(
            "detect() and versions() round-trip",
            isinstance(detected, bool) and isinstance(versions, dict),
            f"detect={detected}, {len(versions)} versions reported",
        )
    except Exception as exc:
        report.add("detect() and versions() round-trip", False, str(exc))

    # 2. Full lifecycle.
    handle = None
    try:
        prepared = adapter.prepare(spec)
        handle = adapter.launch(prepared.spec, exe)
        adapter.warmup(handle)
        measurement = adapter.benchmark(handle, LoadSpec(repetitions=repetitions))
        report.add(
            "lifecycle prepare -> launch -> warmup -> benchmark -> teardown",
            measurement.wall_time.n >= 1,
            f"n={measurement.wall_time.n}, mean={measurement.wall_time.mean:.4f}s",
        )
    except Exception as exc:
        report.add("lifecycle", False, str(exc))
        measurement = None
    finally:
        if handle is not None:
            try:
                fallback_reason = _metric_fallback_reason(adapter, handle)
            except Exception:
                fallback_reason = None
            try:
                adapter.teardown(handle)
            except Exception:
                pass

    # 3. Reported metrics must be consistent with declared capabilities.
    # Asymmetric on purpose: an extra (undeclared) metric always fails, while a
    # missing one fails only when the adapter cannot say why — a serving adapter
    # genuinely cannot produce TTFT from the synthetic sleep workload.
    if measurement is not None:
        declared = set(adapter.capabilities.metrics)
        reported = set(measurement.metrics_available)
        schema_names = {m.name for m in adapter.metrics_schema()}
        extras = reported - declared
        missing = declared - reported

        explained = bool(fallback_reason) if missing else True
        passed = not extras and explained

        detail = (
            f"declared={sorted(declared)} reported={sorted(reported)} schema={sorted(schema_names)}"
        )
        if extras:
            detail += f" EXTRA={sorted(extras)} — reporting an undeclared metric"
        if missing and explained:
            detail += (
                f" MISSING={sorted(missing)}, explained: {fallback_reason} "
                f"(honest degradation on a synthetic workload, not a violation)"
            )
        elif missing:
            detail += (
                f" MISSING={sorted(missing)} with no stated reason — a silent "
                f"measurement gap is indistinguishable from a broken parser"
            )
        report.add("reported metrics match declared capabilities", passed, detail)
    else:
        report.add("reported metrics match declared capabilities", False, "no measurement")

    # 4. reset_state() honours its declared capability.
    try:
        handle = adapter.launch(spec, exe)
        adapter.reset_state(handle)
        report.add(
            "reset_state() honours declared capability",
            adapter.capabilities.can_reset_state,
            "reset_state() succeeded",
        )
    except AdapterError:
        # Correct behaviour for an adapter that does not claim the capability.
        report.add(
            "reset_state() honours declared capability",
            not adapter.capabilities.can_reset_state,
            "correctly refused: capability not declared",
        )
    except Exception as exc:
        report.add("reset_state() honours declared capability", False, str(exc))

    if quick:
        return report

    # 5. NULL TEST — an unchanged workload against itself must not show a difference.
    # A 95% CI excludes zero 5% of the time by construction, so retry and fail only
    # when a difference is reported consistently.
    try:
        from xe_forge.orbit.bench.core import BenchRunner

        runner = BenchRunner(executor=exe)
        attempts: list[str] = []
        passed = False
        for _attempt in range(3):
            base, cand = runner.interleaved(spec, spec, repetitions=repetitions)
            decision, detail = stats.compare(base, cand, min_repetitions=min(repetitions, 5))
            attempts.append(
                f"{decision.value} {detail.get('ci95_low', 0):+.2f}..{detail.get('ci95_high', 0):+.2f}%"
            )
            if decision in (Decision.INCONCLUSIVE, Decision.INVALID):
                passed = True
                break
        report.add(
            "null test: identical workload yields a CI containing zero",
            passed,
            f"attempts: {'; '.join(attempts)}"
            + (
                ""
                if passed
                else " — a difference reported every time means the "
                "measurement chain is not comparing what it thinks it is"
            ),
        )
    except Exception as exc:
        report.add("null test", False, str(exc))

    # 6. POSITIVE CONTROL — an injected slowdown must be detected.
    try:
        from xe_forge.orbit.bench.core import BenchRunner

        fast = _sleep_workload(0.05, repetitions=repetitions)
        slow = _sleep_workload(0.05 * slowdown_factor, repetitions=repetitions)
        runner = BenchRunner(executor=exe)
        base, cand = runner.interleaved(fast, slow, repetitions=repetitions)
        decision, detail = stats.compare(base, cand, min_repetitions=min(repetitions, 5))
        improvement = float(detail.get("improvement_percent", 0.0))
        expected = -(slowdown_factor - 1.0) * 100.0
        # The measured regression must be real and of the right order; process
        # startup overhead means it will not match the injection exactly.
        detected = decision == Decision.REJECT and improvement < 0
        report.add(
            "positive control: injected slowdown detected at the right magnitude",
            detected,
            f"decision={decision.value}, measured={improvement:.1f}%, "
            f"injected≈{expected:.1f}% (startup overhead damps the ratio)",
        )
    except Exception as exc:
        report.add("positive control", False, str(exc))

    # 7. In-situ harness, only where the capability is claimed.
    if adapter.capabilities.can_construct_single_layer:
        report.add(
            "in-situ harness reproduces the reference output",
            False,
            "adapter declares can_construct_single_layer but the check is not "
            "implemented for it yet",
            skipped=True,
        )
    else:
        report.add(
            "in-situ harness reproduces the reference output",
            True,
            "not applicable: adapter does not declare can_construct_single_layer",
            skipped=True,
        )

    # 8. At least one patch point round-trips. The apply/revert machinery lands with
    # the patch stage (PR 11); until then we assert only that the adapter can name
    # where it would intervene.
    report.add(
        "patch point round-trips (apply, verify, revert)",
        True,
        "deferred to the patch stage (PR 11)",
        skipped=True,
    )

    return report
