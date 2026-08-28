"""
The conformance suite's own rules (plan §10.4, §10.7).

Check 3 is asymmetric on purpose, and that asymmetry is the whole design:

* Reporting an *undeclared* metric always fails. The decision layer reads the
  capability set to choose which actions exist, so a set that overstates the adapter
  is a lie with consequences downstream.
* Reporting *fewer* metrics than declared fails only when the adapter cannot say why.
  The suite benchmarks a synthetic sleep workload, so a serving adapter genuinely
  cannot produce TTFT from it; refusing to invent one is correct behaviour. What §10.4
  forbids is degrading *silently*.

Treating the two symmetrically made this check unpassable for every Tier 1 serving
adapter, which would have made conformance useless as a CI gate for exactly the
adapters it matters most for.
"""

from __future__ import annotations

import sys

import pytest

from xe_forge.orbit.adapters.base import BaseAdapter, Handle, LoadSpec
from xe_forge.orbit.adapters.conformance import run_conformance
from xe_forge.orbit.bench.core import BenchRunner
from xe_forge.orbit.executor import LocalExecutor
from xe_forge.orbit.models import FrameworkCapabilities, WorkloadSpec


class _OverDeclaringAdapter(BaseAdapter):
    """Declares serving metrics and degrades without ever saying why."""

    name = "silent_degrader"
    tier = 1
    capabilities = FrameworkCapabilities(metrics={"wall_time", "ttft", "throughput"})

    def detect(self, spec: WorkloadSpec) -> bool:
        return False

    def benchmark(self, handle: Handle, load: LoadSpec):
        runner = BenchRunner(executor=handle.state.get("executor") or LocalExecutor())
        measurement = runner.measure(handle.spec, repetitions=load.repetitions)
        measurement.metrics_available = ["wall_time"]
        return measurement


class _ExplainingAdapter(_OverDeclaringAdapter):
    """Same degradation, but states the reason."""

    name = "honest_degrader"

    def benchmark(self, handle: Handle, load: LoadSpec):
        measurement = super().benchmark(handle, load)
        handle.state["metric_fallback"] = (
            "no serving metrics in this workload's output; reported wall_time only"
        )
        return measurement

    def metric_fallback(self, handle: Handle) -> str | None:
        return handle.state.get("metric_fallback")


class _FabricatingAdapter(BaseAdapter):
    """Reports a metric it never declared."""

    name = "fabricator"
    tier = 1
    capabilities = FrameworkCapabilities(metrics={"wall_time"})

    def detect(self, spec: WorkloadSpec) -> bool:
        return False

    def benchmark(self, handle: Handle, load: LoadSpec):
        runner = BenchRunner(executor=handle.state.get("executor") or LocalExecutor())
        measurement = runner.measure(handle.spec, repetitions=load.repetitions)
        measurement.metrics_available = ["wall_time", "ttft"]
        return measurement


def _metric_check(adapter):
    report = run_conformance(adapter, repetitions=5, quick=True)
    return next(
        c for c in report.checks if c.name == "reported metrics match declared capabilities"
    )


class TestMetricDeclarationRule:
    def test_silent_degradation_fails(self):
        """A missing metric with no stated reason is indistinguishable from a broken parser."""
        check = _metric_check(_OverDeclaringAdapter())
        assert not check.passed
        assert "no stated reason" in check.detail

    def test_explained_degradation_passes(self):
        """Refusing to invent a metric the workload cannot produce is correct behaviour."""
        check = _metric_check(_ExplainingAdapter())
        assert check.passed
        assert "explained" in check.detail

    def test_reporting_an_undeclared_metric_always_fails(self):
        """Even with an explanation, an extra metric overstates the adapter."""
        check = _metric_check(_FabricatingAdapter())
        assert not check.passed
        assert "EXTRA" in check.detail

    def test_an_adapter_reporting_exactly_what_it_declares_passes(self):
        from xe_forge.orbit.adapters import GenericTorchAdapter

        check = _metric_check(GenericTorchAdapter())
        assert check.passed


class TestNullTestRetry:
    def test_null_test_reports_every_attempt(self):
        """The retry exists because a 95% CI excludes zero 5% of the time by construction."""
        from xe_forge.orbit.adapters import GenericTorchAdapter

        report = run_conformance(GenericTorchAdapter(), repetitions=5)
        null = next(c for c in report.checks if c.name.startswith("null test"))
        assert "attempts:" in null.detail
        assert null.passed


@pytest.fixture(autouse=True)
def _fast_workload(monkeypatch):
    """Keep these structural checks quick; they are not measuring anything real."""
    import xe_forge.orbit.adapters.conformance as conformance

    original = conformance._sleep_workload

    def quick(seconds: float, repetitions: int = 5) -> WorkloadSpec:
        return original(min(seconds, 0.01), repetitions=repetitions)

    monkeypatch.setattr(conformance, "_sleep_workload", quick)
    assert sys.executable
