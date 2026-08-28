"""
The full loop and its stop conditions (plan §24 PR 13).

Most of the value in this orchestration is in refusing to continue, so most of these
tests assert that it stopped for the right reason. A pipeline that always runs to
completion is not measuring anything; it is just spending budget.
"""

from __future__ import annotations

import json

import pytest

from xe_forge.orbit.artifacts import EVENTS, KERNEL_CATALOG, RunStore
from xe_forge.orbit.pipeline import run_pipeline
from xe_forge.orbit.profiling.trace import ingest_trace_file


@pytest.fixture
def traced_run(tmp_path, decode_trace_path):
    """A run with a GPU-bound trace already ingested."""
    store = RunStore.create(base=tmp_path / ".orbit")
    store.save(EVENTS, ingest_trace_file(decode_trace_path))
    return store


class TestStopConditions:
    def test_a_run_with_no_trace_stops_immediately(self, tmp_path):
        store = RunStore.create(base=tmp_path / ".orbit")
        result = run_pipeline(store)
        assert result.stages[0].stopped
        assert "xe-orbit trace" in result.stop_reason

    def test_host_bound_workload_stops_at_the_gate(self, tmp_path, decode_trace_path):
        """Optimizing a kernel in a host-bound workload is the easiest way to waste a week."""
        from xe_forge.orbit.analysis.catalog import build_catalog

        store = RunStore.create(base=tmp_path / ".orbit")
        events = ingest_trace_file(decode_trace_path)
        store.save(EVENTS, events)
        store.save(
            KERNEL_CATALOG,
            build_catalog(events, run_id=store.run_id, gpu_busy_percent=11.0),
        )

        result = run_pipeline(store)
        assert not result.completed
        assert "host" in result.stop_reason.lower()

    def test_opaque_kernel_routes_to_a_non_source_action(self, traced_run):
        """The top kernel is a oneDNN GEMM: verified, but with no source to optimize."""
        result = run_pipeline(traced_run)
        assert "opaque" in result.stop_reason
        emit = next(s for s in result.stages if s.name == "emit")
        assert "region fusion" in emit.detail

    def test_unverifiable_bundle_stops_before_optimization(self, traced_run):
        """Without launch records the specialization cannot be confirmed (§12.10)."""
        result = run_pipeline(traced_run, kernel_id="k1")
        assert "could not be verified" in result.stop_reason
        assert not any(s.name == "emit" for s in result.stages)

    def test_targeting_an_unknown_kernel_stops_cleanly(self, traced_run):
        result = run_pipeline(traced_run, kernel_id="k999")
        assert not result.completed
        assert "skipped" in result.stop_reason or "actionable" in result.stop_reason


class TestHonestReporting:
    def test_e4_does_not_claim_identity_was_proven(self, traced_run):
        """Passing because every check was skipped is not the same as being proven."""
        result = run_pipeline(traced_run)
        bundle_stage = next(s for s in result.stages if s.name == "bundle test")
        assert "nothing was proven" in bundle_stage.detail
        assert "identity established" not in bundle_stage.detail

    def test_gate_reason_is_carried_into_the_report(self, traced_run):
        result = run_pipeline(traced_run)
        gate = next(s for s in result.stages if s.name == "gate")
        assert "GPU busy" in gate.detail

    def test_selection_shows_the_amdahl_arithmetic(self, traced_run):
        result = run_pipeline(traced_run)
        select = next(s for s in result.stages if s.name == "select")
        assert "Amdahl ceiling" in select.detail
        assert "% GPU" in select.detail

    def test_a_deliberate_stop_is_not_a_failure(self, traced_run):
        """NO_ACTION is a first-class result, not an error (§7.6)."""
        result = run_pipeline(traced_run)
        stopped = [s for s in result.stages if s.stopped]
        assert len(stopped) == 1
        # Everything before the stop succeeded.
        assert all(s.ok for s in result.stages)

    def test_report_renders_every_stage(self, traced_run):
        rendered = run_pipeline(traced_run).format()
        for stage in ("trace", "kernels", "gate", "select", "extract"):
            assert stage in rendered
        assert "STOPPED" in rendered


class TestArtifacts:
    def test_pipeline_persists_the_catalog_and_bundle(self, traced_run):
        run_pipeline(traced_run)
        assert traced_run.exists(KERNEL_CATALOG)
        assert traced_run.exists("bundles/k0/manifest.json")

    def test_bundle_records_its_verification_state(self, traced_run):
        from xe_forge.orbit.models import KernelBundle

        run_pipeline(traced_run)
        bundle = traced_run.load("bundles/k0/manifest.json", KernelBundle)
        # E4: verified in the sense that nothing contradicted it, with the identity
        # checks recorded as not-applicable rather than as passes.
        assert bundle.verification.mutation_detected is None
        assert bundle.verification.isolated_import is None

    def test_results_are_json_serializable(self, traced_run):
        result = run_pipeline(traced_run)
        payload = {
            "run_id": result.run_id,
            "kernel_id": result.kernel_id,
            "stop_reason": result.stop_reason,
            "stages": [s.__dict__ for s in result.stages],
        }
        assert json.loads(json.dumps(payload))["kernel_id"] == "k0"
