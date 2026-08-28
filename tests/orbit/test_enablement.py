"""The enablement ladder's v0.1 slice: diagnosis and the runnable gate (plan §5.6).

Classification is deterministic on purpose — a failed launch's output is trace
parsing, and §3 forbids an LLM where a deterministic answer exists. The fixtures
below include the two literal strings from the measured Wildcat Lake dead end that
motivated the ladder.
"""

from __future__ import annotations

from xe_forge.orbit.enablement import (
    IMPLEMENTED_RUNGS,
    CapabilityGap,
    Rung,
    diagnose,
    runnable_gate,
)


class TestDiagnosis:
    def test_success_yields_no_gaps(self):
        assert diagnose(0, "all good", "") == []

    def test_wildcat_lake_codegen_failure_maps_to_serve_flag(self):
        # The measured case §5.6 names: GRAPH_CAPTURE unavailable on this device.
        stderr = (
            "torch._inductor.exc.InductorError: No valid triton configs\n"
            "Internal Triton ZEBIN codegen error"
        )
        gaps = diagnose(1, "", stderr)
        assert gaps[0].kind == "backend_codegen"
        assert gaps[0].rung is Rung.SERVE_FLAG
        assert not gaps[0].deferred
        assert "eager" in gaps[0].suggestion

    def test_oom_maps_to_serve_flag_with_footprint_advice(self):
        gaps = diagnose(1, "", "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY on allocation")
        assert gaps[0].kind == "oom"
        assert "KV-cache" in gaps[0].suggestion or "batch" in gaps[0].suggestion

    def test_missing_device_is_the_floor_of_every_ladder(self):
        gaps = diagnose(1, "", "RuntimeError: xpu is not available")
        assert gaps[0].kind == "missing_device"
        assert gaps[0].rung is Rung.SERVE_FLAG

    def test_missing_op_points_at_the_operator_override(self):
        stderr = (
            "NotImplementedError: could not run 'aten::foo' with arguments from the 'XPU' backend"
        )
        gaps = diagnose(1, "", stderr)
        assert gaps[0].kind == "missing_op"
        assert gaps[0].rung is Rung.SOURCE_PATCH
        assert "override" in gaps[0].suggestion

    def test_missing_package_points_at_the_scoped_runtime_climb(self):
        gaps = diagnose(1, "", "ModuleNotFoundError: No module named 'vllm'")
        assert gaps[0].kind == "missing_package"
        assert gaps[0].rung is Rung.SCOPED_RUNTIME
        # Rung 3 is built now: the gap is actionable and its suggestion points at
        # the climb instead of apologising for an unimplemented rung.
        assert not gaps[0].deferred
        assert "actionable now" in gaps[0].format()
        assert "climb_missing_package" in gaps[0].suggestion
        # The old honesty still holds one rung up: 4 and 5 remain deferred, and a
        # gap that lands there must still say so.
        assert Rung.SOURCE_LOCALIZE not in IMPLEMENTED_RUNGS
        assert Rung.COMPILED_BUILD not in IMPLEMENTED_RUNGS

    def test_config_error_blames_the_command_not_the_workload(self):
        gaps = diagnose(2, "", "error: unrecognized arguments: --frobnicate")
        assert gaps[0].kind == "config"
        assert "flag" in gaps[0].suggestion

    def test_unknown_failure_is_a_finding_not_a_guess(self):
        gaps = diagnose(137, "", "Killed")
        assert len(gaps) == 1
        assert gaps[0].kind == "unknown"
        assert gaps[0].rung is Rung.DIAGNOSE
        assert "guess" in gaps[0].suggestion

    def test_unknown_with_no_output_still_names_the_exit_code(self):
        gaps = diagnose(139, "", "")
        assert "exit code 139" in gaps[0].evidence

    def test_multiple_gaps_are_all_reported(self):
        stderr = "ModuleNotFoundError: No module named 'triton'\nRuntimeError: xpu is not available"
        kinds = {g.kind for g in diagnose(1, "", stderr)}
        assert kinds == {"missing_package", "missing_device"}

    def test_evidence_is_the_matched_line_verbatim(self):
        stderr = "prefix noise\nRuntimeError: xpu is not available\nsuffix"
        gaps = diagnose(1, "", stderr)
        assert gaps[0].evidence == "RuntimeError: xpu is not available"


class TestLadderShape:
    def test_implemented_rungs_are_the_bottom_of_the_ladder(self):
        assert IMPLEMENTED_RUNGS == {
            Rung.DIAGNOSE,
            Rung.SERVE_FLAG,
            Rung.SOURCE_PATCH,
            Rung.SCOPED_RUNTIME,
        }

    def test_deferred_is_derived_from_the_rung_not_declared(self):
        gap = CapabilityGap(kind="x", evidence="e", rung=Rung.COMPILED_BUILD, suggestion="s")
        assert gap.deferred


class TestRunnableGate:
    """§5.6: a fix earns KEEP only when the workload boots AND re-passes the eval."""

    def test_boot_failure_is_not_kept_and_carries_the_diagnosis(self):
        result = runnable_gate(lambda: (1, "", "RuntimeError: xpu is not available"))
        assert not result.booted and not result.kept
        assert result.gaps and result.gaps[0].kind == "missing_device"

    def test_boot_without_an_eval_is_not_a_keep(self):
        result = runnable_gate(lambda: (0, "serving", ""))
        assert result.booted and not result.evaluated
        assert not result.kept
        assert "boot alone does not" in result.reason

    def test_boot_plus_failing_eval_is_not_a_keep(self):
        result = runnable_gate(lambda: (0, "serving", ""), quality=lambda: False)
        assert result.booted and result.evaluated and result.eval_passed is False
        assert not result.kept
        assert "changed what the workload computes" in result.reason

    def test_boot_plus_passing_eval_is_the_only_keep(self):
        result = runnable_gate(lambda: (0, "serving", ""), quality=lambda: True)
        assert result.kept and result.eval_passed


class TestWiredIntoTheBenchRunner:
    """Diagnosis must fire where failures actually surface, or it is a module
    nothing calls — the same consumed-by-nothing failure §15's fixture had."""

    def test_a_workload_that_never_ran_gets_a_diagnosis(self):
        import sys

        import pytest

        from xe_forge.orbit.bench.core import BenchRunner
        from xe_forge.orbit.models import WorkloadSpec

        spec = WorkloadSpec(
            command=[sys.executable, "-c", "import definitely_not_a_module"],
            repetitions=1,
            warmup_iterations=0,
        )
        with pytest.raises(RuntimeError) as excinfo:
            BenchRunner().measure(spec)
        message = str(excinfo.value)
        assert "enablement diagnosis" in message
        assert "missing_package" in message

    def test_a_healthy_workload_is_untouched(self):
        import sys

        from xe_forge.orbit.bench.core import BenchRunner
        from xe_forge.orbit.models import WorkloadSpec

        spec = WorkloadSpec(
            command=[sys.executable, "-c", "pass"],
            repetitions=1,
            warmup_iterations=0,
        )
        measurement = BenchRunner().measure(spec)
        assert measurement.wall_time.n == 1
