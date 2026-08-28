"""
The correctness ladder and matrix acceptance (plan §19, §14.3, §17).

The ladder's job is to stop a wrong candidate from ever being timed. The matrix's job
is to stop a trade from being reported as an improvement. Both are tested here mostly
through their refusals, because that is where the value is.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.compare import Gate, decide_matrix, run_ladder
from xe_forge.orbit.models import Decision, ServingProfile, WorkloadMatrix
from xe_forge.orbit.optimize import Budget, OptimizeError, optimize_kernel_dir, resolve_candidate

STABLE = [1.000, 1.004, 0.997, 1.002, 0.999, 1.001, 0.998, 1.003]


def _faster(samples, factor=0.80):
    return [s * factor for s in samples]


def _slower(samples, factor=1.20):
    return [s * factor for s in samples]


class TestLadderOrdering:
    def test_a_passing_candidate_reaches_l5(self):
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=True,
            kernel_samples=(STABLE, _faster(STABLE)),
            e2e_samples=(STABLE, _faster(STABLE, 0.9)),
            reprofile_changed=True,
            reprofile_detail="new kernel present, old absent",
        )
        assert ladder.passed
        assert {r.gate for r in ladder.results} >= {
            Gate.L0,
            Gate.L0B,
            Gate.L1,
            Gate.L2,
            Gate.L4,
            Gate.L5,
        }

    def test_unverified_extraction_blocks_at_l0b(self):
        """An unverified bundle is never optimized, let alone timed."""
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=False,
            correctness_ok=True,
            kernel_samples=(STABLE, _faster(STABLE)),
        )
        assert not ladder.passed
        assert ladder.failed_at is Gate.L0B

    def test_incorrect_candidate_never_reaches_timing(self):
        """The whole point of ordering: a wrong kernel must not produce a timing number."""
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=False,
            correctness_detail="fails tightened tolerance",
            kernel_samples=(STABLE, _faster(STABLE, 0.25)),
            e2e_samples=(STABLE, _faster(STABLE, 0.5)),
        )
        assert ladder.failed_at is Gate.L1
        assert not any(r.gate is Gate.L4 for r in ladder.results)

    def test_microbench_win_that_loses_e2e_is_rejected_at_l4(self):
        """The known-bad case: faster kernel, slower workload."""
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=True,
            kernel_samples=(STABLE, _faster(STABLE, 0.4)),
            e2e_samples=(STABLE, _slower(STABLE, 1.08)),
            reprofile_changed=True,
        )
        assert ladder.failed_at is Gate.L4

    def test_missing_reprofile_blocks_at_l5(self):
        """L5 is not skippable: without it the gain cannot be attributed to this change."""
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=True,
            kernel_samples=(STABLE, _faster(STABLE)),
            e2e_samples=(STABLE, _faster(STABLE, 0.9)),
            reprofile_changed=None,
        )
        assert ladder.failed_at is Gate.L5
        assert "attributed" in ladder.results[-1].detail

    def test_absent_model_gate_is_skipped_not_failed(self):
        ladder = run_ladder(
            "k0",
            build_ok=True,
            extraction_verified=True,
            correctness_ok=True,
            kernel_samples=(STABLE, _faster(STABLE)),
            model_gate_ok=None,
            e2e_samples=(STABLE, _faster(STABLE, 0.9)),
            reprofile_changed=True,
        )
        l3 = next(r for r in ladder.results if r.gate is Gate.L3)
        assert l3.skipped
        assert ladder.passed

    def test_missing_correctness_result_is_not_assumed_pass(self):
        ladder = run_ladder("k0", build_ok=True, extraction_verified=True, correctness_ok=None)
        assert ladder.failed_at is Gate.L1
        assert "refusing to assume" in ladder.results[-1].detail

    def test_build_failure_short_circuits_everything(self):
        ladder = run_ladder("k0", build_ok=False, extraction_verified=True)
        assert ladder.failed_at is Gate.L0
        assert len(ladder.results) == 1


class TestMatrixAcceptance:
    @pytest.fixture
    def matrix(self) -> WorkloadMatrix:
        return WorkloadMatrix(
            profiles=[
                ServingProfile(id="decode", weight=0.6),
                ServingProfile(id="prefill", weight=0.4),
            ]
        )

    def test_win_everywhere_is_accepted(self, matrix):
        samples = {
            "decode": (STABLE, _faster(STABLE, 0.85)),
            "prefill": (STABLE, _faster(STABLE, 0.90)),
        }
        decision = decide_matrix(matrix, samples)
        assert decision.decision is Decision.ACCEPT
        assert decision.weighted_improvement > 0

    def test_a_trade_is_rejected_even_when_weighted_positive(self, matrix):
        """Wins decode 20%, loses prefill 8%: a trade, not an improvement (§14.3)."""
        samples = {
            "decode": (STABLE, _faster(STABLE, 0.80)),
            "prefill": (STABLE, _slower(STABLE, 1.08)),
        }
        decision = decide_matrix(matrix, samples, regression_threshold_percent=2.0)
        assert decision.decision is Decision.REJECT
        assert decision.regressions
        assert "trade" in decision.reason

    def test_regression_within_threshold_is_tolerated(self, matrix):
        samples = {
            "decode": (STABLE, _faster(STABLE, 0.80)),
            "prefill": (STABLE, _slower(STABLE, 1.01)),
        }
        decision = decide_matrix(matrix, samples, regression_threshold_percent=2.0)
        assert decision.decision is Decision.ACCEPT

    def test_a_missing_profile_invalidates_rather_than_averaging_around_it(self, matrix):
        samples = {"decode": (STABLE, _faster(STABLE))}
        decision = decide_matrix(matrix, samples)
        assert decision.decision is Decision.INVALID
        assert "prefill" in decision.reason

    def test_all_inconclusive_is_inconclusive(self, matrix):
        samples = {"decode": (STABLE, list(STABLE)), "prefill": (STABLE, list(STABLE))}
        decision = decide_matrix(matrix, samples)
        assert decision.decision is Decision.INCONCLUSIVE

    def test_report_is_per_profile_never_a_single_number(self, matrix):
        samples = {
            "decode": (STABLE, _faster(STABLE, 0.85)),
            "prefill": (STABLE, _faster(STABLE, 0.95)),
        }
        rendered = decide_matrix(matrix, samples).format()
        assert "decode" in rendered
        assert "prefill" in rendered
        assert "95% CI" in rendered

    def test_weights_are_normalized(self):
        matrix = WorkloadMatrix(
            profiles=[ServingProfile(id="a", weight=3.0), ServingProfile(id="b", weight=1.0)]
        )
        weights = matrix.normalized_weights()
        assert weights["a"] == pytest.approx(0.75)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_zero_weights_are_rejected(self):
        matrix = WorkloadMatrix(profiles=[ServingProfile(id="a", weight=0.0)])
        with pytest.raises(ValueError):
            matrix.normalized_weights()


class TestOptimizeWrapper:
    def _candidate(self, tmp_path, with_reference=True, stub=True):
        directory = tmp_path / "cand"
        directory.mkdir()
        (directory / "kernel.py").write_text("def optimized_kernel(x):\n    return x\n")
        (directory / "spec.yaml").write_text("inputs: {}\n")
        if with_reference:
            body = "raise NotImplementedError('stub')\n" if stub else "def ref(x):\n    return x\n"
            (directory / "kernel_pytorch.py").write_text(body)
        return directory

    def test_resolves_the_conventional_layout(self, tmp_path):
        resolved = resolve_candidate(self._candidate(tmp_path))
        assert resolved["kernel"].name == "kernel.py"
        assert resolved["spec"].name == "spec.yaml"
        assert "reference" in resolved

    def test_missing_kernel_names_the_command_that_produces_it(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(OptimizeError, match="xe-orbit emit"):
            resolve_candidate(empty)

    def test_missing_spec_is_a_clear_error(self, tmp_path):
        directory = tmp_path / "cand"
        directory.mkdir()
        (directory / "kernel.py").write_text("x = 1\n")
        with pytest.raises(OptimizeError, match="spec"):
            resolve_candidate(directory)

    def test_dry_run_flags_a_stub_reference(self, tmp_path):
        """A stub reference means the correctness gate cannot compare anything."""
        outcome = optimize_kernel_dir(self._candidate(tmp_path), dry_run=True)
        assert outcome.success
        assert any("stub" in note for note in outcome.notes)

    def test_dry_run_no_longer_flags_the_weighted_objective(self, tmp_path):
        # §9.1 landed: 'weighted_latency' now maps onto the pipeline's weighted
        # objective instead of degrading to a single variant, so the old
        # "not implemented yet" note must be gone.
        outcome = optimize_kernel_dir(
            self._candidate(tmp_path), objective="weighted_latency", dry_run=True
        )
        assert not any("does not implement" in note for note in outcome.notes)

    def test_missing_reference_is_reported(self, tmp_path):
        outcome = optimize_kernel_dir(self._candidate(tmp_path, with_reference=False), dry_run=True)
        assert any("kernel_pytorch.py" in note for note in outcome.notes)

    def test_budget_defaults_are_explicit(self):
        assert Budget().trials == 10
