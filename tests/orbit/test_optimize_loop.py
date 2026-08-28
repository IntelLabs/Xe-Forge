"""
The agentic optimization loop (plan §13.7).

Every test uses a stub agent: §15.3 forbids CI from calling an LLM, and a loop that can
only be tested against a live model is a loop nobody will run in CI.

The properties defended here are the ones that decide whether the loop can be trusted
with a shared tree: a wrong candidate is reverted, an unmeasurable one is reverted and
recorded as a gap rather than as evidence, and nothing unaccepted survives the run.
"""

from __future__ import annotations

import pytest

from xe_forge.orbit.novelty import NoveltyLedger
from xe_forge.orbit.optimize.harness import CheckOutcome, CheckResult
from xe_forge.orbit.optimize.loop import (
    OptimizationLoop,
    Proposal,
    TrialVerdict,
)
from xe_forge.orbit.patch.inplace import InPlacePatcher

ORIGINAL = b"BLOCK_SIZE = 1024\n"


@pytest.fixture
def target(tmp_path):
    tree = tmp_path / "site-packages" / "framework"
    tree.mkdir(parents=True)
    path = tree / "kernel.py"
    path.write_bytes(ORIGINAL)
    return path


@pytest.fixture
def patcher(tmp_path, target):
    return InPlacePatcher(journal_dir=tmp_path / "run", sandbox_roots=[target.parent])


def _loop(target, patcher, check, measure, **kw):
    return OptimizationLoop(
        target=target,
        patcher=patcher,
        check=check,
        measure=measure,
        ledger=NoveltyLedger(),
        **kw,
    )


def _proposal(title="raise BLOCK_SIZE", source=b"BLOCK_SIZE = 2048\n", **params):
    return Proposal(title=title, rationale="because", new_source=source, parameters=params)


CORRECT = CheckResult(CheckOutcome.CORRECT, accuracy=1.0, detail="matches")
WRONG = CheckResult(CheckOutcome.WRONG, accuracy=0.0, detail="does not match")
UNCHECKED = CheckResult(CheckOutcome.UNCHECKED, detail="import failed")


class TestKeepingAWinner:
    def test_a_correct_and_faster_candidate_is_kept(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__)
        result = loop.run([_proposal()])
        assert result.trials[0].verdict is TrialVerdict.KEPT
        assert result.kept
        assert result.trials[0].delta_percent == pytest.approx(20.0)

    def test_the_winner_stays_on_disk(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__)
        loop.run([_proposal()])
        assert target.read_bytes() == b"BLOCK_SIZE = 2048\n"


class TestRevertingLosers:
    def test_a_numerically_wrong_candidate_is_reverted(self, target, patcher):
        """The positive-control case: correctness gates before anything is measured."""
        loop = _loop(target, patcher, lambda: WRONG, lambda: 50.0)
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.REVERTED_WRONG
        assert target.read_bytes() == ORIGINAL

    def test_correctness_is_checked_before_paying_for_a_measurement(self, target, patcher):
        """A wrong kernel must never cost an engine load."""
        calls = []

        def measure():
            calls.append(1)
            return 50.0

        loop = _loop(target, patcher, lambda: WRONG, measure)
        loop.run([_proposal()], baseline_us=100.0)
        assert calls == [], "measured a candidate that had already failed correctness"

    def test_a_slower_candidate_is_reverted(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 130.0]).__next__)
        result = loop.run([_proposal()])
        assert result.trials[0].verdict is TrialVerdict.REVERTED_SLOWER
        assert target.read_bytes() == ORIGINAL

    def test_a_win_inside_the_noise_floor_is_not_kept(self, target, patcher):
        """Keeping these is how a stack of no-ops becomes a reported speedup (§17.6)."""
        loop = _loop(
            target,
            patcher,
            lambda: CORRECT,
            iter([100.0, 99.5]).__next__,
            min_improvement_percent=1.0,
        )
        result = loop.run([_proposal()])
        assert result.trials[0].verdict is TrialVerdict.REVERTED_SLOWER
        assert "cannot be distinguished from noise" in result.trials[0].reason

    def test_nothing_unaccepted_survives_the_run(self, target, patcher):
        loop = _loop(target, patcher, lambda: WRONG, lambda: 50.0)
        loop.run([_proposal("a", b"A\n"), _proposal("b", b"B\n")], baseline_us=100.0)
        assert target.read_bytes() == ORIGINAL
        assert patcher.outstanding == []


class TestUnprovenIsNotDisproven:
    def test_an_uncheckable_candidate_is_unproven_not_wrong(self, target, patcher):
        """A harness that could not run is a gap in the run, not evidence against it."""
        loop = _loop(target, patcher, lambda: UNCHECKED, lambda: 50.0)
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.UNPROVEN
        assert result.trials[0].verdict is not TrialVerdict.REVERTED_WRONG

    def test_an_unproven_candidate_is_still_reverted(self, target, patcher):
        """Unproven must not stay on disk, however innocent it may be."""
        loop = _loop(target, patcher, lambda: UNCHECKED, lambda: 50.0)
        loop.run([_proposal()], baseline_us=100.0)
        assert target.read_bytes() == ORIGINAL

    def test_a_correct_candidate_with_no_measurement_is_unproven(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, lambda: None)
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.UNPROVEN

    def test_a_missing_baseline_blocks_acceptance_and_says_so(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, lambda: None)
        result = loop.run([_proposal()])
        assert not result.kept
        assert any("no baseline" in n for n in result.notes)


class TestGatesBeforeWriting:
    def test_a_repeated_proposal_is_refused_by_the_novelty_ledger(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 99.0, 99.0]).__next__)
        result = loop.run([_proposal(knob="a"), _proposal(knob="a")])
        assert result.trials[1].verdict is TrialVerdict.REFUSED
        assert "already ran" in result.trials[1].reason

    def test_a_proposal_with_no_source_is_refused(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, lambda: 100.0)
        result = loop.run([Proposal(title="vague idea", rationale="", new_source=None)])
        assert result.trials[0].verdict is TrialVerdict.REFUSED
        assert target.read_bytes() == ORIGINAL

    def test_a_critic_veto_stops_the_patch_being_written(self, target, patcher):
        loop = _loop(
            target,
            patcher,
            lambda: CORRECT,
            lambda: 100.0,
            critic=lambda p, d: (False, "changes an unrelated code path"),
        )
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.REFUSED
        assert "unrelated code path" in result.trials[0].reason
        assert target.read_bytes() == ORIGINAL

    def test_the_critic_sees_a_diff_of_the_proposed_change(self, target, patcher):
        seen = {}

        def critic(proposal, diff):
            seen["diff"] = diff
            return True, "fine"

        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__, critic=critic)
        loop.run([_proposal()])
        assert "-BLOCK_SIZE = 1024" in seen["diff"]
        assert "+BLOCK_SIZE = 2048" in seen["diff"]

    def test_a_target_outside_the_sandbox_is_refused(self, tmp_path, target):
        stray = InPlacePatcher(journal_dir=tmp_path / "j", sandbox_roots=[tmp_path / "elsewhere"])
        loop = _loop(target, stray, lambda: CORRECT, lambda: 100.0)
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.REFUSED
        assert "sandbox" in result.trials[0].reason


class TestReporting:
    def test_a_run_with_no_winner_says_the_tree_is_unchanged(self, target, patcher):
        loop = _loop(target, patcher, lambda: WRONG, lambda: 50.0)
        rendered = loop.run([_proposal()], baseline_us=100.0).format()
        assert "NO CANDIDATE ACCEPTED" in rendered
        assert "tree is unchanged" in rendered

    def test_the_accepted_candidate_is_named_with_its_delta(self, target, patcher):
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__)
        rendered = loop.run([_proposal()]).format()
        assert "ACCEPTED" in rendered
        assert "+20.00%" in rendered


class TestRejectionReasonsAreTrue:
    """A right verdict with a false reason is worse than no reason.

    A live run rejected a candidate that measured 2x slower with the words
    "indistinguishable from noise" — which invites the reader to dismiss a real result.
    """

    def test_a_large_regression_is_called_a_regression(self, target, patcher):
        loop = _loop(
            target,
            patcher,
            lambda: CORRECT,
            iter([100.0, 208.0]).__next__,
            min_improvement_percent=2.0,
        )
        record = loop.run([_proposal()]).trials[0]
        assert record.verdict is TrialVerdict.REVERTED_SLOWER
        assert "clear regression" in record.reason
        assert "noise" not in record.reason

    def test_a_small_shortfall_is_still_called_noise(self, target, patcher):
        loop = _loop(
            target,
            patcher,
            lambda: CORRECT,
            iter([100.0, 99.5]).__next__,
            min_improvement_percent=2.0,
        )
        record = loop.run([_proposal()]).trials[0]
        assert "cannot be distinguished from noise" in record.reason

    def test_a_regression_exactly_at_the_floor_reads_as_a_regression(self, target, patcher):
        loop = _loop(
            target,
            patcher,
            lambda: CORRECT,
            iter([100.0, 102.0]).__next__,
            min_improvement_percent=2.0,
        )
        assert "clear regression" in loop.run([_proposal()]).trials[0].reason


class TestStatisticalDecision:
    """§17's rule: a verdict comes from an interval, not a threshold.

    The loop originally compared a point estimate against a fixed floor, so a real but
    small improvement (+0.63% measured on a live kernel) was rejected exactly as though
    it had been disproved. A threshold cannot express "we cannot tell".
    """

    def _loop(self, target, patcher, baseline, candidate, check=None):
        runs = [baseline, candidate]

        def samples():
            return runs.pop(0) if runs else candidate

        return OptimizationLoop(
            target=target,
            patcher=patcher,
            check=check or (lambda: CORRECT),
            measure=lambda: None,
            measure_samples=samples,
            ledger=NoveltyLedger(),
        )

    def test_a_clear_improvement_is_kept(self, target, patcher):
        loop = self._loop(
            target,
            patcher,
            [100.0, 101.0, 99.0, 100.5, 99.5, 100.2],
            [80.0, 80.5, 79.5, 80.2, 79.8, 80.1],
        )
        record = loop.run([_proposal()]).trials[0]
        assert record.verdict is TrialVerdict.KEPT
        assert "faster" in record.reason

    def test_the_winner_stays_applied(self, target, patcher):
        loop = self._loop(
            target,
            patcher,
            [100.0, 101.0, 99.0, 100.5, 99.5, 100.2],
            [80.0, 80.5, 79.5, 80.2, 79.8, 80.1],
        )
        result = loop.run([_proposal()])
        assert result.kept
        assert target.read_bytes() == b"BLOCK_SIZE = 2048\n"

    def test_a_clear_regression_is_reverted_as_slower(self, target, patcher):
        loop = self._loop(
            target,
            patcher,
            [100.0, 101.0, 99.0, 100.5, 99.5, 100.2],
            [130.0, 131.0, 129.0, 130.5, 129.5, 130.2],
        )
        record = loop.run([_proposal()]).trials[0]
        assert record.verdict is TrialVerdict.REVERTED_SLOWER
        assert target.read_bytes() == ORIGINAL

    def test_an_unresolvable_difference_is_unproven_not_slower(self, target, patcher):
        """The distinction a floor cannot make: 'we cannot tell' is not 'it is worse'."""
        loop = self._loop(
            target,
            patcher,
            [100.0, 105.0, 95.0, 103.0, 97.0, 101.0],
            [99.0, 104.0, 96.0, 102.0, 98.0, 100.0],
        )
        record = loop.run([_proposal()]).trials[0]
        assert record.verdict is TrialVerdict.UNPROVEN
        assert record.verdict is not TrialVerdict.REVERTED_SLOWER

    def test_an_unresolvable_candidate_is_still_reverted(self, target, patcher):
        loop = self._loop(
            target,
            patcher,
            [100.0, 105.0, 95.0, 103.0, 97.0, 101.0],
            [99.0, 104.0, 96.0, 102.0, 98.0, 100.0],
        )
        loop.run([_proposal()])
        assert target.read_bytes() == ORIGINAL

    def test_the_mde_is_reported_when_nothing_is_resolvable(self, target, patcher):
        """So the reader learns what the workload could have detected."""
        loop = self._loop(
            target,
            patcher,
            [100.0, 105.0, 95.0, 103.0, 97.0, 101.0],
            [99.0, 104.0, 96.0, 102.0, 98.0, 100.0],
        )
        assert "MDE" in loop.run([_proposal()]).trials[0].reason

    def test_a_candidate_with_no_samples_is_unproven(self, target, patcher):
        loop = OptimizationLoop(
            target=target,
            patcher=patcher,
            check=lambda: CORRECT,
            measure=lambda: None,
            measure_samples=iter([[100.0] * 6, []]).__next__,
            ledger=NoveltyLedger(),
        )
        assert loop.run([_proposal()]).trials[0].verdict is TrialVerdict.UNPROVEN

    def test_correctness_still_gates_before_the_statistics(self, target, patcher):
        loop = self._loop(target, patcher, [100.0] * 6, [1.0] * 6, check=lambda: WRONG)
        record = loop.run([_proposal()]).trials[0]
        assert record.verdict is TrialVerdict.REVERTED_WRONG

    def test_the_threshold_path_still_works_without_samples(self, target, patcher):
        """Callers that can only produce one number keep the old behaviour."""
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__)
        assert loop.run([_proposal()]).trials[0].verdict is TrialVerdict.KEPT


class TestInterleavedSampling:
    """ABBA at kernel level (§17 item 2).

    Baseline-then-candidate lets drift over the run land entirely on the second arm, and
    at kernel level that drift is real and indistinguishable from the effect under test.
    """

    def test_both_arms_get_samples(self, target, patcher):
        from xe_forge.orbit.optimize.loop import interleaved_samples

        base, cand = interleaved_samples(patcher, target, b"CANDIDATE\n", lambda: 1.0, pairs=2)
        assert len(base) == 4 and len(cand) == 4

    def test_the_arm_order_alternates_between_cycles(self, target, patcher):
        """ABAB would give the baseline first position every time."""
        from xe_forge.orbit.optimize.loop import interleaved_samples

        seen = []
        interleaved_samples(
            patcher,
            target,
            b"CANDIDATE\n",
            lambda: (seen.append(target.read_bytes()), 1.0)[1],
            pairs=2,
        )
        arms = ["C" if s == b"CANDIDATE\n" else "B" for s in seen]
        assert arms[:4] == ["B", "C", "C", "B"]
        assert arms[4:] == ["C", "B", "B", "C"]

    def test_the_tree_is_clean_afterwards(self, target, patcher):
        from xe_forge.orbit.optimize.loop import interleaved_samples

        interleaved_samples(patcher, target, b"CANDIDATE\n", lambda: 1.0, pairs=1)
        assert target.read_bytes() == ORIGINAL
        assert patcher.outstanding == []

    def test_a_failed_measurement_is_dropped_not_recorded_as_zero(self, target, patcher):
        """A run that produced nothing is missing data, not a fast result."""
        from xe_forge.orbit.optimize.loop import interleaved_samples

        values = iter([1.0, None, 1.0, 1.0])
        base, cand = interleaved_samples(
            patcher, target, b"CANDIDATE\n", lambda: next(values, 1.0), pairs=1
        )
        assert 0.0 not in base and 0.0 not in cand
        assert len(base) + len(cand) == 3

    def test_the_candidate_is_actually_applied_while_measured(self, target, patcher):
        from xe_forge.orbit.optimize.loop import interleaved_samples

        observed = []
        interleaved_samples(
            patcher,
            target,
            b"CANDIDATE\n",
            lambda: (observed.append(target.read_bytes()), 1.0)[1],
            pairs=1,
        )
        assert b"CANDIDATE\n" in observed
        assert ORIGINAL in observed


class TestAWinnerSurvivesLaterTrials:
    """A later failing trial must not undo an earlier accepted one.

    Measured: a two-round run reported two KEPT verdicts and finished with an unmodified
    tree, because the last trial's revert reverted the whole journal and took the
    accepted patch with it. The loop reported success and had silently discarded it.
    """

    def test_a_failure_after_a_win_leaves_the_win_applied(self, target, patcher):
        checks = iter([CORRECT, WRONG])
        loop = OptimizationLoop(
            target=target,
            patcher=patcher,
            check=lambda: next(checks),
            measure=iter([100.0, 80.0, 80.0]).__next__,
            ledger=NoveltyLedger(),
        )
        result = loop.run(
            [_proposal("winner", b"WINNER\n", knob="a"), _proposal("loser", b"LOSER\n", knob="b")]
        )
        assert result.trials[0].verdict is TrialVerdict.KEPT
        assert result.trials[1].verdict is TrialVerdict.REVERTED_WRONG
        assert target.read_bytes() == b"WINNER\n", "the winner was reverted by a later trial"

    def test_a_slower_candidate_after_a_win_also_leaves_it(self, target, patcher):
        loop = OptimizationLoop(
            target=target,
            patcher=patcher,
            check=lambda: CORRECT,
            measure=iter([100.0, 80.0, 130.0]).__next__,
            ledger=NoveltyLedger(),
            min_improvement_percent=2.0,
        )
        result = loop.run(
            [_proposal("winner", b"WINNER\n", knob="a"), _proposal("slower", b"SLOWER\n", knob="b")]
        )
        assert result.trials[0].verdict is TrialVerdict.KEPT
        assert target.read_bytes() == b"WINNER\n"

    def test_an_unproven_candidate_after_a_win_also_leaves_it(self, target, patcher):
        """The exact live case: a NameError in the second candidate."""
        checks = iter([CORRECT, UNCHECKED])
        loop = OptimizationLoop(
            target=target,
            patcher=patcher,
            check=lambda: next(checks),
            measure=iter([100.0, 80.0, 80.0]).__next__,
            ledger=NoveltyLedger(),
        )
        result = loop.run(
            [_proposal("winner", b"WINNER\n", knob="a"), _proposal("broken", b"BROKEN\n", knob="b")]
        )
        assert result.trials[1].verdict is TrialVerdict.UNPROVEN
        assert target.read_bytes() == b"WINNER\n"

    def test_with_no_winner_the_tree_is_still_restored(self, target, patcher):
        """The scoping fix must not leave losers behind."""
        loop = _loop(target, patcher, lambda: WRONG, lambda: 50.0)
        loop.run(
            [_proposal("a", b"A\n", knob="a"), _proposal("b", b"B\n", knob="b")], baseline_us=100.0
        )
        assert target.read_bytes() == ORIGINAL
