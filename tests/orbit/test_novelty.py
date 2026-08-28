"""
The stall gate (plan §20.4).

A loop that can retry will retry, and the cheapest thing to retry is what it just did.
The result looks busy and ends where it started.
"""

from __future__ import annotations

from xe_forge.orbit.novelty import Attempt, NoveltyLedger, Verdict


class TestRepeatsAreRefused:
    def test_a_fresh_attempt_is_novel(self):
        ledger = NoveltyLedger()
        assert ledger.classify(Attempt("build", "sgl-kernel"))[0] is Verdict.NOVEL

    def test_an_identical_attempt_is_a_stall(self):
        ledger = NoveltyLedger()
        attempt = Attempt("build", "sgl-kernel")
        ledger.record(attempt)
        verdict, reason = ledger.classify(attempt)
        assert verdict is Verdict.STALL
        assert "already ran" in reason

    def test_a_refusal_always_explains_itself(self):
        """A refusal the caller cannot explain is indistinguishable from a bug."""
        ledger = NoveltyLedger()
        attempt = Attempt("build", "sgl-kernel")
        ledger.record(attempt)
        assert ledger.classify(attempt)[1].strip()

    def test_admits_is_the_same_decision_as_classify(self):
        ledger = NoveltyLedger()
        attempt = Attempt("build", "x")
        assert ledger.admits(attempt)
        ledger.record(attempt)
        assert not ledger.admits(attempt)


class TestNoveltyIsAboutTheAttemptNotTheOutcome:
    def test_a_different_target_is_novel(self):
        ledger = NoveltyLedger()
        ledger.record(Attempt("build", "sgl-kernel"))
        assert ledger.admits(Attempt("build", "vllm-xpu-kernels"))

    def test_a_different_action_is_novel(self):
        ledger = NoveltyLedger()
        ledger.record(Attempt("build", "x"))
        assert ledger.admits(Attempt("patch", "x"))

    def test_different_parameters_make_a_different_experiment(self):
        ledger = NoveltyLedger()
        ledger.record(Attempt("compile", "k0", {"grf": "large"}))
        assert ledger.admits(Attempt("compile", "k0", {"grf": "default"}))

    def test_failing_differently_still_counts_as_progress(self):
        """Only sameness is the problem; a novel attempt that fails moves the search."""
        ledger = NoveltyLedger()
        for grf in ("large", "default", "auto"):
            attempt = Attempt("compile", "k0", {"grf": grf})
            assert ledger.admits(attempt)
            ledger.record(attempt)
        assert ledger.distinct_attempts == 3


class TestIdentityIsNormalized:
    def test_dict_ordering_does_not_make_a_repeat_look_novel(self):
        """Key order says nothing about the attempt, so it must not affect identity."""
        ledger = NoveltyLedger()
        ledger.record(Attempt("compile", "k0", {"grf": "large", "sg": 32}))
        assert not ledger.admits(Attempt("compile", "k0", {"sg": 32, "grf": "large"}))

    def test_unserializable_parameters_do_not_crash_the_gate(self):
        """A gate that raises on an odd parameter is worse than one that is coarse."""
        ledger = NoveltyLedger()
        attempt = Attempt("compile", "k0", {"path": object()})
        ledger.record(attempt)
        assert ledger.classify(attempt)[0] is Verdict.STALL


class TestTimeoutsAreNotRepeats:
    def test_one_retry_is_allowed_after_a_timeout(self):
        """A timeout describes the machine, not the attempt."""
        ledger = NoveltyLedger()
        attempt = Attempt("build", "vllm")
        ledger.record(attempt, timed_out=True)
        verdict, reason = ledger.classify(attempt)
        assert verdict is Verdict.RETRY
        assert "timed out" in reason

    def test_the_allowance_is_bounded(self):
        """Retrying forever is how a stall gate becomes decorative."""
        ledger = NoveltyLedger(timeout_retries=1)
        attempt = Attempt("build", "vllm")
        ledger.record(attempt, timed_out=True)
        ledger.record(attempt, timed_out=True)
        verdict, reason = ledger.classify(attempt)
        assert verdict is Verdict.STALL
        assert "does not finish" in reason

    def test_a_real_outcome_ends_the_allowance_immediately(self):
        """Once it produced a result, more time is not the missing ingredient."""
        ledger = NoveltyLedger()
        attempt = Attempt("build", "vllm")
        ledger.record(attempt, timed_out=True)
        ledger.record(attempt)
        assert ledger.classify(attempt)[0] is Verdict.STALL

    def test_zero_retries_refuses_a_repeat_timeout(self):
        ledger = NoveltyLedger(timeout_retries=0)
        attempt = Attempt("build", "vllm")
        ledger.record(attempt, timed_out=True)
        assert ledger.classify(attempt)[0] is Verdict.STALL


class TestAccounting:
    def test_an_empty_ledger_says_so(self):
        assert "nothing attempted yet" in NoveltyLedger().format()

    def test_repeats_are_surfaced_in_the_summary(self):
        ledger = NoveltyLedger()
        attempt = Attempt("build", "vllm")
        ledger.record(attempt, timed_out=True)
        ledger.record(attempt, timed_out=True)
        assert ledger.distinct_attempts == 1
        assert ledger.total_attempts == 2
        assert "repeats" in ledger.format()
