"""
The learning loop (plan §13.7).

Written after a live run where PLAN produced two well-argued candidates, both measured
~2x slower, and the agent was never told. Asked again it would have proposed the same
two things, because nothing carried a result backwards.
"""

from __future__ import annotations

from xe_forge.orbit.device import DeviceFacts
from xe_forge.orbit.optimize.loop import Proposal, TrialRecord, TrialVerdict
from xe_forge.orbit.optimize.session import (
    RoundOutcome,
    SessionHistory,
    build_round_prompt,
    violates_device_limits,
)


def _facts() -> DeviceFacts:
    return DeviceFacts(
        name="Intel(R) Graphics",
        integrated=True,
        eu_count=16,
        compute_units=16,
        max_work_group_size=1024,
        sub_group_sizes=[16, 32],
        memory_bus_width=64,
        total_memory_bytes=14862204928,
        available=True,
    )


def _trial(title, delta, verdict=TrialVerdict.REVERTED_SLOWER, reason="measured"):
    base, cand = (100.0, 100.0 - delta) if delta is not None else (None, None)
    return TrialRecord(
        index=0,
        proposal=Proposal(title=title, rationale=""),
        verdict=verdict,
        reason=reason,
        baseline_us=base,
        candidate_us=cand,
    )


class TestHistoryCarriesResultsForward:
    def test_an_empty_history_renders_as_nothing(self):
        assert SessionHistory().render() == ""

    def test_results_reach_the_next_round_with_their_numbers(self):
        h = SessionHistory(rounds=[RoundOutcome(0, [_trial("bigger blocks", -108.87)])])
        rendered = h.render()
        assert "bigger blocks" in rendered
        assert "-108.87%" in rendered

    def test_the_best_result_so_far_is_identified_across_rounds(self):
        h = SessionHistory(
            rounds=[
                RoundOutcome(0, [_trial("a", -50.0)]),
                RoundOutcome(1, [_trial("b", 3.0, TrialVerdict.KEPT)]),
            ]
        )
        assert h.best_so_far.proposal.title == "b"

    def test_failed_directions_are_stated_as_outcomes_not_prohibitions(self):
        """'Larger blocks measured 2x slower' is reasonable-from; 'do not' is only obeyable."""
        h = SessionHistory(rounds=[RoundOutcome(0, [_trial("raise BLOCK_SIZE", -108.87)])])
        directions = h.failed_directions()
        assert any("-108.87%" in d for d in directions)
        assert not any(d.lower().startswith("do not") for d in directions)

    def test_a_kept_trial_is_not_listed_as_a_failed_direction(self):
        h = SessionHistory(rounds=[RoundOutcome(0, [_trial("good", 5.0, TrialVerdict.KEPT)])])
        assert h.failed_directions() == []

    def test_unmeasured_trials_do_not_pollute_the_best(self):
        h = SessionHistory(
            rounds=[RoundOutcome(0, [_trial("unproven", None, TrialVerdict.UNPROVEN)])]
        )
        assert h.best_so_far is None
        assert not h.anything_measured


class TestDeviceLimits:
    """Rejecting the impossible for free, where a trial costs patch + check + measure."""

    def test_an_absurd_block_size_is_rejected_before_any_trial(self):
        p = Proposal("huge blocks", "", parameters={"BLOCK_SIZE": 65536})
        assert "register pressure" in violates_device_limits(p, _facts())

    def test_a_reasonable_block_size_is_left_to_the_measurement(self):
        """Being wrong about what is slow is what the gate is for."""
        p = Proposal("bigger blocks", "", parameters={"BLOCK_SIZE": 4096})
        assert violates_device_limits(p, _facts()) == ""

    def test_num_warps_past_the_work_group_limit_is_rejected(self):
        """The exact arithmetic that failed live: 16 warps x width 32 = 512."""
        p = Proposal("more warps", "", parameters={"num_warps": 64})
        why = violates_device_limits(p, _facts())
        assert "past this device's 1024 limit" in why

    def test_nothing_is_rejected_when_the_device_is_unknown(self):
        """An unknown device must not become a source of invented constraints."""
        p = Proposal("huge blocks", "", parameters={"BLOCK_SIZE": 65536})
        assert violates_device_limits(p, DeviceFacts()) == ""

    def test_a_proposal_with_no_parameters_is_not_rejected(self):
        assert violates_device_limits(Proposal("restructure", ""), _facts()) == ""


class TestRoundPrompt:
    def test_the_device_leads_the_prompt(self):
        """The failure was sound reasoning about a machine nobody named."""
        prompt = build_round_prompt("SRC", "ctx", _facts(), SessionHistory(), 2, "k", 0)
        assert "Intel(R) Graphics" in prompt
        assert "HARD LIMIT" in prompt
        assert "16 EUs" in prompt

    def test_the_first_round_carries_no_history_section(self):
        prompt = build_round_prompt("SRC", "ctx", _facts(), SessionHistory(), 2, "k", 0)
        assert "ALREADY BEEN TRIED" not in prompt

    def test_later_rounds_carry_the_measurements(self):
        h = SessionHistory(rounds=[RoundOutcome(0, [_trial("raise BLOCK_SIZE", -108.87)])])
        prompt = build_round_prompt("SRC", "ctx", _facts(), h, 2, "k", 1)
        assert "ALREADY BEEN TRIED AND MEASURED" in prompt
        assert "-108.87%" in prompt
        assert "Do not re-propose anything above" in prompt

    def test_the_agent_is_told_results_are_measurements_not_opinions(self):
        h = SessionHistory(rounds=[RoundOutcome(0, [_trial("x", -50.0)])])
        prompt = build_round_prompt("SRC", "ctx", _facts(), h, 2, "k", 1)
        assert "measurements, not opinions" in prompt

    def test_the_agent_may_decline_to_propose(self):
        """A refusal is a legitimate answer when nothing can beat the floor (§18)."""
        prompt = build_round_prompt("SRC", "ctx", _facts(), SessionHistory(), 2, "k", 0)
        assert "rather than proposing something you expect to fail" in prompt

    def test_the_round_number_is_visible_to_the_agent(self):
        assert "round 3" in build_round_prompt("S", "c", _facts(), SessionHistory(), 1, "k", 2)
