"""`xe-orbit optimize --apply --rounds N` — the in-process learning loop (§13.7).

test_cli_optimize.py covers one round; these tests cover what N rounds add: each
round's measured verdicts, with their reasons, reach the next round's plan call; a
KEPT round stops the session with the winner applied; the novelty ledger persists
across rounds so an identical re-proposal is refused as a stall; and a session that
stops says why. A stub proposer scripts each round's batch — §15.3 requires that CI
never call an LLM.

The measurement command reads the target file itself (counting '#' markers), so
which arm is faster is decided by what is on disk at sample time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from xe_forge.orbit.models import KernelRecord
from xe_forge.orbit.optimize.loop import Proposal

SLOW_SOURCE = b"# slow\n# slow\n# slow\nVALUE = 1\n"  # 3 markers -> 400us
FAST_SOURCE = b"VALUE = 1\n"  # 0 markers -> 100us
SLOWER_SOURCE = b"# slow\n" * 8 + b"VALUE = 1\n"  # 8 markers -> 900us


class ScriptedRoundProposer:
    """Extends test_cli_optimize's stub with scripted plan() batches for later rounds.

    Round 1's proposals enter `_run_optimize_loop` directly — the CLI plans them in
    `cmd_optimize` — so `plan` here serves only rounds 2..N. Each call pops the next
    scripted batch and records the knowledge text it was handed, which is what the
    feedback tests assert on.
    """

    def __init__(
        self,
        edits: dict[str, bytes | None],
        round_batches: list[list[str]] | None = None,
    ):
        self.edits = edits
        self.round_batches = list(round_batches or [])
        self.plan_knowledge: list[str] = []
        self.workspaces: list[Path] = []

    def plan(self, source, knowledge="", count=3, kernel_label=""):
        self.plan_knowledge.append(knowledge)
        if not self.round_batches:
            return []
        return [Proposal(title=t, rationale="scripted") for t in self.round_batches.pop(0)]

    def implement(self, proposal, target, workspace, harness=None, harness_command=""):
        self.workspaces.append(Path(workspace))
        return self.edits.get(proposal.title)


@pytest.fixture
def target(tmp_path):
    kernel_file = tmp_path / "tree" / "kernel.py"
    kernel_file.parent.mkdir()
    kernel_file.write_bytes(SLOW_SOURCE)
    return kernel_file


@pytest.fixture
def harness_ok(tmp_path):
    script = tmp_path / "harness_ok.py"
    script.write_text("import sys\nprint('ACCURACY 1.000000 10/10')\nsys.exit(0)\n")
    return script


def _measure_command(kernel_file: Path) -> str:
    return (
        f'{sys.executable} -c "import pathlib; '
        f"print(100 + 100 * pathlib.Path(r'{kernel_file}').read_text().count('#'))\""
    )


def _args(harness: Path, kernel_file: Path, **overrides) -> argparse.Namespace:
    base = {
        "apply": True,
        "harness": str(harness),
        "measure": _measure_command(kernel_file),
        "samples": 0,
        "min_improvement": 1.0,
        "sandbox": None,
        "trials": 3,
        "rounds": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _kernel(kernel_file: Path) -> KernelRecord:
    return KernelRecord(id="k0", runtime_name="mini_kernel", source_file=str(kernel_file))


def _run(store, target, harness, proposer, first_round_titles, **arg_overrides) -> int:
    from xe_forge.orbit.cli import _run_optimize_loop

    proposals = [Proposal(title=t, rationale="stub") for t in first_round_titles]
    args = _args(harness, target, **arg_overrides)
    return _run_optimize_loop(args, store, _kernel(target), proposer, proposals)


def _round_payload(store, n: int):
    path = store.subdir("experiments", "k0") / f"loop_result_round{n}.json"
    return json.loads(path.read_text()) if path.is_file() else None


class TestFeedbackReachesTheNextRound:
    def test_round_two_plan_receives_round_one_verdicts_and_reasons(
        self, store, target, harness_ok
    ):
        proposer = ScriptedRoundProposer(
            {"add markers": SLOWER_SOURCE, "strip markers": FAST_SOURCE},
            round_batches=[["strip markers"]],
        )
        code = _run(store, target, harness_ok, proposer, ["add markers"], rounds=2)
        assert code == 0
        [knowledge] = proposer.plan_knowledge
        assert "add markers" in knowledge
        assert "REVERTED_SLOWER" in knowledge
        # (400 - 900) / 400: the measured delta itself, not just the verdict label.
        assert "-125.00%" in knowledge
        # The loop's *reason* travels too — the words the verdict was decided with.
        assert "a clear regression" in knowledge
        assert "do not re-propose" in knowledge
        # The corrected direction from round 2 is the one that ends up on disk.
        assert target.read_bytes() == FAST_SOURCE

    def test_round_workspaces_do_not_collide_with_round_one(self, store, target, harness_ok):
        proposer = ScriptedRoundProposer(
            {"add markers": SLOWER_SOURCE, "strip markers": FAST_SOURCE},
            round_batches=[["strip markers"]],
        )
        _run(store, target, harness_ok, proposer, ["add markers"], rounds=2)
        experiments = store.subdir("experiments", "k0")
        assert proposer.workspaces == [
            experiments / "proposal_0",
            experiments / "round2_proposal_0",
        ]


class TestEarlyStopOnKept:
    def test_a_kept_round_stops_the_session_before_planning_again(
        self, store, target, harness_ok, capsys
    ):
        proposer = ScriptedRoundProposer(
            {"strip markers": FAST_SOURCE},
            round_batches=[["never planned"]],
        )
        code = _run(store, target, harness_ok, proposer, ["strip markers"], rounds=3)
        assert code == 0
        assert proposer.plan_knowledge == []  # round 2 was never planned
        assert target.read_bytes() == FAST_SOURCE  # the winner stays applied
        assert _round_payload(store, 1)["trials"][0]["verdict"] == "KEPT"
        assert _round_payload(store, 2) is None
        out = capsys.readouterr().out
        assert "1 round(s) run" in out
        assert "KEPT" in out


class TestRoundsExhausted:
    def test_every_round_persists_and_the_last_is_the_final_record(
        self, store, target, harness_ok, capsys
    ):
        proposer = ScriptedRoundProposer(
            {
                "more markers": SLOWER_SOURCE,
                "even more markers": SLOWER_SOURCE + b"# slow\n",
            },
            round_batches=[["even more markers"]],
        )
        code = _run(store, target, harness_ok, proposer, ["more markers"], rounds=2)
        assert code == 0
        assert target.read_bytes() == SLOW_SOURCE  # nothing accepted; the tree is unchanged
        assert _round_payload(store, 1)["trials"][0]["title"] == "more markers"
        assert _round_payload(store, 2)["trials"][0]["title"] == "even more markers"
        final = json.loads((store.subdir("experiments", "k0") / "loop_result.json").read_text())
        assert final == _round_payload(store, 2)
        out = capsys.readouterr().out
        assert "2 round(s) run" in out
        assert "exhausted" in out


class TestLedgerPersistsAcrossRounds:
    def test_an_identical_round_two_reproposal_is_refused_as_a_stall(
        self, store, target, harness_ok
    ):
        proposer = ScriptedRoundProposer(
            {"more markers": SLOWER_SOURCE},
            round_batches=[["more markers"]],  # the same attempt, re-proposed
        )
        code = _run(store, target, harness_ok, proposer, ["more markers"], rounds=2)
        assert code == 0
        assert _round_payload(store, 1)["trials"][0]["verdict"] == "REVERTED_SLOWER"
        round2 = _round_payload(store, 2)
        assert round2["trials"][0]["verdict"] == "REFUSED"
        assert "identical attempt" in round2["trials"][0]["reason"]
        assert target.read_bytes() == SLOW_SOURCE


class TestNoProposalsStops:
    def test_an_empty_plan_stops_the_session_and_says_so(self, store, target, harness_ok, capsys):
        proposer = ScriptedRoundProposer({"more markers": SLOWER_SOURCE}, round_batches=[])
        code = _run(store, target, harness_ok, proposer, ["more markers"], rounds=3)
        assert code == 0
        assert _round_payload(store, 2) is None
        out = capsys.readouterr().out
        assert "no proposals" in out
        assert "1 round(s) run" in out

    def test_a_round_with_no_implementable_edit_stops_the_session(
        self, store, target, harness_ok, capsys
    ):
        proposer = ScriptedRoundProposer(
            {"more markers": SLOWER_SOURCE, "nothing produced": None},
            round_batches=[["nothing produced"]],
        )
        code = _run(store, target, harness_ok, proposer, ["more markers"], rounds=2)
        assert code == 0
        assert _round_payload(store, 2) is None
        out = capsys.readouterr().out
        assert "no implementable edit" in out


class TestSingleRoundIsUnchanged:
    def test_rounds_one_never_plans_and_writes_no_round_files(self, store, target, harness_ok):
        proposer = ScriptedRoundProposer(
            {"strip markers": FAST_SOURCE}, round_batches=[["should not be planned"]]
        )
        code = _run(store, target, harness_ok, proposer, ["strip markers"], rounds=1)
        assert code == 0
        assert proposer.plan_knowledge == []
        assert _round_payload(store, 1) is None
        assert (store.subdir("experiments", "k0") / "loop_result.json").is_file()
        assert target.read_bytes() == FAST_SOURCE

    def test_a_namespace_without_rounds_means_one_round(self, store, target, harness_ok):
        # test_cli_optimize.py builds its namespace without `rounds`; the seam must
        # treat the missing attribute as the single-round default, not as an error.
        from xe_forge.orbit.cli import _run_optimize_loop

        proposer = ScriptedRoundProposer({"strip markers": FAST_SOURCE})
        args = _args(harness_ok, target)
        delattr(args, "rounds")
        proposals = [Proposal(title="strip markers", rationale="stub")]
        code = _run_optimize_loop(args, store, _kernel(target), proposer, proposals)
        assert code == 0
        assert target.read_bytes() == FAST_SOURCE
        assert _round_payload(store, 1) is None


class TestParser:
    def test_rounds_defaults_to_one(self):
        from xe_forge.orbit.cli import build_parser

        args = build_parser().parse_args(["optimize", "k0"])
        assert args.rounds == 1

    def test_rounds_is_settable(self):
        from xe_forge.orbit.cli import build_parser

        args = build_parser().parse_args(["optimize", "k0", "--rounds", "4"])
        assert args.rounds == 4


class TestRenderForKnowledge:
    def test_an_empty_history_renders_as_nothing(self):
        from xe_forge.orbit.optimize.session import SessionHistory

        assert SessionHistory().render_for_knowledge() == ""

    def test_the_framing_wraps_the_measurements(self):
        from xe_forge.orbit.optimize.loop import TrialRecord, TrialVerdict
        from xe_forge.orbit.optimize.session import RoundOutcome, SessionHistory

        trial = TrialRecord(
            index=0,
            proposal=Proposal(title="raise BLOCK_SIZE", rationale=""),
            verdict=TrialVerdict.REVERTED_SLOWER,
            reason="a clear regression",
            baseline_us=100.0,
            candidate_us=208.87,
        )
        rendered = SessionHistory(rounds=[RoundOutcome(0, [trial])]).render_for_knowledge()
        assert "measurements, not opinions" in rendered
        assert "do not re-propose" in rendered
        assert "-108.87%" in rendered
