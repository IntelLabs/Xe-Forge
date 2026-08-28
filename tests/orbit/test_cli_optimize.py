"""`xe-orbit optimize --apply` — the CLI seam onto §13.5's loop (gap G1).

The loop itself is covered in test_optimize_loop.py; these tests cover the wiring the
CLI adds around it: implement-in-workspace, the journalled patcher with the target's
directory as sandbox, the measurement-command contract, and the persisted trial
record. A stub proposer stands in for Claude — the seam is what is under test, and
§15.3 requires that CI never call an LLM.

The measurement command reads the *target file itself* (counting '#' markers), so
which arm is faster is decided by what is on disk at sample time — exactly the
property the loop is supposed to guarantee via apply/revert ordering.
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


class StubProposer:
    """Duck-types the one method the seam calls; records the workspaces it was given."""

    def __init__(self, edits: dict[str, bytes | None]):
        self.edits = edits
        self.workspaces: list[Path] = []

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


@pytest.fixture
def harness_wrong(tmp_path):
    script = tmp_path / "harness_wrong.py"
    script.write_text("import sys\nprint('ACCURACY 0.500000 5/10')\nsys.exit(1)\n")
    return script


def _measure_command(kernel_file: Path) -> str:
    # Prints 100 + 100 per '#" marker, so the on-disk content decides the timing.
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
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _kernel(kernel_file: Path) -> KernelRecord:
    return KernelRecord(id="k0", runtime_name="mini_kernel", source_file=str(kernel_file))


def _run(store, target, harness, edits, **arg_overrides) -> int:
    from xe_forge.orbit.cli import _run_optimize_loop

    proposals = [Proposal(title=title, rationale="stub") for title in edits]
    proposer = StubProposer(edits)
    args = _args(harness, target, **arg_overrides)
    return _run_optimize_loop(args, store, _kernel(target), proposer, proposals)


class TestAcceptPath:
    def test_faster_correct_candidate_is_kept_on_disk(self, store, target, harness_ok):
        code = _run(store, target, harness_ok, {"strip markers": FAST_SOURCE})
        assert code == 0
        assert target.read_bytes() == FAST_SOURCE

    def test_trial_record_is_persisted(self, store, target, harness_ok):
        _run(store, target, harness_ok, {"strip markers": FAST_SOURCE})
        payload = json.loads((store.subdir("experiments", "k0") / "loop_result.json").read_text())
        assert payload["accepted"] == "strip markers"
        assert payload["trials"][0]["verdict"] == "KEPT"
        assert payload["trials"][0]["accuracy"] == pytest.approx(1.0)

    def test_journal_records_the_kept_edit(self, store, target, harness_ok):
        from xe_forge.orbit.patch.inplace import InPlacePatcher

        _run(store, target, harness_ok, {"strip markers": FAST_SOURCE})
        patcher = InPlacePatcher(journal_dir=store.run_dir)
        outstanding = patcher.outstanding
        assert [Path(r.target).name for r in outstanding] == ["kernel.py"]
        # Recovery restores the pristine source — the revert path §13.2 is judged by.
        patcher.recover()
        assert target.read_bytes() == SLOW_SOURCE


class TestRevertPaths:
    def test_slower_candidate_is_reverted(self, store, target, harness_ok):
        code = _run(store, target, harness_ok, {"add more markers": SLOWER_SOURCE})
        assert code == 0
        assert target.read_bytes() == SLOW_SOURCE
        payload = json.loads((store.subdir("experiments", "k0") / "loop_result.json").read_text())
        assert payload["accepted"] is None
        assert payload["trials"][0]["verdict"] == "REVERTED_SLOWER"

    def test_wrong_candidate_is_reverted_before_any_measurement(self, store, target, harness_wrong):
        code = _run(store, target, harness_wrong, {"break numerics": FAST_SOURCE})
        assert code == 0
        assert target.read_bytes() == SLOW_SOURCE
        payload = json.loads((store.subdir("experiments", "k0") / "loop_result.json").read_text())
        assert payload["trials"][0]["verdict"] == "REVERTED_WRONG"

    def test_proposal_without_an_edit_never_reaches_the_tree(self, store, target, harness_ok):
        code = _run(store, target, harness_ok, {"produced nothing": None})
        assert code == 1
        assert target.read_bytes() == SLOW_SOURCE


class TestSeamContracts:
    def test_workspaces_are_per_proposal_under_experiments(self, store, target, harness_ok):
        from xe_forge.orbit.cli import _run_optimize_loop

        edits = {"a": FAST_SOURCE, "b": SLOWER_SOURCE}
        proposals = [Proposal(title=t, rationale="stub") for t in edits]
        proposer = StubProposer(edits)
        _run_optimize_loop(_args(harness_ok, target), store, _kernel(target), proposer, proposals)
        experiments = store.subdir("experiments", "k0")
        assert proposer.workspaces == [
            experiments / "proposal_0",
            experiments / "proposal_1",
        ]

    def test_measure_once_reads_last_token_of_last_line(self, target):
        from xe_forge.orbit.cli import _measure_once

        assert _measure_once(_measure_command(target)) == pytest.approx(400.0)
        target.write_bytes(FAST_SOURCE)
        assert _measure_once(_measure_command(target)) == pytest.approx(100.0)

    def test_measure_once_returns_none_for_a_failing_command(self):
        from xe_forge.orbit.cli import _measure_once

        assert _measure_once(f"{sys.executable} -c 'import sys; sys.exit(3)'") is None
        assert _measure_once("printf 'not a number\\n'") is None

    def test_missing_harness_refuses_before_implementing(self, store, target, tmp_path):
        code = _run(store, target, tmp_path / "missing.py", {"x": FAST_SOURCE})
        assert code == 1
        assert target.read_bytes() == SLOW_SOURCE
