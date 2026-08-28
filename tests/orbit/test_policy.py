"""
The minimal policy gate (plan §24 Tier C).

The gate is deliberately three invariants — action allowlist, sandbox, single-writer —
and the properties defended here are the ones that make it a gate rather than a wrapper:
every refusal names the invariant that refused and why, the path invariants are the
patcher's own rather than a second copy that could drift, and a stale lock is broken
loudly rather than silently.

CPU-only, like the rest of the suite: nothing here needs a device.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from xe_forge.orbit.novelty import NoveltyLedger
from xe_forge.orbit.optimize.harness import CheckOutcome, CheckResult
from xe_forge.orbit.optimize.loop import OptimizationLoop, Proposal, TrialVerdict
from xe_forge.orbit.patch.inplace import InPlacePatcher, PatchSafetyError
from xe_forge.orbit.policy import PolicyGate, PolicyViolation

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


@pytest.fixture
def gate(patcher):
    return PolicyGate(patcher)


def _dead_pid() -> int:
    """A pid that provably belongs to no running process: a reaped child's."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


class TestActionAllowlist:
    def test_an_allowed_action_passes(self, gate):
        gate.check_action("apply_patch")

    def test_a_disallowed_action_is_refused_naming_the_action(self, gate):
        with pytest.raises(PolicyViolation) as excinfo:
            gate.check_action("delete_tree")
        assert "delete_tree" in str(excinfo.value)

    def test_the_refusal_says_what_would_have_been_allowed(self, gate):
        """A caller told only "no" cannot fix the request; name the granted set."""
        with pytest.raises(PolicyViolation) as excinfo:
            gate.check_action("rewrite_config")
        assert "apply_patch" in str(excinfo.value)

    def test_an_empty_allowlist_grants_nothing_and_says_so(self, patcher):
        sealed = PolicyGate(patcher, allowed_actions=frozenset())
        with pytest.raises(PolicyViolation, match="nothing"):
            sealed.check_action("apply_patch")


class TestWriteInvariants:
    """`check_write` is the patcher's own `check`, seen through one exception type.

    The invariants themselves are tested in test_inplace.py; here we prove they are
    reached, not re-implemented, and that the wrapped refusal keeps its reason.
    """

    def test_a_target_inside_the_sandbox_passes(self, gate, target):
        gate.check_write(target)

    def test_a_target_outside_the_sandbox_is_refused_with_the_reason(self, tmp_path, target):
        stray = InPlacePatcher(journal_dir=tmp_path / "j", sandbox_roots=[tmp_path / "elsewhere"])
        with pytest.raises(PolicyViolation, match="sandbox") as excinfo:
            PolicyGate(stray).check_write(target)
        # One gate, one exception type: the patcher's refusal is the cause, so the
        # original invariant stays visible without the caller catching two types.
        assert isinstance(excinfo.value.__cause__, PatchSafetyError)

    def test_a_symlink_target_is_refused(self, gate, target):
        link = target.parent / "alias.py"
        link.symlink_to(target)
        with pytest.raises(PolicyViolation, match="symlink"):
            gate.check_write(link)

    def test_a_missing_target_is_refused(self, gate, target):
        with pytest.raises(PolicyViolation, match="does not exist"):
            gate.check_write(target.parent / "no_such_kernel.py")


class TestSingleWriter:
    def test_a_second_acquisition_on_the_same_target_is_refused_naming_the_holder(
        self, gate, target
    ):
        with gate.single_writer(target):
            with pytest.raises(PolicyViolation) as excinfo:
                with gate.single_writer(target):
                    pass
        assert str(os.getpid()) in str(excinfo.value)
        assert "single-writer" in str(excinfo.value)

    def test_a_released_lock_is_reacquirable(self, gate, target):
        with gate.single_writer(target):
            pass
        with gate.single_writer(target):
            pass

    def test_release_removes_the_lock_file(self, gate, target):
        with gate.single_writer(target):
            locks = list(gate.lock_dir.glob("*.lock"))
            assert len(locks) == 1
        assert list(gate.lock_dir.glob("*.lock")) == []

    def test_a_stale_lock_is_broken_with_a_note_naming_the_dead_holder(self, gate, target, caplog):
        dead = _dead_pid()
        lock = gate._lock_path(target)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": dead, "target": str(target)}))
        with caplog.at_level(logging.WARNING, logger="xe_forge.orbit.policy"):
            with gate.single_writer(target):
                pass
        assert "stale" in caplog.text
        assert str(dead) in caplog.text

    def test_an_unreadable_lock_is_broken_with_a_note_not_silently(self, gate, target, caplog):
        """A writer killed between creating and stamping the lock leaves no holder."""
        lock = gate._lock_path(target)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("")
        with caplog.at_level(logging.WARNING, logger="xe_forge.orbit.policy"):
            with gate.single_writer(target):
                pass
        assert "stale" in caplog.text


CORRECT = CheckResult(CheckOutcome.CORRECT, accuracy=1.0, detail="matches")


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


class TestLoopIntegration:
    def test_a_gate_that_grants_nothing_refuses_every_proposal_with_the_policy_reason(
        self, target, patcher
    ):
        sealed = PolicyGate(patcher, allowed_actions=frozenset())
        loop = _loop(target, patcher, lambda: CORRECT, lambda: 100.0, policy=sealed)
        result = loop.run(
            [_proposal("a", b"A\n", knob="a"), _proposal("b", b"B\n", knob="b")],
            baseline_us=100.0,
        )
        assert [t.verdict for t in result.trials] == [TrialVerdict.REFUSED] * 2
        assert all("apply_patch" in t.reason for t in result.trials)
        assert target.read_bytes() == ORIGINAL

    def test_the_default_gate_matches_a_policy_less_loop_on_the_happy_path(
        self, target, patcher, tmp_path
    ):
        gated = _loop(
            target,
            patcher,
            lambda: CORRECT,
            iter([100.0, 80.0]).__next__,
            policy=PolicyGate(patcher),
        )
        gated_result = gated.run([_proposal()])

        # The same trial through a policy-less loop, on a fresh copy of the tree.
        bare_target = tmp_path / "bare" / "kernel.py"
        bare_target.parent.mkdir(parents=True)
        bare_target.write_bytes(ORIGINAL)
        bare_patcher = InPlacePatcher(
            journal_dir=tmp_path / "bare-run", sandbox_roots=[bare_target.parent]
        )
        bare = _loop(bare_target, bare_patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__)
        bare_result = bare.run([_proposal()])

        assert gated_result.trials[0].verdict is TrialVerdict.KEPT
        assert bare_result.trials[0].verdict is TrialVerdict.KEPT
        assert gated_result.trials[0].reason == bare_result.trials[0].reason
        assert target.read_bytes() == bare_target.read_bytes() == b"BLOCK_SIZE = 2048\n"

    def test_an_outside_sandbox_target_is_refused_through_the_gate(self, tmp_path, target):
        stray = InPlacePatcher(journal_dir=tmp_path / "j", sandbox_roots=[tmp_path / "elsewhere"])
        loop = _loop(target, stray, lambda: CORRECT, lambda: 100.0, policy=PolicyGate(stray))
        result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.REFUSED
        assert "sandbox" in result.trials[0].reason

    def test_the_lock_is_released_after_every_trial(self, target, patcher):
        gate = PolicyGate(patcher)
        loop = _loop(target, patcher, lambda: CORRECT, iter([100.0, 80.0]).__next__, policy=gate)
        loop.run([_proposal()])
        assert list(gate.lock_dir.glob("*.lock")) == []
        # The target is not left locked against the next run either.
        with gate.single_writer(target):
            pass

    def test_a_lock_held_elsewhere_refuses_the_trial_as_refused_not_an_error(self, target, patcher):
        """A live concurrent writer turns the trial into a REFUSED verdict, not a crash."""
        gate = PolicyGate(patcher)
        loop = _loop(target, patcher, lambda: CORRECT, lambda: 100.0, policy=gate)
        with gate.single_writer(target):
            result = loop.run([_proposal()], baseline_us=100.0)
        assert result.trials[0].verdict is TrialVerdict.REFUSED
        assert str(os.getpid()) in result.trials[0].reason
        assert target.read_bytes() == ORIGINAL
