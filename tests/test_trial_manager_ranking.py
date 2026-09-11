"""How the trial tree orders a gated comparison against a measured one.

A host may gate its comparison -- report both arms' times and withhold the ratio because
the difference was below what the part can resolve (see :mod:`xe_forge.external`). These
tests pin the consequences of that, because the failure they guard against is silent: the
run ends, a file is written, and the kernel in it is the slow one.
"""

from __future__ import annotations

import pytest

from xe_forge.core.trial_manager import TrialManager


@pytest.fixture
def mgr(tmp_path):
    m = TrialManager(tmp_path / "trials")
    baseline = tmp_path / "base.cpp"
    baseline.write_text("// baseline\n")
    m.init("k", baseline)
    return m


def _trial(mgr, tmp_path, name, **result):
    src = tmp_path / name
    src.write_text(f"// {name}\n")
    tid = mgr.save_trial("k", src, strategy=name)
    mgr.record_result("k", tid, correctness="pass", **result)
    return tid


def test_gated_trial_outranks_a_measured_regression(mgr, tmp_path):
    """The case that shipped the wrong kernel: parity must beat 0.66x.

    Ranking on ``speedup`` alone makes the gated trial invisible, so the regression is
    the only candidate and becomes ``best_trial``.
    """
    gated = _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6,
                   verdict="INDISTINGUISHABLE")
    _trial(mgr, tmp_path, "t1.cpp", speedup=0.66, baseline_us=255.4, triton_us=390.2)

    assert mgr.get_best("k")["id"] == gated


def test_a_gated_trial_records_no_speedup(mgr, tmp_path):
    """Ranking at parity must not put a ratio into the record the loop reads back."""
    tid = _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6,
                 verdict="BELOW_FLOOR")
    best = mgr.get_best("k")

    assert best["id"] == tid
    assert best["speedup"] is None
    assert best["verdict"] == "BELOW_FLOOR"
    assert best["status"] == "completed"  # finished, not half-recorded


def test_a_measured_win_outranks_parity(mgr, tmp_path):
    _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6,
           verdict="INDISTINGUISHABLE")
    win = _trial(mgr, tmp_path, "t1.cpp", speedup=1.4, baseline_us=255.2, triton_us=182.3)

    assert mgr.get_best("k")["id"] == win


def test_a_measured_parity_outranks_a_gated_one(mgr, tmp_path):
    """Same rank, so the tie goes to the trial whose ratio was actually resolved."""
    _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6,
           verdict="INDISTINGUISHABLE")
    measured = _trial(mgr, tmp_path, "t1.cpp", speedup=1.0, baseline_us=255.2,
                      triton_us=255.2)

    assert mgr.get_best("k")["id"] == measured


def test_an_incorrect_trial_never_ranks(mgr, tmp_path):
    src = tmp_path / "t0.cpp"
    src.write_text("// wrong\n")
    tid = mgr.save_trial("k", src)
    mgr.record_result("k", tid, correctness="fail", baseline_us=255.2, triton_us=10.0)

    assert mgr.get_best("k") is None


def test_finalize_refuses_a_regression(mgr, tmp_path):
    """Nothing to keep is a result; writing out the slower kernel is not."""
    _trial(mgr, tmp_path, "t0.cpp", speedup=0.66, baseline_us=255.4, triton_us=390.2)
    out = tmp_path / "out.cpp"

    assert mgr.finalize("k", out) is None
    assert not out.exists()


def test_finalize_keeps_parity(mgr, tmp_path):
    """A kernel that matches the baseline is a legitimate thing to keep."""
    tid = _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6,
                 verdict="INDISTINGUISHABLE")
    out = tmp_path / "out.cpp"

    assert mgr.finalize("k", out) == tid
    assert out.read_text() == "// t0.cpp\n"
