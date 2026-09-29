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
    gated = _trial(
        mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6, verdict="INDISTINGUISHABLE"
    )
    _trial(mgr, tmp_path, "t1.cpp", speedup=0.66, baseline_us=255.4, triton_us=390.2)

    assert mgr.get_best("k")["id"] == gated


def test_a_gated_trial_records_no_speedup(mgr, tmp_path):
    """Ranking at parity must not put a ratio into the record the loop reads back."""
    tid = _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6, verdict="BELOW_FLOOR")
    best = mgr.get_best("k")

    assert best["id"] == tid
    assert best["speedup"] is None
    assert best["verdict"] == "BELOW_FLOOR"
    assert best["status"] == "completed"  # finished, not half-recorded


def test_a_measured_win_outranks_parity(mgr, tmp_path):
    _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6, verdict="INDISTINGUISHABLE")
    win = _trial(mgr, tmp_path, "t1.cpp", speedup=1.4, baseline_us=255.2, triton_us=182.3)

    assert mgr.get_best("k")["id"] == win


def test_a_measured_parity_outranks_a_gated_one(mgr, tmp_path):
    """Same rank, so the tie goes to the trial whose ratio was actually resolved."""
    _trial(mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6, verdict="INDISTINGUISHABLE")
    measured = _trial(mgr, tmp_path, "t1.cpp", speedup=1.0, baseline_us=255.2, triton_us=255.2)

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
    tid = _trial(
        mgr, tmp_path, "t0.cpp", baseline_us=255.2, triton_us=252.6, verdict="INDISTINGUISHABLE"
    )
    out = tmp_path / "out.cpp"

    assert mgr.finalize("k", out) == tid
    assert out.read_text() == "// t0.cpp\n"


def test_required_profiles_gate_next_trial_and_finalization(tmp_path):
    manager = TrialManager(tmp_path / "trials")
    source = tmp_path / "kernel.cpp"
    source.write_text("baseline")
    manager.init("k", source, required_profile_groups=("ComputeBasic", "EuStallSampling"))
    manager.save_trial("k", source)
    source.write_text("candidate")
    trial = manager.save_trial("k", source)
    manager.record_result("k", trial, correctness="pass", speedup=1.1)
    digest = manager.profile_source_hash("k", trial, source)
    with pytest.raises(ValueError, match="Required profiling"):
        manager.save_trial("k", source)
    manager.record_profile(
        "k", trial, "ComputeBasic", digest, artifacts_dir="compute", error=None, warnings=[]
    )
    with pytest.raises(ValueError, match="EuStallSampling"):
        manager.finalize("k", tmp_path / "output.cpp")
    manager.record_profile(
        "k",
        trial,
        "EuStallSampling",
        digest,
        artifacts_dir="stalls",
        error="no samples",
        warnings=[],
    )
    assert manager.finalize("k", tmp_path / "output.cpp") == trial
    assert (
        manager._load_state("k")["trials"][trial]["profiles"]["EuStallSampling"]["status"]
        == "failed"
    )
    saved = manager._trial_dir("k") / "t1.cpp"
    saved.write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        manager.save_trial("k", source)
    with pytest.raises(ValueError, match="does not match"):
        manager.profile_source_hash("k", trial, source)


def test_required_profiles_accept_directory_trials(tmp_path):
    manager = TrialManager(tmp_path / "trials")
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "main.cpp").write_text("candidate")
    manager.init("k", source, required_profile_groups=("ComputeBasic", "EuStallSampling"))
    manager.save_trial("k", source)
    trial = manager.save_trial("k", source)
    manager.record_result("k", trial, correctness="pass", speedup=1.1)
    digest = manager.profile_source_hash("k", trial, source)
    for group in ("ComputeBasic", "EuStallSampling"):
        manager.record_profile(
            "k", trial, group, digest, artifacts_dir=None, error=None, warnings=[]
        )
    assert manager.finalize("k", tmp_path / "output") == trial


def test_rounded_speedup_tie_goes_to_faster_trial(mgr, tmp_path):
    _trial(mgr, tmp_path, "t0.cpp", speedup=1.28, baseline_us=11100.0, triton_us=8686.05)
    faster = _trial(mgr, tmp_path, "t1.cpp", speedup=1.28, baseline_us=11100.0, triton_us=8645.67)

    assert mgr.get_best("k")["id"] == faster


def test_required_profiles_gate_an_optimized_t0(tmp_path):
    manager = TrialManager(tmp_path / "trials")
    baseline = tmp_path / "base.cpp"
    baseline.write_text("baseline")
    manager.init("k", baseline, required_profile_groups=("ComputeBasic", "EuStallSampling"))
    source = tmp_path / "kernel.cpp"
    source.write_text("optimized")
    trial = manager.save_trial("k", source)
    manager.record_result("k", trial, correctness="pass", speedup=1.1)

    assert trial == "t0"
    with pytest.raises(ValueError, match="Required profiling"):
        manager.finalize("k", tmp_path / "output.cpp")


def test_finalize_replaces_a_previous_output(tmp_path):
    """A second finalization leaves only the new winner's files behind."""
    manager = TrialManager(tmp_path / "trials")
    source = tmp_path / "kernel"
    source.mkdir()
    (source / "main.cpp").write_text("baseline")
    manager.init("k", source)
    out = tmp_path / "output"
    out.mkdir()
    (out / "stale.cpp").write_text("from an earlier winner")
    (source / "main.cpp").write_text("candidate")
    trial = manager.save_trial("k", source)
    manager.record_result("k", trial, correctness="pass", speedup=1.1)
    assert manager.finalize("k", out) == trial
    assert sorted(p.name for p in out.iterdir()) == ["main.cpp"]


def test_finalize_refuses_to_replace_a_directory_holding_the_trials(mgr, tmp_path):
    _trial(mgr, tmp_path, "t.cpp", speedup=1.1)
    with pytest.raises(ValueError, match="refusing to replace"):
        mgr.finalize("k", tmp_path)
    assert (tmp_path / "trials").is_dir()


@pytest.mark.parametrize(
    "result,kept",
    [
        ({"speedup": 0.66, "baseline_us": 255.4, "triton_us": 390.2}, False),
        ({"speedup": 1.3, "baseline_us": 255.4, "triton_us": 196.5}, True),
        ({"baseline_us": 255.2, "triton_us": 252.6, "verdict": "INDISTINGUISHABLE"}, True),
    ],
)
def test_the_claude_engine_returns_only_what_finalize_would_keep(tmp_path, result, kept):
    """A session whose best trial is a regression returns no kernel, as finalize writes none."""
    from xe_forge.config import Config
    from xe_forge.engines.claude_engine import ClaudeEngine

    config = Config()
    config.trial.trials_dir = str(tmp_path / "trials")
    m = TrialManager(config.trial.trials_dir)
    baseline = tmp_path / "base.cpp"
    baseline.write_text("// baseline\n")
    m.init("k", baseline)
    _trial(m, tmp_path, "t0.cpp", **result)

    best = ClaudeEngine(config)._best_trial(tmp_path, "k")
    assert (best is not None) is kept
    assert (m.finalize("k", tmp_path / "out.cpp") is not None) is kept
    if kept:
        assert best["code"] == "// t0.cpp\n"
