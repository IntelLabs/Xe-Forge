"""Who wrote a trial's numbers, and what finalize does with a claim it cannot check.

A session that runs `benchmark` and then types the result into `trial result` is the
source of the number the tree ranks on. `benchmark --kernel-name --trial-id` records
what it measured itself, marked `measured`; `finalize --require-measured` only accepts
those, so a correctness verdict or a speedup the model wrote cannot ship a kernel.
"""

from __future__ import annotations

import argparse
import json

import pytest

from xe_forge.core.trial_manager import TrialManager
from xe_forge.skills import benchmark


@pytest.fixture
def tree(tmp_path):
    baseline = tmp_path / "k.py"
    baseline.write_text("baseline\n")
    trial = tmp_path / "cand.py"
    trial.write_text("candidate\n")
    mgr = TrialManager(tmp_path / "trials")
    mgr.init("k", baseline)
    tid = mgr.save_trial("k", trial, strategy="s")
    return mgr, baseline, tmp_path / "trials" / "k" / f"{tid}.py", tid


def _args(tmp_path, baseline, optimized, tid, **overrides):
    fields = {
        "baseline": str(baseline),
        "optimized": str(optimized),
        "spec": None,
        "reference": None,
        "variant": None,
        "baseline_us": None,
        "device": "xpu",
        "dsl": "triton",
        "triton_baseline": False,
        "external_benchmark": "fake",
        "builtin_benchmark": False,
        "kernel_name": "k",
        "trial_id": tid,
        "trials_dir": str(tmp_path / "trials"),
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _run_with_output(monkeypatch, args, printed: str, code: int = 0) -> int:
    def fake_external(_args, _template):
        print(printed)
        return code

    monkeypatch.setattr(benchmark, "_run_external", fake_external)
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(args)
    return exit_info.value.code


def test_parse_outcome_reads_the_printed_contract():
    text = (
        "Correctness: PASSED\n"
        "Performance: baseline_us=120.00, kernel_us=80.00, speedup=1.50x\n"
        "VERDICT: OK\n"
    )
    assert benchmark.parse_outcome(text) == {
        "correctness": "pass",
        "baseline_us": 120.0,
        "triton_us": 80.0,
        "speedup": 1.5,
        "verdict": "OK",
    }
    gated = "Correctness: PASSED\nPerformance: baseline_us=1.00, kernel_us=1.00, speedup=none\n"
    assert "speedup" not in benchmark.parse_outcome(gated)
    assert benchmark.parse_outcome("Using cached baseline\n") is None


def test_benchmark_records_measured_result(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    code = _run_with_output(
        monkeypatch,
        _args(tmp_path, baseline, trial_file, tid),
        "Correctness: PASSED\nPerformance: baseline_us=100.00, kernel_us=50.00, speedup=2.00x",
    )
    assert code == 0
    trial = mgr.get_best("k")
    assert trial["id"] == tid
    assert trial["source"] == "measured"
    assert trial["speedup"] == 2.0
    assert mgr.finalize("k", tmp_path / "out.py", require_measured=True) == tid


def test_benchmark_refuses_a_file_that_is_not_the_saved_trial(tmp_path, tree, monkeypatch):
    _, baseline, _, tid = tree
    other = tmp_path / "other.py"
    other.write_text("something else\n")
    with pytest.raises(SystemExit, match="does not match the saved trial"):
        benchmark.run(_args(tmp_path, baseline, other, tid))


def test_reported_result_is_not_finalized_when_measurement_is_required(tmp_path, tree):
    mgr, _, _, tid = tree
    mgr.record_result("k", tid, correctness="pass", speedup=9.0, baseline_us=1, triton_us=0.1)
    assert mgr.get_best("k")["source"] == "reported"
    assert mgr.finalize("k", tmp_path / "out.py", require_measured=True) is None
    # Without the requirement the old behaviour is unchanged.
    assert mgr.finalize("k", tmp_path / "out.py") == tid


def test_a_typed_result_over_a_measured_one_taints_it(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    _run_with_output(
        monkeypatch,
        _args(tmp_path, baseline, trial_file, tid),
        "Correctness: FAILED\nOutput:\nmismatch",
        code=1,
    )
    mgr.record_result("k", tid, correctness="pass", speedup=3.0, baseline_us=3, triton_us=1)
    assert mgr.finalize("k", tmp_path / "out.py", require_measured=True) is None


def test_measurement_against_a_different_baseline_is_refused(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    ok = "Correctness: PASSED\nPerformance: baseline_us=100.00, kernel_us=50.00, speedup=2.00x"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, trial_file, tid), ok)

    second = tmp_path / "cand2.py"
    second.write_text("candidate 2\n")
    tid2 = mgr.save_trial("k", second, parent=tid)
    saved2 = tmp_path / "trials" / "k" / f"{tid2}.py"
    baseline.write_text("a different baseline\n")
    code = _run_with_output(monkeypatch, _args(tmp_path, baseline, saved2, tid2), ok)
    assert code == 1
    state = json.loads((tmp_path / "trials" / "k" / "state.json").read_text())
    assert state["trials"][tid2]["correctness"] is None


def test_state_written_before_sources_existed_still_loads(tmp_path, tree):
    mgr, _, _, tid = tree
    path = tmp_path / "trials" / "k" / "state.json"
    state = json.loads(path.read_text())
    state["trials"][tid].update(correctness="pass", speedup=1.2, baseline_us=1.2, triton_us=1.0)
    state["best_trial"] = tid
    path.write_text(json.dumps(state))
    assert mgr.get_best("k")["id"] == tid
    assert mgr.finalize("k", tmp_path / "out.py") == tid
    assert mgr.finalize("k", tmp_path / "out2.py", require_measured=True) is None


def test_a_copy_of_the_baseline_is_not_finalized_as_a_result(tmp_path, tree, monkeypatch):
    mgr, baseline, _, _ = tree
    copy_id = mgr.save_trial("k", baseline, strategy="baseline copy")
    saved = tmp_path / "trials" / "k" / f"{copy_id}.py"
    parity = "Correctness: PASSED\nPerformance: baseline_us=100.00, kernel_us=100.00, speedup=1.00x"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, saved, copy_id), parity)
    assert mgr.get_best("k")["id"] == copy_id
    assert mgr.finalize("k", tmp_path / "out.py", require_measured=True) is None


def test_a_failed_run_does_not_pin_the_baseline(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    _run_with_output(
        monkeypatch,
        _args(tmp_path, baseline, trial_file, tid),
        "Correctness: FAILED\nOutput:\nbaseline did not compile",
        code=1,
    )
    baseline.write_text("repaired baseline\n")
    ok = "Correctness: PASSED\nPerformance: baseline_us=100.00, kernel_us=50.00, speedup=2.00x"
    assert _run_with_output(monkeypatch, _args(tmp_path, baseline, trial_file, tid), ok) == 0
    assert mgr.get_best("k")["id"] == tid


def test_a_baseline_repaired_after_init_is_still_not_a_result(tmp_path, tree, monkeypatch):
    mgr, baseline, _, _ = tree
    baseline.write_text("repaired baseline\n")
    copy_id = mgr.save_trial("k", baseline, strategy="repair baseline")
    saved = tmp_path / "trials" / "k" / f"{copy_id}.py"
    parity = "Correctness: PASSED\nPerformance: baseline_us=100.00, kernel_us=99.00, speedup=1.01x"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, saved, copy_id), parity)
    assert mgr.finalize("k", tmp_path / "out.py", require_measured=True) is None


def test_a_baseline_copy_is_not_a_measured_attempt(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    copy_id = mgr.save_trial("k", baseline, strategy="control")
    saved = tmp_path / "trials" / "k" / f"{copy_id}.py"
    ok = "Correctness: PASSED\nPerformance: baseline_us=10.00, kernel_us=10.00, speedup=none\nVERDICT: INDISTINGUISHABLE"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, saved, copy_id), ok)
    assert not mgr.has_measured_attempt("k")
    _run_with_output(monkeypatch, _args(tmp_path, baseline, trial_file, tid), ok)
    assert mgr.has_measured_attempt("k")


def test_best_measured_skips_baseline_copies(tmp_path, tree, monkeypatch):
    mgr, baseline, trial_file, tid = tree
    copy_id = mgr.save_trial("k", baseline, strategy="control")
    saved = tmp_path / "trials" / "k" / f"{copy_id}.py"
    fast = "Correctness: PASSED\nPerformance: baseline_us=10.00, kernel_us=5.00, speedup=2.00x"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, saved, copy_id), fast)
    assert mgr.best_measured("k") is None
    slower = "Correctness: PASSED\nPerformance: baseline_us=10.00, kernel_us=8.00, speedup=1.25x"
    _run_with_output(monkeypatch, _args(tmp_path, baseline, trial_file, tid), slower)
    assert mgr.best_measured("k")["id"] == tid
