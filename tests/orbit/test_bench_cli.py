"""
The `orbit-bench` standalone CLI: JSON contract and exit codes (plan §5.4, §24 Tier C).

Real subprocess runs use `sys.executable -c` with trivial bodies so the suite stays
fast and CPU-only. The compare tests construct documents synthetically rather than
timing two commands with different sleeps — timing-based fixtures would test the
scheduler, not the arithmetic.
"""

from __future__ import annotations

import json
import sys

import pytest

from xe_forge.orbit import stats
from xe_forge.orbit.bench import cli
from xe_forge.orbit.models import SCHEMA_VERSION


def _run_main(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _document(samples: list[float], **overrides) -> dict:
    """A synthetic run document with the same shape `cmd_run` emits."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "tool": "orbit-bench",
        "command": ["synthetic"],
        "repetitions": len(samples),
        "warmup": 1,
        "wall_time": stats.estimate(samples, unit="s").model_dump(mode="json"),
        "minimum_detectable_effect_percent": None,
        "exit_codes": [0] * len(samples),
        "environment": {"hostname": "test", "python": "3", "platform": "test"},
        "valid": True,
        "decision_grade": len(samples) >= cli.DECISION_MIN_REPETITIONS,
    }
    doc.update(overrides)
    return doc


def _write(tmp_path, name: str, doc: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


class TestRun:
    def test_emits_valid_json_with_n_repetitions_and_warmup_discarded(self, capsys, tmp_path):
        """Warmup runs execute (the marker file proves it) but yield no samples."""
        marker = tmp_path / "runs.txt"
        body = f"open(r'{marker}', 'a').write('.')"
        code, out, _ = _run_main(
            capsys,
            ["run", "--repetitions", "5", "--warmup", "2", "--", sys.executable, "-c", body],
        )
        assert code == 0
        doc = json.loads(out)
        assert doc["schema_version"] == SCHEMA_VERSION
        assert doc["tool"] == "orbit-bench"
        assert doc["command"] == [sys.executable, "-c", body]
        assert doc["repetitions"] == 5
        assert doc["warmup"] == 2
        assert doc["valid"] is True
        assert doc["decision_grade"] is True
        assert doc["exit_codes"] == [0] * 5
        assert doc["wall_time"]["n"] == 5
        assert len(doc["wall_time"]["samples"]) == 5
        assert doc["wall_time"]["unit"] == "s"
        assert doc["wall_time"]["ci95_low"] <= doc["wall_time"]["mean"]
        assert doc["minimum_detectable_effect_percent"] is not None
        # 7 executions happened; only 5 became samples.
        assert marker.read_text() == "." * 7

    def test_failed_repetition_marks_run_invalid_and_names_it(self, capsys):
        code, out, _ = _run_main(
            capsys,
            [
                "run",
                "--repetitions",
                "5",
                "--warmup",
                "0",
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(3)",
            ],
        )
        assert code == 1
        doc = json.loads(out)
        assert doc["valid"] is False
        assert "repetition 0" in doc["reason"]
        assert "exit 3" in doc["reason"]
        assert "5 of 5 repetitions failed" in doc["reason"]
        assert doc["exit_codes"] == [3] * 5
        # The samples are still present for inspection — but flagged unusable.
        assert len(doc["wall_time"]["samples"]) == 5

    def test_json_flag_writes_to_file_not_stdout(self, capsys, tmp_path):
        target = tmp_path / "measurement.json"
        code, out, err = _run_main(
            capsys,
            [
                "run",
                "--repetitions",
                "2",
                "--warmup",
                "0",
                "--json",
                str(target),
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
        )
        assert code == 0
        assert out == ""
        assert str(target) in err
        doc = json.loads(target.read_text())
        assert doc["wall_time"]["n"] == 2

    def test_fewer_than_five_repetitions_is_not_decision_grade(self, capsys):
        code, out, _ = _run_main(
            capsys,
            ["run", "--repetitions", "3", "--warmup", "0", "--", sys.executable, "-c", "pass"],
        )
        assert code == 0  # measuring succeeded; it is the grade that is limited
        doc = json.loads(out)
        assert doc["valid"] is True
        assert doc["decision_grade"] is False
        assert "3 repetitions" in doc["decision_grade_reason"]
        assert str(cli.DECISION_MIN_REPETITIONS) in doc["decision_grade_reason"]

    def test_timeout_is_a_failure_not_a_fast_run(self, capsys):
        code, out, _ = _run_main(
            capsys,
            [
                "run",
                "--repetitions",
                "1",
                "--warmup",
                "0",
                "--timeout",
                "0.3",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
        )
        assert code == 1
        doc = json.loads(out)
        assert doc["valid"] is False
        assert "timed out" in doc["reason"]
        assert doc["exit_codes"] == [None]

    def test_missing_command_is_a_usage_error(self, capsys):
        code, _, err = _run_main(capsys, ["run", "--repetitions", "5"])
        assert code == 2
        assert "no workload command" in err

    def test_run_help_states_the_declared_warmup_rule(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "DISCARDED" in out
        assert "declared" in out
        assert "17.5" in out


class TestCompare:
    def test_faster_candidate_is_accepted(self, capsys, tmp_path):
        base = _write(tmp_path, "a.json", _document([2.00, 2.01, 1.99, 2.00, 2.02]))
        cand = _write(tmp_path, "b.json", _document([1.00, 1.01, 0.99, 1.00, 1.02]))
        code, out, _ = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_ACCEPT
        assert "ACCEPT" in out
        assert "95% CI" in out
        assert "MDE" in out

    def test_slower_candidate_is_rejected(self, capsys, tmp_path):
        base = _write(tmp_path, "a.json", _document([1.00, 1.01, 0.99, 1.00, 1.02]))
        cand = _write(tmp_path, "b.json", _document([2.00, 2.01, 1.99, 2.00, 2.02]))
        code, out, _ = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_REJECT
        assert "REJECT" in out

    def test_overlapping_samples_are_inconclusive_not_reject(self, capsys, tmp_path):
        noisy = [1.00, 1.42, 0.68, 1.31, 0.75, 1.28]
        base = _write(tmp_path, "a.json", _document(noisy))
        cand = _write(tmp_path, "b.json", _document([x * 1.01 for x in noisy]))
        code, out, _ = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_INCONCLUSIVE
        assert "INCONCLUSIVE" in out

    def test_zero_baseline_is_invalid_exit_3(self, capsys, tmp_path):
        base = _write(tmp_path, "a.json", _document([0.0] * 5))
        cand = _write(tmp_path, "b.json", _document([1.00, 1.01, 0.99, 1.00, 1.02]))
        code, out, _ = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_INVALID
        assert "INVALID" in out

    def test_refuses_document_marked_invalid(self, capsys, tmp_path):
        bad = _document([1.0] * 5, valid=False, reason="repetition 2 of 5: exit 1")
        base = _write(tmp_path, "a.json", bad)
        cand = _write(tmp_path, "b.json", _document([1.0] * 5))
        code, _, err = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_REFUSED
        assert "marked invalid" in err
        assert "repetition 2" in err

    def test_refuses_non_decision_grade_document(self, capsys, tmp_path):
        base = _write(tmp_path, "a.json", _document([1.0] * 5))
        cand = _write(tmp_path, "b.json", _document([1.0, 1.01, 0.99]))
        code, _, err = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_REFUSED
        assert "not decision grade" in err

    def test_refuses_schema_major_mismatch(self, capsys, tmp_path):
        future = _document([1.0] * 5, schema_version="2.0")
        base = _write(tmp_path, "a.json", future)
        cand = _write(tmp_path, "b.json", _document([1.0] * 5))
        code, _, err = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_REFUSED
        assert "major" in err
        assert "2.0" in err

    def test_refuses_document_without_samples(self, capsys, tmp_path):
        base = _write(tmp_path, "a.json", {"schema_version": SCHEMA_VERSION, "valid": True})
        cand = _write(tmp_path, "b.json", _document([1.0] * 5))
        code, _, err = _run_main(capsys, ["compare", "--baseline", base, "--candidate", cand])
        assert code == cli.EXIT_REFUSED
        assert "wall_time.samples" in err
