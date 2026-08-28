"""The agent arena — isolation, resume, honesty (plan §5.3 lesson 2, §5.4).

Everything here is CPU-only with stub runners: what is under test is the arena's own
contract — the shared tasks stay untouched, pairs resume instead of re-spending their
budget, one crash stops nothing else, and the leaderboard never turns an unmeasured
quantity into a number — not any engine's ability to optimize.
"""

from __future__ import annotations

import argparse
import json
import shutil

import pytest

from xe_forge.orbit.arena import (
    ArenaError,
    Contestant,
    ContestantResult,
    build_contestants,
    contestant_available,
    discover_tasks,
    run_arena,
)


def _make_task_dir(root, name, arena_yaml=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "kernel.py").write_text("def optimized_kernel(x):\n    return x\n")
    (directory / "spec.yaml").write_text("inputs: {}\n")
    (directory / "kernel_pytorch.py").write_text("def ref(x):\n    return x\n")
    if arena_yaml is not None:
        (directory / "arena.yaml").write_text(arena_yaml)
    return directory


def _scripted(outcomes, calls=None):
    """A runner with scripted results per (contestant, task); optionally records calls."""

    def runner(contestant, task, workspace):
        if calls is not None:
            calls.append((contestant.name, task.task_id))
        spec = outcomes[(contestant.name, task.task_id)]
        if isinstance(spec, Exception):
            raise spec
        succeeded, train, heldout = spec
        return ContestantResult(
            task_id=task.task_id,
            contestant=contestant.name,
            succeeded=succeeded,
            train_speedup=train,
            heldout_speedups=dict(heldout or {}),
        )

    return runner


class TestDiscoverTasks:
    def test_candidate_subdirs_become_tasks_and_others_are_skipped(self, tmp_path):
        _make_task_dir(tmp_path, "t_b")
        _make_task_dir(tmp_path, "t_a")
        (tmp_path / "not_a_task").mkdir()  # no kernel.py, no spec.yaml
        (tmp_path / "half_a_task").mkdir()
        (tmp_path / "half_a_task" / "kernel.py").write_text("x = 1\n")  # spec.yaml missing
        (tmp_path / ".arena").mkdir()  # the arena's own workspace is never a task
        (tmp_path / "README.md").write_text("plain file\n")

        tasks = discover_tasks(tmp_path)
        assert [t.task_id for t in tasks] == ["t_a", "t_b"]
        assert all(t.candidate_dir.is_dir() for t in tasks)

    def test_defaults_without_arena_yaml_carry_an_honest_note(self, tmp_path):
        _make_task_dir(tmp_path, "t0")
        (task,) = discover_tasks(tmp_path)
        assert task.train_variant == "bench-gpu"
        assert task.heldout_variants == []
        assert any("generalization" in note for note in task.notes)

    def test_arena_yaml_sets_train_and_heldout(self, tmp_path):
        _make_task_dir(
            tmp_path,
            "t0",
            arena_yaml="train: bench-xpu\nheldout:\n  - decode-8\n  - prefill-512\n",
        )
        (task,) = discover_tasks(tmp_path)
        assert task.train_variant == "bench-xpu"
        assert task.heldout_variants == ["decode-8", "prefill-512"]
        assert task.notes == []

    def test_train_variant_listed_as_heldout_is_rejected(self, tmp_path):
        # A shape the contestant optimizes against cannot also measure generalization;
        # accepting it would produce a near-zero "gap" that answers the wrong question.
        _make_task_dir(tmp_path, "t0", arena_yaml="train: decode-8\nheldout: [decode-8]\n")
        with pytest.raises(ArenaError, match="held-out"):
            discover_tasks(tmp_path)


class TestRunArena:
    def test_default_runner_never_touches_the_original_candidate(self, tmp_path):
        task_dir = _make_task_dir(tmp_path / "tasks", "t0", arena_yaml="heldout: [decode-8]\n")
        before = {p.relative_to(task_dir): p.read_bytes() for p in sorted(task_dir.rglob("*"))}
        tasks = discover_tasks(tmp_path / "tasks")
        # dry_run keeps the default runner's copy-and-optimize path CPU-only.
        contestant = Contestant(name="dry", config={"engine": "dspy", "dry_run": True})

        report = run_arena(tasks, [contestant], tmp_path / "arena")

        after = {p.relative_to(task_dir): p.read_bytes() for p in sorted(task_dir.rglob("*"))}
        assert after == before
        (result,) = report.results
        assert result.succeeded
        assert result.train_speedup is None  # a dry run measures nothing
        assert result.heldout_speedups == {"decode-8": None}
        assert (result.workspace / "candidate" / "kernel.py").is_file()

    def test_resume_loads_persisted_pairs_and_reports_the_count(self, tmp_path):
        for name in ("t0", "t1"):
            _make_task_dir(tmp_path / "tasks", name)
        tasks = discover_tasks(tmp_path / "tasks")
        outcomes = {
            ("a", "t0"): (True, 1.2, {}),
            ("a", "t1"): (True, 1.4, {}),
            ("b", "t0"): (False, None, {}),
            ("b", "t1"): (True, 1.1, {}),
        }
        calls: list = []
        contestants = [
            Contestant(name="a", runner=_scripted(outcomes, calls)),
            Contestant(name="b", runner=_scripted(outcomes, calls)),
        ]

        first = run_arena(tasks, contestants, tmp_path / "arena")
        assert first.resumed == 0
        assert len(calls) == 4
        for contestant, task in outcomes:
            assert (tmp_path / "arena" / contestant / task / "result.json").is_file()

        second = run_arena(tasks, contestants, tmp_path / "arena")
        assert len(calls) == 4  # nothing re-ran
        assert second.resumed == 4
        assert "4 resumed" in second.format()
        assert {(r.contestant, r.task_id, r.train_speedup) for r in second.results} == {
            (r.contestant, r.task_id, r.train_speedup) for r in first.results
        }

    def test_no_resume_reruns_every_pair(self, tmp_path):
        _make_task_dir(tmp_path / "tasks", "t0")
        tasks = discover_tasks(tmp_path / "tasks")
        calls: list = []
        contestant = Contestant(name="a", runner=_scripted({("a", "t0"): (True, 1.2, {})}, calls))

        run_arena(tasks, [contestant], tmp_path / "arena")
        report = run_arena(tasks, [contestant], tmp_path / "arena", resume=False)
        assert len(calls) == 2
        assert report.resumed == 0

    def test_a_result_that_belongs_to_another_pair_is_rerun_not_trusted(self, tmp_path):
        # A result.json copied into the wrong directory must not resume as somebody
        # else's score — the identity check runs before anything is believed.
        _make_task_dir(tmp_path / "tasks", "t0")
        _make_task_dir(tmp_path / "tasks", "t1")
        tasks = discover_tasks(tmp_path / "tasks")
        outcomes = {("a", "t0"): (True, 1.2, {}), ("a", "t1"): (True, 1.4, {})}
        calls: list = []
        contestant = Contestant(name="a", runner=_scripted(outcomes, calls))

        run_arena(tasks, [contestant], tmp_path / "arena")
        misplaced = tmp_path / "arena" / "a" / "t1" / "result.json"
        shutil.copyfile(tmp_path / "arena" / "a" / "t0" / "result.json", misplaced)

        calls.clear()
        report = run_arena(tasks, [contestant], tmp_path / "arena")
        assert calls == [("a", "t1")]
        assert report.resumed == 1

    def test_one_contestant_crashing_stops_nothing_else(self, tmp_path):
        for name in ("t0", "t1"):
            _make_task_dir(tmp_path / "tasks", name)
        tasks = discover_tasks(tmp_path / "tasks")
        outcomes = {
            ("boom", "t0"): RuntimeError("kaboom"),
            ("boom", "t1"): (True, 1.2, {}),
            ("ok", "t0"): (True, 1.1, {}),
            ("ok", "t1"): (True, 1.3, {}),
        }
        contestants = [
            Contestant(name="boom", runner=_scripted(outcomes)),
            Contestant(name="ok", runner=_scripted(outcomes)),
        ]

        report = run_arena(tasks, contestants, tmp_path / "arena")
        assert len(report.results) == 4
        failed = next(r for r in report.results if r.contestant == "boom" and r.task_id == "t0")
        assert not failed.succeeded
        assert "RuntimeError: kaboom" in failed.error
        persisted = json.loads((tmp_path / "arena" / "boom" / "t0" / "result.json").read_text())
        assert persisted["succeeded"] is False
        assert "boom/t0" in report.format()


class TestArenaReport:
    def _report(self, tmp_path, outcomes, arena_yaml=None, tasks=("t0", "t1")):
        for name in tasks:
            _make_task_dir(tmp_path / "tasks", name, arena_yaml=arena_yaml)
        discovered = discover_tasks(tmp_path / "tasks")
        contestants = [
            Contestant(name=name, runner=_scripted(outcomes))
            for name in sorted({contestant for contestant, _ in outcomes})
        ]
        return run_arena(discovered, contestants, tmp_path / "arena")

    def test_leaderboard_shows_solved_counts_and_mean_speedups(self, tmp_path):
        report = self._report(
            tmp_path,
            {
                ("dspy", "t0"): (True, 1.2, {}),
                ("dspy", "t1"): (True, 1.8, {}),
                ("other", "t0"): (True, 1.1, {}),
                ("other", "t1"): (False, None, {}),
            },
        )
        out = report.format()
        assert "2/2" in out and "1/2" in out
        assert "1.50x" in out  # mean over dspy's solved tasks
        assert report.summary()["contestants"]["dspy"]["train_mean_speedup"] == pytest.approx(1.5)

    def test_unmeasured_heldout_is_marked_never_a_number(self, tmp_path):
        report = self._report(
            tmp_path,
            {("dspy", "t0"): (True, 1.5, {"decode-8": None})},
            arena_yaml="heldout: [decode-8]\n",
            tasks=("t0",),
        )
        out = report.format()
        assert "unmeasured" in out
        assert "0.00" not in out  # None must never surface as a numeric heldout mean
        stats = report.summary()["contestants"]["dspy"]
        assert stats["heldout_mean_speedup"] is None
        assert stats["heldout_unmeasured"] == 1
        assert stats["generalization_gap"] is None

    def test_generalization_gap_only_where_heldout_was_measured(self, tmp_path):
        report = self._report(
            tmp_path,
            {
                ("meas", "t0"): (True, 1.5, {"decode-8": 1.2, "prefill-512": 1.4}),
                ("unm", "t0"): (True, 1.4, {"decode-8": None, "prefill-512": None}),
            },
            arena_yaml="heldout: [decode-8, prefill-512]\n",
            tasks=("t0",),
        )
        contestants = report.summary()["contestants"]
        assert contestants["meas"]["heldout_mean_speedup"] == pytest.approx(1.3)
        assert contestants["meas"]["generalization_gap"] == pytest.approx(0.2)
        assert contestants["unm"]["generalization_gap"] is None
        assert "+0.20" in report.format()

    def test_a_measured_mean_is_never_ranked_against_an_unmeasured_one(self, tmp_path):
        report = self._report(
            tmp_path,
            {
                # "async" solved more tasks but measured nothing (the fire-and-forget
                # shape); it must sit below the ranked rows, not above them.
                ("async", "t0"): (True, None, {}),
                ("async", "t1"): (True, None, {}),
                ("dspy", "t0"): (True, 1.1, {}),
                ("dspy", "t1"): (False, None, {}),
            },
        )
        rows = {
            line.split()[1]: line.split()[0]
            for line in report.format().splitlines()
            if line.split() and line.split()[0] in {"1", "2", "-"}
        }
        assert rows["dspy"] == "1"
        assert rows["async"] == "-"

    def test_summary_is_json_serializable_and_complete(self, tmp_path):
        report = self._report(tmp_path, {("dspy", "t0"): (True, 1.2, {})}, tasks=("t0",))
        report.skipped["claude"] = "the `claude` binary is not on PATH"
        payload = json.loads(json.dumps(report.summary()))
        assert payload["tasks"] == 1
        assert payload["pairs"] == 1
        assert payload["resumed"] == 0
        assert payload["skipped"] == {"claude": "the `claude` binary is not on PATH"}
        assert payload["results"][0]["task_id"] == "t0"


class TestBuildContestants:
    def test_unavailable_contestant_is_skipped_with_reason(self, monkeypatch):
        import xe_forge.orbit.arena as arena_mod

        def fake_available(name):
            if name == "claude":
                return False, "the `claude` binary is not on PATH"
            return True, ""

        monkeypatch.setattr(arena_mod, "contestant_available", fake_available)
        contestants, skipped = build_contestants(["dspy", "claude"])
        assert [c.name for c in contestants] == ["dspy"]
        assert contestants[0].config == {"engine": "dspy"}
        assert skipped == {"claude": "the `claude` binary is not on PATH"}

    def test_claude_availability_checks_the_binary_itself(self, monkeypatch):
        import xe_forge.orbit.arena as arena_mod

        monkeypatch.setattr(arena_mod.shutil, "which", lambda _: None)
        ok, reason = contestant_available("claude")
        assert not ok
        assert "PATH" in reason


class TestCmdArena:
    def _args(self, task_root, **over):
        base = {
            "task_root": str(task_root),
            "contestants": "stub",
            "arena_dir": None,
            "no_resume": False,
            "json": False,
        }
        base.update(over)
        return argparse.Namespace(**base)

    def _stub_contestants(self, monkeypatch, outcomes):
        import xe_forge.orbit.arena as arena_mod

        def fake_build(names):
            return [Contestant(name="stub", runner=_scripted(outcomes))], {}

        monkeypatch.setattr(arena_mod, "build_contestants", fake_build)

    def test_happy_path_prints_leaderboard_and_persists_summary(
        self, monkeypatch, tmp_path, capsys
    ):
        from xe_forge.orbit.cli import cmd_arena

        root = tmp_path / "tasks"
        _make_task_dir(root, "t0")
        self._stub_contestants(monkeypatch, {("stub", "t0"): (True, 1.3, {})})

        assert cmd_arena(self._args(root)) == 0
        out = capsys.readouterr().out
        assert "agent arena:" in out
        assert "1/1" in out
        summary = json.loads((root / ".arena" / "summary.json").read_text())
        assert summary["contestants"]["stub"]["train_mean_speedup"] == pytest.approx(1.3)

    def test_a_root_without_tasks_is_exit_1(self, tmp_path, capsys):
        from xe_forge.orbit.cli import cmd_arena

        root = tmp_path / "tasks"
        root.mkdir()
        assert cmd_arena(self._args(root)) == 1
        assert "no tasks" in capsys.readouterr().out

    def test_no_available_contestant_is_exit_1_with_reasons(self, monkeypatch, tmp_path, capsys):
        import xe_forge.orbit.arena as arena_mod
        from xe_forge.orbit.cli import cmd_arena

        root = tmp_path / "tasks"
        _make_task_dir(root, "t0")
        monkeypatch.setattr(
            arena_mod, "contestant_available", lambda name: (False, f"{name} is unavailable")
        )

        assert cmd_arena(self._args(root, contestants="dspy,claude")) == 1
        out = capsys.readouterr().out
        assert "skipped contestant: dspy (dspy is unavailable)" in out
        assert "skipped contestant: claude (claude is unavailable)" in out
