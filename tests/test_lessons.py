"""The lessons ledger, and the two ways it can go wrong.

A ledger is only worth having if a session can be trusted with it, and the trust
has two halves. It must not overwrite what another session measured -- hence one
file per kernel, read-all/write-one, asserted here rather than hoped for. And it
must not become a place where predictions are stored as findings, which is a
property of what the workspace *says*, so the rendered brief is what these tests
read: a session told it may consult earlier measurements, and not told that a
lesson is not a measurement, has been handed a way to skip benchmarking.
"""

from __future__ import annotations

import pytest

from xe_forge.claude.generator import generate_workspace
from xe_forge.config import Config
from xe_forge.core.lessons import lessons_filename, load_lessons_log


def _workspace(tmp_path, lessons_dir=None, *, name="moe_double_gemm"):
    config = Config()
    config.external.benchmark = "host-bench {trial}"
    config.external.lessons = str(lessons_dir) if lessons_dir else None
    ws = tmp_path / f"ws-{name}"
    generate_workspace(ws, config, name, "// kernel\n", reference_code="x = 1\n")
    return ws


# --------------------------------------------------------------------- loading


def test_no_ledger_is_not_an_error(tmp_path):
    assert load_lessons_log(None, "k") is None
    assert load_lessons_log("", "k") is None


def test_ledger_directory_is_created_and_absolute(tmp_path):
    log = load_lessons_log(tmp_path / "nested" / "lessons", "gemm_n2048")
    assert log is not None
    assert (tmp_path / "nested" / "lessons").is_dir()
    # The session's cwd is the workspace, which is somewhere else entirely.
    assert log.dir.startswith("/") and log.own_file.startswith("/")
    assert log.own_file.endswith("/gemm_n2048.md")


def test_a_file_where_a_directory_was_named_raises(tmp_path):
    path = tmp_path / "lessons"
    path.write_text("not a ledger")
    with pytest.raises(NotADirectoryError):
        load_lessons_log(path, "k")


def test_filename_survives_a_hostile_kernel_name():
    assert lessons_filename("moe/../../etc/passwd") == "moe_.._.._etc_passwd.md"
    assert lessons_filename("...") == "session.md"


# ------------------------------------------------------------------- rendering


def test_workspace_without_ledger_is_unchanged(tmp_path):
    claude_md = (_workspace(tmp_path) / "CLAUDE.md").read_text()
    assert "lessons ledger" not in claude_md.lower()
    assert "Step 5" not in claude_md


def test_ledger_file_is_seeded_with_its_own_format(tmp_path):
    ledger = tmp_path / "lessons"
    _workspace(tmp_path, ledger)
    seeded = (ledger / "moe_double_gemm.md").read_text()
    assert seeded.startswith("# Lessons — `moe_double_gemm`")
    # The shape of an entry lives in the file, not only in the brief.
    assert "One entry per run" in seeded


def test_seeding_never_clobbers_what_a_session_recorded(tmp_path):
    ledger = tmp_path / "lessons"
    _workspace(tmp_path, ledger)
    recorded = (ledger / "moe_double_gemm.md").read_text() + "\n## 2026-09-15 — measured\n"
    (ledger / "moe_double_gemm.md").write_text(recorded)

    _workspace(tmp_path, ledger)  # a second run on the same kernel
    assert (ledger / "moe_double_gemm.md").read_text() == recorded


def test_each_kernel_owns_one_file(tmp_path):
    ledger = tmp_path / "lessons"
    _workspace(tmp_path, ledger, name="kernel_a")
    _workspace(tmp_path, ledger, name="kernel_b")
    assert sorted(p.name for p in ledger.iterdir()) == ["kernel_a.md", "kernel_b.md"]


def test_workspace_with_ledger_says_read_all_write_one(tmp_path):
    ledger = tmp_path / "lessons"
    claude_md = (_workspace(tmp_path, ledger) / "CLAUDE.md").read_text()

    assert str(ledger.resolve()) in claude_md
    assert str((ledger / "moe_double_gemm.md").resolve()) in claude_md
    assert "including the files for other kernels" in claude_md
    assert "only by appending" in claude_md
    assert "Delete nothing" in claude_md


def test_workspace_with_ledger_states_the_boundary(tmp_path):
    claude_md = (_workspace(tmp_path, tmp_path / "lessons") / "CLAUDE.md").read_text()

    assert "never a substitute for one" in claude_md
    assert "no prediction" in claude_md
    # Rule 3 survives: benchmark is still the only source of a timing.
    assert "`xe-forge-skill benchmark` is the\n   only source of a timing" in claude_md


def test_recording_happens_even_when_the_run_went_badly(tmp_path):
    claude_md = (_workspace(tmp_path, tmp_path / "lessons") / "CLAUDE.md").read_text()
    assert "This step runs even when the run went badly." in claude_md
    assert "If this run measured nothing" in claude_md


def test_the_ledger_entry_is_the_only_new_writable_file(tmp_path):
    ledger = tmp_path / "lessons"
    claude_md = (_workspace(tmp_path, ledger) / "CLAUDE.md").read_text()
    assert "is exactly one exception, and no others" in claude_md
    assert "never write to any other file under" in claude_md


def test_ledger_and_dataset_render_together(tmp_path):
    """Both seams on at once: two writable files, and both rules numbered."""
    import json

    record = tmp_path / "kernel.fib.json"
    record.write_text(json.dumps({"dataset": str(tmp_path / "trace-set")}))

    config = Config()
    config.external.benchmark = "host-bench {trial}"
    config.external.dataset_record = str(record)
    config.external.lessons = str(tmp_path / "lessons")
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "moe_double_gemm", "// kernel\n")
    claude_md = (ws / "CLAUDE.md").read_text()

    assert "are exactly two exceptions, and no others" in claude_md
    assert "7. **Specialize on the distribution" in claude_md
    assert "8. **A lesson is a record" in claude_md


def test_ledger_rule_is_rule_seven_without_a_dataset(tmp_path):
    claude_md = (_workspace(tmp_path, tmp_path / "lessons") / "CLAUDE.md").read_text()
    assert "7. **A lesson is a record" in claude_md
    assert "8. **" not in claude_md


def test_slash_command_mirrors_the_workspace(tmp_path):
    ledger = tmp_path / "lessons"
    cmd = (_workspace(tmp_path, ledger) / ".claude" / "commands" / "optimize-kernel.md").read_text()
    assert str(ledger.resolve()) in cmd
    assert str((ledger / "moe_double_gemm.md").resolve()) in cmd
    assert "no number you did not measure" in cmd


def test_kernel_repo_baseline_is_a_copy_not_a_seed(tmp_path):
    config = Config()
    config.device_config.dsl = "sycl"
    config.external.kernel_repo = str(tmp_path / "repo")
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "gemm", "", reference_code="x = 1\n")
    assert not (ws / "test_kernels" / "gemm.cpp").exists()
    claude_md = (ws / "CLAUDE.md").read_text()
    assert "the baseline is that kernel" in claude_md
    assert "write none of them yourself" in claude_md
    assert "never the op's own kernel headers" in claude_md
    assert "from-scratch task" not in claude_md
