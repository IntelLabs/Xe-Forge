"""The workspace tools a DSPy agent works through, and what they refuse.

The agent may read the workspace and what it links to, write its own files, and run
`xe-forge-skill`. It may not reach outside, write what a skill owns, or run anything but a skill. What a
trial is judged against is pinned by digest in the skills, for every engine alike.
"""

from __future__ import annotations

import pytest

from xe_forge.agents.workspace_tools import WorkspacePolicy, make_workspace_tools


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "test_kernels").mkdir(parents=True)
    (root / "test_kernels" / "k.py").write_text("baseline = 1\n")
    (root / "test_kernels" / "k_pytorch.py").write_text("reference = 1\n")
    (root / "test_kernels" / "k.yaml").write_text("inputs: {}\n")
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "pattern.yaml").write_text("patterns: []\n")
    (root / "knowledge_base").symlink_to(kb)
    (tmp_path / "secret.txt").write_text("outside\n")
    policy = WorkspacePolicy.for_workspace(root, skill_timeout_s=120)
    return root, {t.name: t for t in make_workspace_tools(policy)}


def test_reads_inside_and_through_the_knowledge_base_link(ws):
    _, tools = ws
    assert "baseline = 1" in tools["read_file"](path="test_kernels/k.py")
    assert "patterns" in tools["read_file"](path="knowledge_base/pattern.yaml")
    assert "pattern.yaml" in tools["list_files"](path="knowledge_base")
    assert "test_kernels/k_pytorch.py:1:" in tools["search"](regex="reference", path=".")


@pytest.mark.parametrize(
    "path", ["../secret.txt", "/etc/passwd", "knowledge_base/../../secret.txt"]
)
def test_reads_outside_are_refused(ws, path):
    _, tools = ws
    with pytest.raises(PermissionError):
        tools["read_file"](path=path)


@pytest.mark.parametrize(
    "path",
    [
        "trials/k/state.json",  # the trial tree a skill owns
        "output/k.py",
        "knowledge_base/pattern.yaml",  # through a symlink, out of the workspace
        "../escape.py",
        ".",
    ],
)
def test_skill_owned_and_escaping_writes_are_refused(ws, path):
    _, tools = ws
    with pytest.raises(PermissionError):
        tools["write_file"](path=path, content="x")


def test_integration_repositories_are_readable(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "caller.py").write_text("op(x)\n")
    policy = WorkspacePolicy.for_workspace(root, integration_repos=[str(vllm)])
    tools = {t.name: t for t in make_workspace_tools(policy)}
    assert "op(x)" in tools["read_file"](path=str(vllm / "caller.py"))


def test_writes_its_own_files(ws):
    root, tools = ws
    assert "wrote" in tools["write_file"](path="work/cand.py", content="cand = 1\n")
    assert (root / "work" / "cand.py").read_text() == "cand = 1\n"


def test_only_skills_run_and_never_through_a_shell(ws):
    root, tools = ws
    assert tools["skill"](args="rm -rf .").startswith("refused")
    assert tools["skill"](args="").startswith("refused")
    out = tools["skill"](args="trial status 'k; touch pwned'")
    assert not (root / "pwned").exists()
    assert out.startswith("exit=")


def test_trial_round_trip_through_the_skill_tool(ws):
    root, tools = ws
    tools["write_file"](path="work/cand.py", content="cand = 1\n")
    assert "exit=0" in tools["skill"](args="trial init k test_kernels/k.py")
    out = tools["skill"](args="xe-forge-skill trial save k work/cand.py --strategy 'first try'")
    assert "Saved trial t0" in out
    assert "first try" in tools["skill"](args="trial status k")
    assert (root / "trials" / "k" / "t0.py").exists()


def test_long_output_is_clipped_at_both_ends(tmp_path):
    policy = WorkspacePolicy(root=tmp_path, max_output_chars=100)
    text = "head" + "x" * 1000 + "tail"
    clipped = policy.clip(text)
    assert clipped.startswith("head") and clipped.endswith("tail")
    assert "omitted" in clipped and len(clipped) < 200


def test_given_files_are_not_read_twice(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("policy " * 1000)
    tools = {
        t.name: t
        for t in make_workspace_tools(WorkspacePolicy(root=tmp_path, given=("CLAUDE.md",)))
    }
    assert "already in your instructions" in tools["read_file"](path="CLAUDE.md")


def test_git_directories_are_not_writable(ws):
    _, tools = ws
    with pytest.raises(PermissionError, match="git directory"):
        tools["write_file"](path="upstream/.git/hooks/post-checkout", content="#!/bin/sh\n")


def test_port_back_through_the_skill_tool(ws, tmp_path):
    import subprocess

    root, tools = ws
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "op.py").write_text("SCALE = 1\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-qm", "base"], check=True)

    assert "Cloned" in tools["skill"](args=f"upstream clone {repo}")
    tools["write_file"](path="upstream/pkg/op.py", content="SCALE = 2\n")
    tools["write_file"](path="upstream/pkg/new.py", content="X = 1\n")
    out = tools["skill"](args="upstream patch k --notes 'scale change'")
    assert out.startswith("exit=0"), out
    patch = (root / "output" / "upstream.patch").read_text()
    assert "SCALE = 2" in patch and "pkg/new.py" in patch
    port = (root / "output" / "PORT.txt").read_text()
    assert "py_compile pkg/op.py: ok" in port and "scale change" in port
    assert (repo / "pkg" / "op.py").read_text() == "SCALE = 1\n"  # the repository is untouched


def test_the_lessons_ledger_is_appended_to_never_replaced(tmp_path):
    ledger = tmp_path / "lessons" / "k.md"
    ledger.parent.mkdir()
    ledger.write_text("# earlier session\n")
    root = tmp_path / "ws"
    root.mkdir()
    policy = WorkspacePolicy.for_workspace(root, lessons_file=str(ledger))
    tools = {t.name: t for t in make_workspace_tools(policy)}
    assert "appended" in tools["write_file"](path=str(ledger), content="## this session\n")
    text = ledger.read_text()
    assert text.startswith("# earlier session") and "## this session" in text


def test_a_serving_only_port_patches_every_repository_it_wires_into(ws, tmp_path):
    import subprocess

    root, tools = ws
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    repos = {}
    for name, rel, text in (("kernels", "op.py", "SCALE = 1\n"), ("vllm", "caller.py", "op(x)\n")):
        repo = tmp_path / name
        repo.mkdir()
        (repo / rel).write_text(text)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run([*git, "-C", str(repo), "add", "."], check=True)
        subprocess.run([*git, "-C", str(repo), "commit", "-qm", "base"], check=True)
        repos[name] = repo

    tools["skill"](args=f"upstream clone {repos['kernels']}")
    assert "upstream-vllm/" in tools["skill"](args=f"upstream clone {repos['vllm']} --name vllm")
    tools["write_file"](path="upstream/op.py", content="SCALE = 2\n")
    tools["write_file"](path="upstream-vllm/caller.py", content="op(x, fused=True)\n")
    out = tools["skill"](args="upstream patch k --kind serving-only --notes 'call site changed'")
    assert out.startswith("exit=0"), out
    assert "SCALE = 2" in (root / "output" / "upstream.patch").read_text()
    assert "fused=True" in (root / "output" / "upstream-vllm.patch").read_text()
    port = (root / "output" / "PORT.txt").read_text()
    assert port.startswith("kind: serving-only")
    assert "base commit:" in port and port.count("py_compile") == 2


def test_copy_then_edit_derives_a_trial_without_rewriting_it(ws):
    root, tools = ws
    assert "copied" in tools["copy_file"](src="test_kernels/k.py", dst="work/t1.py")
    assert "edited" in tools["edit_file"](path="work/t1.py", old="baseline = 1", new="baseline = 2")
    assert (root / "work" / "t1.py").read_text() == "baseline = 2\n"
    assert "not found" in tools["edit_file"](path="work/t1.py", old="nope", new="x")
    (root / "work" / "t2.py").write_text("a\na\n")
    assert "occurs 2 times" in tools["edit_file"](path="work/t2.py", old="a", new="b")
    assert "2 replacement" in tools["edit_file"](
        path="work/t2.py", old="a", new="b", replace_all=True
    )
    with pytest.raises(PermissionError):
        tools["copy_file"](src="work/t1.py", dst="output/k.py")
