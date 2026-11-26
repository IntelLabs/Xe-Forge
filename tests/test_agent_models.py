"""Agents named by tier: the fast tier's model reaches each agent CLI in its own form."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from x_forge import agent_cli
from x_forge.claude.render import render
from x_forge.config import Config


def _config(fast=None):
    config = Config()
    config.llm.model, config.llm.fast_model = "openai/big-model", fast
    return config


def test_the_fast_tier_is_the_sessions_model_unless_one_is_named():
    assert agent_cli.agent_models(_config()) == {"fast": "inherit"}
    assert agent_cli.agent_models(_config("openai/small-model")) == {"fast": "small-model"}


def test_a_workspaces_tool_runner_runs_on_the_fast_tier(tmp_path):
    from x_forge.claude.generator import generate_workspace

    for fast, want in ((None, "model: inherit"), ("openai/small-model", "model: small-model")):
        ws = tmp_path / str(fast).replace("/", "_")
        generate_workspace(ws, _config(fast), "scale", "", reference_code="x = 1\n")
        assert want in (ws / ".claude/agents/tool-runner.md").read_text().splitlines()


def test_codex_agents_name_a_model_only_when_the_tier_has_one(tmp_path):
    codex = agent_cli.BACKENDS["codex"]
    for fast in (None, "openai/small-model"):
        cwd = tmp_path / str(fast).replace("/", "_")
        (cwd / ".claude/agents").mkdir(parents=True)
        (cwd / ".claude/agents/code-explorer.md").write_text(
            render("code-explorer.md.j2", models=agent_cli.agent_models(_config(fast)))
        )
        assert codex.install_agents(cwd) == ["code-explorer"]
        agent = tomllib.loads((cwd / ".codex/agents/code-explorer.toml").read_text())
        assert agent.get("model") == (None if fast is None else "small-model")
        assert "never read it" in agent["developer_instructions"]
    assert agent_cli.BACKENDS["claude"].install_agents(cwd) == ["code-explorer"]


def test_a_sessions_agents_reach_claude_as_arguments_with_their_model(tmp_path):
    text = render("code-explorer.md.j2", models={"fast": "small-model"})
    argv = agent_cli.BACKENDS["claude"].session_agents(tmp_path, [text])
    agents = json.loads(Path(argv[1]).read_text())
    assert argv[0] == "--agents" and agents["code-explorer"]["model"] == "small-model"
    assert agents["code-explorer"]["prompt"].startswith("You answer one question")
    plain = render("code-explorer.md.j2", models={"fast": "inherit"})
    argv = agent_cli.BACKENDS["claude"].session_agents(tmp_path, [plain])
    assert "model" not in json.loads(Path(argv[1]).read_text())["code-explorer"]


def test_every_session_and_subagent_that_reads_code_gets_the_reading_rules():
    rules = render("read-rules.md.j2").strip()
    common = {"kernel_name": "k", "kernel_repo": "/r", "profile_path": "p.json"}
    texts = {
        "frontend-rules": render("frontend-rules.md.j2", python="py", answer="a.json", venv=None),
        "kernel-locator": render(
            "kernel-locator.md.j2", **common, baseline_oracle=None, reference_given=False
        ),
        "port-back": render(
            "port-back.md.j2",
            **common,
            integration_repos=[],
            ext=".py",
            stack_python=None,
            repo_trials=False,
        ),
    }
    for name, text in texts.items():
        assert rules in text, name
    assert "code-explorer" in texts["frontend-rules"]


def test_the_kernel_sessions_instructions_carry_the_reading_rules(tmp_path):
    from x_forge.claude.generator import generate_workspace

    generate_workspace(tmp_path / "ws", _config(), "scale", "", reference_code="x = 1\n")
    assert render("read-rules.md.j2").strip() in (tmp_path / "ws/CLAUDE.md").read_text()


def test_codex_is_told_its_agents_before_the_prompt_and_before_resume(tmp_path, monkeypatch):
    codex = agent_cli.BACKENDS["codex"]
    text = render("code-explorer.md.j2", models={"fast": "small-model"})
    extra = codex.session_agents(tmp_path, [text])
    toml = (tmp_path / ".codex/agents/code-explorer.toml").resolve()
    assert extra == ["-c", f'agents.code-explorer.config_file="{toml}"']
    monkeypatch.setattr(codex, "path", lambda: "codex")
    start = codex.start("the prompt", 5, None, extra)
    assert start[-1] == "the prompt" and start.index(extra[1]) < len(start) - 1
    resume = codex.resume("sid", "go on", 5, None, extra)
    assert resume.index(extra[1]) < resume.index("resume")
    assert codex.agent_args(tmp_path / "nowhere") == []
    assert agent_cli.BACKENDS["claude"].agent_args(tmp_path) == []
