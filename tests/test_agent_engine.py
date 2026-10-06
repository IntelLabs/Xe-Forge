"""`--engine dspy-agent`: the Claude workspace, a DSPy agent, and a measured result.

The agent itself is a stub here that drives the real tools in the real generated
workspace; the skills, the trial tree and finalize are not mocked. What is checked is
the engine's contract: a trial `benchmark` recorded comes back, and one whose numbers
the agent typed in does not.
"""

from __future__ import annotations

import dspy
import pytest

from xe_forge.config import Config
from xe_forge.engines import create_engine
from xe_forge.engines.agent_engine import AgentEngine


def _script(agent_steps):
    class StubAgent:
        def __init__(self, tools, instructions, max_iters, **_):
            self.tools = {t.name: t for t in tools}
            assert "xe-forge-skill" in instructions and max_iters == 7

        def __call__(self, goal, workspace_brief):
            assert "kernel_name: k" in workspace_brief and "$ARGUMENTS" not in goal
            for name, kwargs in agent_steps:
                self.tools[name](**kwargs)
            return dspy.Prediction(summary="done", termination_reason="submit")

    return StubAgent


@pytest.fixture
def engine(tmp_path, monkeypatch):
    fake = tmp_path / "fake_bench.sh"
    fake.write_text(
        "#!/bin/sh\nprintf 'CORRECTNESS: pass\\nBASELINE_US: 100\\nTRIAL_US: 50\\n"
        "SPEEDUP: 2.0\\nTIMER: fake\\nVERDICT: OK\\nDONE\\n'\n"
    )
    fake.chmod(0o755)
    # The skills run in child processes and read their configuration from the environment.
    monkeypatch.setenv("EXTERNAL_BENCHMARK", str(fake))
    monkeypatch.setattr("xe_forge.cli._setup_dspy", lambda config: None)
    config = Config()
    config.engine.engine = "dspy-agent"
    config.engine.workspace = str(tmp_path / "ws")
    config.engine.max_turns = 7
    config.external.benchmark = str(fake)
    return config


SAVE_AND_BENCH = [
    ("write_file", {"path": "work/cand.py", "content": "faster = 1\n"}),
    ("skill", {"args": "trial init k test_kernels/k.py"}),
    ("skill", {"args": "trial save k work/cand.py --strategy 'stub'"}),
]


def _optimize(config):
    engine = create_engine(config)
    assert isinstance(engine, AgentEngine)
    return engine.optimize(kernel_code="baseline = 1\n", reference_code="x = 1\n", kernel_name="k")


def test_a_benchmarked_trial_is_returned(engine, monkeypatch):
    steps = [
        *SAVE_AND_BENCH,
        (
            "skill",
            {"args": "benchmark test_kernels/k.py trials/k/t0.py --kernel-name k --trial-id t0"},
        ),
    ]
    monkeypatch.setattr("xe_forge.agents.engineer.EngineerAgent", _script(steps))
    result = _optimize(engine)
    assert result.success, result.error_message
    assert result.total_speedup == 2.0
    assert result.optimized_code == "faster = 1\n"


def test_a_typed_in_result_is_not_returned(engine, monkeypatch):
    steps = [
        *SAVE_AND_BENCH,
        (
            "skill",
            {
                "args": "trial result k t0 --correctness pass --speedup 9 --baseline-us 9 --kernel-us 1"
            },
        ),
    ]
    monkeypatch.setattr("xe_forge.agents.engineer.EngineerAgent", _script(steps))
    result = _optimize(engine)
    assert not result.success
    assert result.optimized_code is None


def test_engineer_agent_builds_a_tool_loop():
    from xe_forge.agents.engineer import EngineerAgent

    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = EngineerAgent([dspy.Tool(ping)], instructions="Policy text.", max_iters=3)
    assert "Policy text." in agent.loop.signature.instructions or hasattr(agent.loop, "tools")


def test_old_steps_are_compacted_for_the_model_but_not_in_the_record():
    from xe_forge.agents.engineer import _CompactHistory

    def event(i):
        calls = dspy.ToolCalls.from_dict_list(
            [{"name": "write_file", "args": {"content": "x" * 5000}}]
        )
        results = dspy.ToolCallResults(
            tool_call_results=[{"name": "write_file", "value": "y" * 5000}]
        )
        ev = {"tool_calls": calls.model_copy(update={"tool_call_results": results})}
        if i == 0:
            ev["goal"] = "the goal"
        return ev

    history = dspy.History(messages=[event(i) for i in range(20)])
    seen = {}

    def inner(history, **kwargs):
        seen["history"] = history
        return "pred"

    assert _CompactHistory(inner, keep=6, block=8, limit=100)(history=history) == "pred"
    shown = seen["history"].messages
    # (20 - 6) // 8 * 8 = 8 steps compacted, a block at a time.
    for i, ev in enumerate(shown):
        call = ev["tool_calls"].tool_calls[0]
        value = ev["tool_calls"].tool_call_results.tool_call_results[0].value
        assert (len(call.args["content"]) < 300) is (i < 8)
        assert (len(value) < 300) is (i < 8)
    assert shown[0]["goal"] == "the goal"
    assert len(history.messages[0]["tool_calls"].tool_calls[0].args["content"]) == 5000


def test_a_submit_without_evidence_is_resumed_until_there_is_some():
    from xe_forge.agents.engineer import EngineerAgent

    calls = []

    def loop(**kwargs):
        calls.append(kwargs)
        history = kwargs.get("history") or dspy.History(messages=[])
        history = dspy.History(messages=[*history.messages, {"step": len(calls)}])
        return dspy.Prediction(summary="s", history=history, termination_reason="submit")

    reasons = iter(["no correct trial yet.", None])
    agent = EngineerAgent([], "Policy.", max_iters=10, unfinished=lambda: next(reasons))
    agent.loop = loop
    agent(goal="g", workspace_brief="b")
    assert len(calls) == 2
    assert calls[1]["max_iters"] == 9 and "no correct trial yet." in calls[1]["goal"]
    assert len(calls[1]["history"].messages) == 1

    stubborn = EngineerAgent([], "Policy.", max_iters=10, unfinished=lambda: "never", resumes=2)
    calls.clear()
    stubborn.loop = loop
    stubborn(goal="g", workspace_brief="b")
    assert len(calls) == 3  # the first run and at most two resumes


def test_a_measured_win_is_finished_port_or_not(engine, tmp_path):
    """Porting is the prompt's to ask for, as in the Claude engine; not a resume reason."""
    import json

    engine.external.kernel_repo = str(tmp_path)
    agent = AgentEngine(engine)
    ws = tmp_path / "ws"
    trials = ws / "trials" / "k"
    trials.mkdir(parents=True)
    (trials / "t0.py").write_text("faster\n")
    state = {
        "kernel_name": "k",
        "baseline_sha256": "other",
        "trials": {
            "t0": {
                "file": "t0.py",
                "source": "measured",
                "correctness": "pass",
                "speedup": 1.6,
                "triton_us": 1.0,
                "status": "completed",
            }
        },
        "best_trial": "t0",
    }
    (trials / "state.json").write_text(json.dumps(state))
    assert agent._unfinished(ws, "k") is None
