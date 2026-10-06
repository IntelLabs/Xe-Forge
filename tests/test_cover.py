"""CoVeR reports failures in dspy's own LM-facing error format."""

import logging

import dspy
from dspy.utils.exceptions import format_error_for_lm

from xe_forge.agents import cover
from xe_forge.agents.cover import CoVeR


def _agent(tool):
    agent = CoVeR("kernel -> code", tools=[tool], max_iters=1)
    agent.extract = lambda **_: {"code": "fallback"}
    return agent


def test_a_tool_error_is_fed_back_with_its_traceback():
    raised = []

    def check(code: str) -> str:
        """Check the code."""
        try:
            raise RuntimeError("compile failed")
        except RuntimeError as err:
            raised.append(err)
            raise

    agent = _agent(check)
    agent.cover = lambda **_: dspy.Prediction(next_thought="try", code="x")
    observation = agent(kernel="k").trajectory["observation_0"]

    assert (
        observation
        == f"Execution error in check: {format_error_for_lm(raised[0], traceback_frames=5)}"
    )
    assert observation.startswith("Execution error in check: \nTraceback")
    assert "RuntimeError: compile failed" in observation


def test_an_untruncatable_trajectory_ends_the_loop_with_the_error_logged(caplog):
    def check(code: str) -> str:
        """Check the code."""
        return "Success!"

    def too_long(**_):
        raise cover.ContextWindowExceededError("too long", "model", "provider")

    agent = _agent(check)
    agent.cover = too_long
    with caplog.at_level(logging.WARNING, logger=cover.__name__):
        result = agent(kernel="k")

    assert result.code == "fallback"
    message = next(r.message for r in caplog.records if "failed to select" in r.message)
    assert "\nTraceback" in message and "cannot be truncated" in message
