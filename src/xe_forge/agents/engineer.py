"""EngineerAgent -- a DSPy tool-calling agent driving a generated workspace.

The workspace already carries the policy (``CLAUDE.md``) and the task
(``.claude/commands/optimize-kernel.md``) a Claude session runs from. This agent reads
the same two files as its instructions and works through
:mod:`xe_forge.agents.workspace_tools`, so the only thing that differs between the two
engines is the model choosing the next action. Nothing here decides what is correct or
fast; the skills record that, and the engine reads it back from the trial tree.
"""

from __future__ import annotations

import logging

import dspy

logger = logging.getLogger(__name__)

# How the workspace's Claude Code vocabulary maps onto this agent's tools.
TOOL_MAPPING = """\
You are not running inside Claude Code; you have exactly these tools: list_files,
read_file, search, write_file, copy_file, edit_file and skill. Where the instructions
below say:
- run `xe-forge-skill <args>` (directly or through the tool-runner agent): call
  skill(args="<args>") yourself, one at a time, and read its full output;
- Read / Glob / Grep / Write / Edit: use read_file / list_files / search / write_file /
  edit_file. Start a file from an existing one with copy_file and change it with
  edit_file; write_file is for new files;
- dispatch another agent (kernel-locator, workload-inspector, port-back): read its
  instructions in .claude/agents/<name>.md and do that work yourself with your tools.
CLAUDE.md and the task are already in these instructions; do not read them again.
Steps older than the last few are shown shortened; the files on disk are the full record.
Finish by calling submit."""


def _shorten(value, limit: int):
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return value
    return (
        f"{text[:limit]} ... [{len(text) - limit} chars elided from an earlier step; "
        "re-read the file or re-run the command if you need them]"
    )


def _compact_event(event: dict, limit: int) -> dict:
    """*event* with its tool arguments and results cut to *limit* characters each."""
    calls = event.get("tool_calls")
    if calls is None:
        return event
    shortened = [
        c.model_copy(update={"args": {k: _shorten(v, limit) for k, v in c.args.items()}})
        for c in calls.tool_calls
    ]
    results = calls.tool_call_results
    if results is not None:
        results = results.model_copy(
            update={
                "tool_call_results": [
                    r.model_copy(update={"value": _shorten(r.value, limit)})
                    for r in results.tool_call_results
                ]
            }
        )
    update = {"tool_calls": shortened, "tool_call_results": results}
    return {**event, "tool_calls": calls.model_copy(update=update)}


class _CompactHistory(dspy.Module):
    """What the model sees of the history: recent steps whole, older ones cut short.

    Every step re-sends the whole history, so a file written or read early is paid for
    on every later step. The workspace on disk is the state -- the trial tree, the
    files -- so an old step's payload can be re-read on demand instead. Older steps are
    compacted a block at a time, which keeps the prompt prefix stable between
    compactions for providers that cache it. The full history stays in the prediction.
    """

    def __init__(self, inner, keep: int = 6, block: int = 8, limit: int = 600):
        super().__init__()
        self.inner, self.keep, self.block, self.limit = inner, keep, block, limit

    def forward(self, history: dspy.History, **kwargs):
        messages = history.messages
        cut = max(0, len(messages) - self.keep) // self.block * self.block
        if cut:
            history = dspy.History(
                messages=[_compact_event(m, self.limit) for m in messages[:cut]] + messages[cut:]
            )
        return self.inner(history=history, **kwargs)


class EngineerSignature(dspy.Signature):
    """Engineer the fastest correct kernel for the goal, using only the tools."""

    goal: str = dspy.InputField(desc="What to achieve, and the task procedure to follow")
    workspace_brief: str = dspy.InputField(desc="Kernel name and the files in the workspace")
    summary: str = dspy.OutputField(
        desc="Best trial id and its measured speedup, what was tried, which stopping "
        "condition fired, and anything that could not be done"
    )


class EngineerAgent(dspy.Module):
    """``dspy.ReActV2`` over workspace tools."""

    def __init__(
        self,
        tools: list,
        instructions: str,
        max_iters: int = 80,
        unfinished=None,
        resumes: int = 2,
    ):
        """*unfinished* returns why the session is not done yet, or ``None`` if it is.

        It is asked after every submit, from the evidence on disk rather than from the
        model's own account. While it names a reason and steps remain, the session is
        resumed with that reason, at most *resumes* times.
        """
        super().__init__()
        self.max_iters, self.unfinished, self.resumes = max_iters, unfinished, resumes
        signature = EngineerSignature.with_instructions(
            f"{EngineerSignature.instructions}\n\n{TOOL_MAPPING}\n\n{instructions}"
        )
        self.loop = dspy.ReActV2(signature=signature, tools=tools, max_iters=max_iters)
        self.loop.react = _CompactHistory(self.loop.react)

    def forward(self, goal: str, workspace_brief: str) -> dspy.Prediction:
        result = self.loop(goal=goal, workspace_brief=workspace_brief)
        for _ in range(self.resumes if self.unfinished else 0):
            remaining = self.max_iters - len(result.history.messages)
            reason = self.unfinished()
            if reason is None or remaining <= 0 or result.termination_reason != "submit":
                break
            logger.info("EngineerAgent resumed: %s", reason)
            result = self.loop(
                history=result.history,
                max_iters=remaining,
                goal=(
                    f"Not done: {reason} You have {remaining} steps left. Read the last "
                    "error, fix the file that caused it and continue the workflow. Submit "
                    "again only on a halting verdict CLAUDE.md names, or when the trial "
                    "loop's stopping conditions are met."
                ),
                workspace_brief="(unchanged)",
            )
        termination = getattr(result, "termination_reason", None)
        if termination:
            logger.info("EngineerAgent termination_reason: %s", termination)
        return result
