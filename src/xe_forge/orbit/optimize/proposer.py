"""
Asking Claude what to try, and letting it try (plan §13.7, §6).

Two agent calls with different jobs, because they are different questions:

* **PLAN** — one cheap call: given the kernel and everything Orbit measured about it,
  list N *specific, bounded* transformations worth trying, ranked. Asking for a ranked
  list rather than a rewrite is deliberate. A model asked to "make this faster" produces
  one answer it is committed to; asked for candidates it produces a spread, and a spread
  is what a search wants. It also costs one call instead of N.
* **IMPLEMENT** — one Claude Code workspace per proposal, where the agent edits the
  kernel and runs the correctness harness itself until its own check passes.

The workspace matters for a reason that is easy to miss: without it, a syntax error or a
wrong tensor shape costs a full Orbit round trip to discover and report back. With it,
the agent sees its own traceback and fixes it before Orbit ever looks. That is the
difference between an agent that can implement a change and one that can only suggest it.

**And its result is never the verdict.** The workspace's own harness run is advisory —
Orbit re-runs the check in a fresh process against the real tree, and believes only
that. An agent grading its own homework is not a gate, however good the agent is.

Xe-Forge already knows how to build such a workspace (`xe_forge.claude.generator`
writes `CLAUDE.md`, a `/optimize-kernel` command and a tool-runner subagent), and it is
importable without dspy or ai_bench. Reuse it rather than growing a second one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from xe_forge.orbit.optimize.loop import Proposal

# A planning call reads one file and answers. Beyond this something is wrong with the
# prompt, not with the model's speed.
PLAN_TIMEOUT_S = 300.0

# An implementation session edits, runs a harness, and iterates. It gets longer, but not
# unbounded — an agent that has not converged in this time is not about to.
IMPLEMENT_TIMEOUT_S = 1800.0

DEFAULT_PROPOSALS = 3


# An implementation session must be allowed to edit files and run its harness, or it
# cannot implement anything. In print mode a permission prompt has nobody to answer it
# and auto-denies, which is not obvious from the outside: the first live run produced two
# perfectly good proposals, opened two workspaces, and returned "no edit produced" for
# both, because every Edit call was silently refused.
#
# The grant is bounded by construction rather than by trust. The session's cwd is a
# throwaway directory holding a *copy* of one file plus the harness, and Orbit reads back
# only `workspace/<filename>` and applies it through `InPlacePatcher`, which enforces the
# sandbox root (§13.6). An agent that wrote somewhere else would not reach the loop.
IMPLEMENT_PERMISSION_MODE = "acceptEdits"


@dataclass
class ProposerConfig:
    binary: str = "claude"
    plan_timeout_s: float = PLAN_TIMEOUT_S
    implement_timeout_s: float = IMPLEMENT_TIMEOUT_S
    extra_args: tuple[str, ...] = ()
    # Set to "" to run implementation sessions read-only, which is useful for confirming
    # that the loop refuses a proposal carrying no source.
    permission_mode: str = IMPLEMENT_PERMISSION_MODE


class ClaudeProposer:
    """Plans candidate transformations, then implements them in isolated workspaces."""

    def __init__(self, config: ProposerConfig | None = None) -> None:
        self.config = config or ProposerConfig()

    def available(self) -> bool:
        return shutil.which(self.config.binary) is not None

    # -- PLAN ---------------------------------------------------------------

    def plan(
        self,
        source: str,
        knowledge: str = "",
        count: int = DEFAULT_PROPOSALS,
        kernel_label: str = "the kernel",
    ) -> list[Proposal]:
        """Ask for a ranked list of bounded transformations.

        `knowledge` is Orbit's measured context (§9.5) — GPU share, Amdahl ceiling,
        observed shapes, the instantiation that ran. Without it the model optimizes a
        kernel in the abstract; with it, it knows a 0.2% ceiling means "propose something
        cheap" rather than "propose an algorithm".
        """
        prompt = _plan_prompt(source, knowledge, count, kernel_label)
        output = self._run([self.config.binary, "-p", prompt], timeout=self.config.plan_timeout_s)
        return _parse_proposals(output)[:count]

    # -- IMPLEMENT ----------------------------------------------------------

    def implement(
        self,
        proposal: Proposal,
        target: Path,
        workspace: Path,
        harness: Path | None = None,
        harness_command: str = "",
    ) -> bytes | None:
        """Let Claude implement one proposal in its own workspace.

        Returns the proposed file contents, or None if the session produced nothing
        usable. The workspace is a *copy*: the agent never edits the live tree, so a
        session that goes wrong cannot leave the framework broken. Orbit applies the
        result through `InPlacePatcher` afterwards, where it is journalled and revertible.
        """
        workspace.mkdir(parents=True, exist_ok=True)
        working_copy = workspace / target.name
        shutil.copy2(target, working_copy)

        if harness is not None and harness.is_file():
            shutil.copy2(harness, workspace / harness.name)

        (workspace / "TASK.md").write_text(
            _task_markdown(proposal, working_copy.name, harness, harness_command),
            encoding="utf-8",
        )

        argv = [self.config.binary, "-p", _implement_prompt(proposal, working_copy.name)]
        if self.config.permission_mode:
            argv += ["--permission-mode", self.config.permission_mode]
        argv += [*self.config.extra_args]
        self._run(argv, timeout=self.config.implement_timeout_s, cwd=workspace)

        try:
            edited = working_copy.read_bytes()
        except OSError:
            return None
        return edited if edited != target.read_bytes() else None

    # -- process ------------------------------------------------------------

    def _run(self, argv: list[str], timeout: float, cwd: Path | None = None) -> str:
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env={**os.environ, "CLAUDE_NONINTERACTIVE": "1"},
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # An unreachable or slow provider is a fact about the run, not a proposal.
            # Returning empty lets the caller report "no candidates" honestly.
            return ""
        return result.stdout or ""


# ---------------------------------------------------------------------------
# Prompts and parsing
# ---------------------------------------------------------------------------


def _plan_prompt(source: str, knowledge: str, count: int, kernel_label: str) -> str:
    return "\n".join(
        [
            f"You are proposing optimizations for {kernel_label}, a GPU kernel running on "
            "an Intel XPU through PyTorch.",
            "",
            "MEASURED CONTEXT (from profiling this exact workload):",
            knowledge or "  (none available)",
            "",
            "KERNEL SOURCE:",
            "```python",
            source[:20000],
            "```",
            "",
            f"Propose exactly {count} DISTINCT, BOUNDED changes worth trying, best first.",
            "",
            "Rules that make a proposal usable:",
            "- Each must be a concrete edit to this file, not a research direction.",
            "- Each must be independently testable and independently revertible.",
            "- Prefer cheap, deterministic levers (block size, launch config, removing a",
            "  redundant load) over rewrites. The measured ceiling above tells you how",
            "  much effort is justified; a low ceiling means propose something cheap.",
            "- Do not propose anything that changes numerical results. Correctness is",
            "  gated separately and a numerically different kernel is a different kernel.",
            "",
            "Answer as a JSON array and nothing else:",
            '[{"title": "...", "rationale": "one sentence", "parameters": {...}}]',
        ]
    )


def _implement_prompt(proposal: Proposal, filename: str) -> str:
    return "\n".join(
        [
            f"Implement exactly this optimization in ./{filename}:",
            "",
            f"  {proposal.title}",
            f"  Rationale: {proposal.rationale}",
            "",
            "See TASK.md for the verification command. Requirements:",
            "- Change only what this optimization requires. An unrelated edit will be",
            "  rejected by review even if it is an improvement.",
            "- Preserve numerical behaviour exactly.",
            "- Run the verification command and iterate until it passes.",
            "- Leave the edited file in place. Do not create copies or variants.",
            "",
            "If the optimization turns out not to be applicable, revert your changes and",
            "say so — an honest 'not applicable' is worth more than a change that merely",
            "compiles.",
        ]
    )


def _task_markdown(
    proposal: Proposal, filename: str, harness: Path | None, harness_command: str
) -> str:
    verify = harness_command or (f"python {harness.name}" if harness else "(none supplied)")
    return "\n".join(
        [
            f"# Task: {proposal.title}",
            "",
            proposal.rationale,
            "",
            f"Edit `./{filename}` in this directory. It is a **copy** — the installed",
            "framework is untouched, so you cannot break anything outside this folder.",
            "",
            "## Verify",
            "",
            "```",
            verify,
            "```",
            "",
            "Exit 0 means correct, 1 means numerically wrong, 2 means the check could not",
            "run at all. Iterate until it exits 0.",
            "",
            "## What happens next",
            "",
            "Xe-Orbit re-runs this check independently against the real installed tree and",
            "measures the kernel. Your own run is advisory; the independent one decides.",
        ]
    )


def _parse_proposals(text: str) -> list[Proposal]:
    """Pull the JSON array out of a model response, tolerating prose around it."""
    if not text.strip():
        return []

    # Ordered by how much we trust them: the whole response if it is already JSON, then
    # a fenced block, then the first balanced array anywhere in the prose.
    #
    # A regex anchored on `[{` was the first attempt and it silently dropped a valid
    # array that did not happen to open with an object. A parser that returns nothing on
    # a formatting variation turns "the agent had three ideas" into "the agent had none",
    # and the run then reports no candidates for a reason unrelated to the kernel.
    candidates: list[str] = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL))
    balanced = _first_balanced_array(text)
    if balanced:
        candidates.append(balanced)

    for blob in candidates:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        proposals: list[Proposal] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            params = item.get("parameters")
            proposals.append(
                Proposal(
                    title=title,
                    rationale=str(item.get("rationale", "")).strip(),
                    parameters=params if isinstance(params, dict) else {},
                )
            )
        if proposals:
            return proposals
    return []


def _first_balanced_array(text: str) -> str | None:
    """The first bracket-balanced JSON array in a blob of prose, if there is one.

    Bracket counting rather than a regex, because a regex cannot balance nesting: a
    proposal whose `parameters` contains a list would truncate at the inner `]`.
    Strings are tracked so a bracket inside a rationale does not confuse the count.
    """
    start = text.find("[")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("[", start + 1)
    return None
