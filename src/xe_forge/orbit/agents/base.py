"""
The `RepoAgent` protocol: bounded repository questions with a checkable answer.
Deterministic questions stay deterministic; only genuinely ambiguous ones go to an
agent, and every answer carries a confidence and evidence. Design rationale:
docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class RepoAgentError(RuntimeError):
    """Raised when an agent cannot be reached, or returns something unusable."""


@dataclass
class AgentTask:
    """One repository question, with the context needed to answer it."""

    question: str
    # Files the agent may read. Bounded on purpose: an agent turned loose on a whole
    # checkout is slow, expensive, and hard to reproduce.
    candidates: list[Path] = field(default_factory=list)
    search_root: Path | None = None
    max_files: int = 12
    context: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentAnswer:
    """What an agent concluded, with its evidence.

    `confidence` is the agent's own; callers must treat it as a claim rather than a
    measurement.
    """

    value: str | None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""
    provider: str = ""
    raw: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.value) and self.confidence > 0.0


@runtime_checkable
class RepoAgent(Protocol):
    """A provider that can answer a bounded question about this repository."""

    name: str

    def available(self) -> bool: ...

    def ask(self, task: AgentTask) -> AgentAnswer: ...


class BaseRepoAgent:
    """Shared behaviour. Subclasses supply only the provider call."""

    name = "base"

    def available(self) -> bool:
        return False

    def ask(self, task: AgentTask) -> AgentAnswer:
        raise NotImplementedError

    # -- prompt construction ----------------------------------------------

    def build_prompt(self, task: AgentTask) -> str:
        """Render a task into a prompt that asks for a checkable answer.

        The agent is asked for a file path and an explicit confidence, because a
        prose answer cannot be verified against the filesystem.
        """
        lines = [
            "You are resolving a compiled GPU kernel symbol back to the source that",
            "defines it, inside an Intel SYCL kernel tree.",
            "",
            f"QUESTION: {task.question}",
        ]

        if task.context:
            lines.append("")
            lines.append("CONTEXT:")
            for key, value in task.context.items():
                lines.append(f"  {key}: {value}")

        if task.candidates:
            lines.append("")
            lines.append("CANDIDATE FILES:")
            for path in task.candidates[: task.max_files]:
                lines.append(f"  {path}")

        if task.search_root:
            lines.append("")
            lines.append(f"SEARCH ROOT: {task.search_root}")
            lines.append("You may grep this tree. Prefer the definition, not a call site.")

        lines += [
            "",
            "Answer with exactly these lines and nothing else:",
            "  FILE: <absolute path, or NONE>",
            "  SYMBOL: <the C++ identifier that names the kernel>",
            "  CONFIDENCE: <0.0-1.0>",
            "  EVIDENCE: <file:line where it is defined>",
            "  REASONING: <one sentence>",
            "",
            "If several instantiations of one template match, say so and lower the",
            "confidence rather than choosing one: which instantiation actually ran is a",
            "separate question that profiling answers, not source reading.",
            "If you cannot find it, answer FILE: NONE. A wrong file is worse than none.",
        ]
        return "\n".join(lines)

    # -- response parsing --------------------------------------------------

    def parse_answer(self, text: str) -> AgentAnswer:
        """Parse the structured reply, refusing anything that does not fit the contract."""
        fields: dict[str, str] = {}
        for line in text.splitlines():
            for key in ("FILE", "SYMBOL", "CONFIDENCE", "EVIDENCE", "REASONING"):
                prefix = f"{key}:"
                if line.strip().upper().startswith(prefix):
                    fields[key] = line.split(":", 1)[1].strip()

        raw_file = fields.get("FILE", "").strip()
        if not raw_file or raw_file.upper() == "NONE":
            return AgentAnswer(
                value=None,
                confidence=0.0,
                reasoning=fields.get("REASONING", "agent reported no match"),
                provider=self.name,
                raw=text[-2000:],
            )

        try:
            confidence = float(fields.get("CONFIDENCE", "0"))
        except ValueError:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # A path the agent invented is worse than no answer, so the claim is checked
        # against the filesystem before it is believed.
        if not Path(raw_file).is_file():
            return AgentAnswer(
                value=None,
                confidence=0.0,
                reasoning=f"agent named a path that does not exist: {raw_file}",
                provider=self.name,
                raw=text[-2000:],
            )

        evidence = [fields[k] for k in ("EVIDENCE", "SYMBOL") if fields.get(k)]
        return AgentAnswer(
            value=raw_file,
            confidence=confidence,
            evidence=evidence,
            reasoning=fields.get("REASONING", ""),
            provider=self.name,
            raw=text[-2000:],
        )
