"""
The `RepoAgent` protocol (plan §3).

§3 states the boundary precisely: *"Orbit never imports DSPy or Claude tooling directly.
Kernel optimization goes through Xe-Forge's existing engine seam … and repository-level
agent tasks go through a new `RepoAgent` protocol added in Orbit, provider selected by
config."* This module is that protocol.

It exists because some questions about a repository are genuinely not deterministic.
Resolving a mangled SYCL kernel name to the source that defines it looked deterministic
and was implemented with regular expressions; on real Intel kernel trees that approach
recovered `GeluErfFunctor` and silently destroyed `IgammaFunctor` and
`ComputeInverseLTFunctor`, because a greedy pattern for Itanium template mangling ate
identifiers that happened to contain an `I`. Regular expressions were not parsing C++,
they were guessing at it, and the guess failed quietly — which is the failure mode this
whole project is built to avoid.

The division of labour §3 asks for still holds, and it is not a formality:

* Anything with an exact answer stays deterministic — a C++ parser for symbol
  definitions, an AST walk for Triton closure, arithmetic for ranking and acceptance.
  Sending those to a model would make a reproducible answer non-reproducible and cost
  tokens for the privilege.
* Anything genuinely ambiguous goes to an agent — which of several template
  instantiations actually ran, what a macro-generated symbol expands to, whether two
  candidate files describe the same kernel. These have no closed form, and the honest
  alternative to an agent is a low-confidence guess.

Every answer carries a confidence and the evidence behind it, so a caller can apply the
same rule the rest of Orbit applies: an ambiguous result reduces confidence rather than
being promoted to a fact (§11.4).
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

    `confidence` is the agent's own, and callers must treat it as a claim rather than a
    measurement — §11.4 grades confidence rather than trusting it, and an agent answer
    is exactly the kind of input that should lower a score rather than settle it.
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

        The output contract matters more than the wording: the agent is asked for a
        file path and an explicit confidence, because a prose answer cannot be verified
        against the filesystem and a confident-sounding paragraph is exactly what this
        project refuses to accept as evidence (§19).
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
