"""Round-based optimization sessions: each round's proposals are informed by the
device's facts and by what previous rounds measured. Design rationale: docs/DESIGN.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from xe_forge.orbit.device import DeviceFacts, launch_constraints
from xe_forge.orbit.optimize.loop import Proposal, TrialRecord, TrialVerdict

# Rounds of propose-trial-learn per session.
DEFAULT_ROUNDS = 3


@dataclass
class RoundOutcome:
    """What one round of proposals achieved, in the terms the next round needs."""

    index: int
    trials: list[TrialRecord] = field(default_factory=list)

    @property
    def best(self) -> TrialRecord | None:
        measured = [t for t in self.trials if t.delta_percent is not None]
        return max(measured, key=lambda t: t.delta_percent or 0.0) if measured else None

    def summarize(self) -> str:
        """Render for the next round's prompt — results first, reasons attached."""
        if not self.trials:
            return f"Round {self.index}: nothing was trialled."
        lines = [f"Round {self.index} results:"]
        for trial in self.trials:
            delta = trial.delta_percent
            measured = f"{delta:+.2f}%" if delta is not None else "not measured"
            lines.append(f"  - {trial.proposal.title}")
            lines.append(f"      {trial.verdict.value}: {measured} — {trial.reason}")
        return "\n".join(lines)


@dataclass
class SessionHistory:
    """Everything learned so far, as evidence rather than as a list of attempts."""

    rounds: list[RoundOutcome] = field(default_factory=list)

    @property
    def anything_measured(self) -> bool:
        return any(t.delta_percent is not None for r in self.rounds for t in r.trials)

    @property
    def best_so_far(self) -> TrialRecord | None:
        candidates = [r.best for r in self.rounds if r.best is not None]
        return max(candidates, key=lambda t: t.delta_percent or 0.0) if candidates else None

    def failed_directions(self) -> list[str]:
        """Directions already shown not to work, phrased as measured outcomes."""
        out: list[str] = []
        for round_ in self.rounds:
            for trial in round_.trials:
                if trial.verdict is TrialVerdict.KEPT:
                    continue
                delta = trial.delta_percent
                if delta is not None:
                    out.append(f"{trial.proposal.title} -> {delta:+.2f}%")
        return out

    def render(self) -> str:
        if not self.rounds:
            return ""
        parts = [r.summarize() for r in self.rounds]
        best = self.best_so_far
        if best is not None and best.delta_percent is not None:
            parts.append(
                f"\nBest measured so far: {best.proposal.title} at "
                f"{best.delta_percent:+.2f}% (verdict {best.verdict.value})."
            )
        return "\n\n".join(parts)

    def render_for_knowledge(self) -> str:
        """Render as a MEASURED CONTEXT block for `ClaudeProposer.plan`'s
        `knowledge` parameter. Empty string when nothing has been trialled."""
        body = self.render()
        if not body:
            return ""
        return (
            "WHAT THIS SESSION ALREADY TRIED AND MEASURED ON THIS EXACT DEVICE"
            " (measurements, not opinions — do not re-propose these; a direction that"
            " measured worse is evidence the opposite direction may be the move):\n" + body
        )


def violates_device_limits(proposal: Proposal, facts: DeviceFacts) -> str:
    """Reject what the hardware cannot do, before a trial is spent on it.

    Only hard device bounds are checked; merely unwise proposals are left to the
    measurement.
    """
    if not facts.available:
        return ""

    block = proposal.parameters.get("BLOCK_SIZE") or proposal.parameters.get("block_size")
    if facts.max_work_group_size and isinstance(block, int):
        if block > facts.max_work_group_size * 8:
            return (
                f"BLOCK_SIZE {block} against a {facts.max_work_group_size} work-group "
                f"limit means each work-item handles {block // facts.max_work_group_size} "
                f"elements, which raises register pressure rather than parallelism"
            )

    warps = proposal.parameters.get("num_warps")
    if facts.sub_group_sizes and facts.eu_count and isinstance(warps, int):
        widest = max(facts.sub_group_sizes)
        requested = warps * widest
        if facts.max_work_group_size and requested > facts.max_work_group_size:
            return (
                f"num_warps={warps} at sub-group width {widest} requests {requested} "
                f"work-items, past this device's {facts.max_work_group_size} limit"
            )
    return ""


def build_round_prompt(
    source: str,
    workload_context: str,
    facts: DeviceFacts,
    history: SessionHistory,
    count: int,
    kernel_label: str,
    round_index: int,
) -> str:
    """The prompt for one round: the machine, the workload, and what has been learned."""
    lines = [
        f"You are optimizing {kernel_label}. This is round {round_index + 1}.",
        "",
        facts.describe(),
    ]

    constraints = launch_constraints(facts)
    if constraints:
        lines.append("")
        lines.append("WHAT THIS DEVICE CONSTRAINS:")
        lines.extend(f"  - {rule}" for rule in constraints)

    lines += ["", "MEASURED WORKLOAD CONTEXT:", workload_context or "  (none available)"]

    if history.rounds:
        lines += [
            "",
            "WHAT HAS ALREADY BEEN TRIED AND MEASURED ON THIS EXACT DEVICE:",
            history.render(),
            "",
            "Those numbers are measurements, not opinions. Use them: a direction that",
            "measured worse is evidence about this hardware, and the opposite direction",
            "may be the correct move. Do not re-propose anything above.",
        ]

    lines += [
        "",
        "KERNEL SOURCE:",
        "```python",
        source[:20000],
        "```",
        "",
        f"Propose exactly {count} DISTINCT, BOUNDED changes, best first.",
        "",
        "- Concrete edits to this file, each independently testable and revertible.",
        "- Reason from THIS device's numbers above, not from GPUs in general.",
        "- Do not change numerical results; correctness is gated separately.",
        "- If you believe no further change can beat the noise floor, say so in the",
        "  rationale rather than proposing something you expect to fail.",
        "",
        'Answer as JSON only: [{"title": "...", "rationale": "...", "parameters": {...}}]',
    ]
    return "\n".join(lines)
