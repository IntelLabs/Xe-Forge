"""
An agent that learns from its own trials (plan §13.7).

The first live run exposed the shape of the problem. PLAN was called once, produced two
well-argued candidates, and both measured roughly 2x slower. The loop reverted them
correctly and stopped — and the agent never found out. Asked again it would have
proposed the same two things, because nothing in the design carried a result backwards.

That is the difference between a batch of suggestions and a search. A search needs three
things the one-shot design lacks:

* **Memory of what was tried and what it measured.** Not "this exact edit was attempted"
  — the novelty ledger already refuses literal repeats — but *why* it failed and by how
  much. `BLOCK_SIZE 1024 -> 8192 measured -108.87%` is a fact about the device that
  should reshape every subsequent proposal, including ones in different directions.
* **The machine.** The two failures were sound reasoning applied to a device the agent
  had not been told about: it argued from a discrete GPU with hundreds of EUs, on a part
  with sixteen. §9.5 says measured facts belong in the context; the device is a measured
  fact.
* **A cheap way to reject the impossible.** `BLOCK_SIZE = 8192` exceeds this device's
  1024 work-group limit. That is checkable from `DeviceFacts` before anything is applied,
  and catching it costs nothing where a trial costs a patch, a correctness run and a
  measurement.

Rounds rather than a fixed stage order. Xe-Forge directs its own loop through a
curriculum — ALGORITHMIC before FUSION before AUTOTUNING — which encodes real knowledge
about what to try when. That structure suits a pipeline whose stages are known in
advance. Here the agent chooses its own next move from what the last round measured,
which is the same shape Hyperloom gives its Orchestration role: *"a single persistent
multi-turn conversation ... the agent's plan and reasoning live in the conversation, so
reasoning continuity is preserved between ticks."*

What does not change is who decides. The agent picks the direction; Orbit applies,
verifies, measures and reverts. A round's result is a measurement, never the agent's
opinion of its own patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xe_forge.orbit.device import DeviceFacts, launch_constraints
from xe_forge.orbit.optimize.loop import Proposal, TrialRecord, TrialVerdict

# Rounds of propose-trial-learn. Two is the minimum that can demonstrate a correction;
# beyond a handful the budget is better spent elsewhere (§11.6).
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
        """Directions already shown not to work, so the agent can stop re-deriving them.

        Phrased as measured outcomes rather than prohibitions. "Larger blocks measured
        2x slower" is evidence the agent can reason from — including reasoning that the
        opposite direction is worth trying. "Do not change the block size" is an
        instruction it can only obey, and it forecloses the correction.
        """
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
        """Render as a block for `ClaudeProposer.plan`'s `knowledge` parameter.

        `build_round_prompt` owns the full-prompt path; this is the seam for a caller
        that can only extend the measured context `plan()` already accepts, where it is
        interpolated as MEASURED CONTEXT — and measured context is exactly what a
        round's verdicts are. The framing matches the CLI's cross-invocation
        `_prior_trials` wording, because the two blocks are the same evidence at
        different distances: results are measurements, not opinions, and re-proposing
        one spends a trial to learn what is already known.
        """
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

    Only checks bounds that are genuinely hard. A proposal that is merely unwise is left
    to the measurement — being wrong about what is slow is what the gate is for, and a
    heuristic that pre-rejects plausible ideas would have removed the only mechanism
    that has actually produced a true answer here.
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
