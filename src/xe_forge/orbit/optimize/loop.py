"""The agentic optimization loop: an agent proposes; Orbit applies, verifies,
measures and keeps or reverts, with gates run cheapest-first (correctness before
measurement). Design rationale: docs/DESIGN.md."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from xe_forge.orbit.novelty import Attempt, NoveltyLedger, Verdict
from xe_forge.orbit.optimize.harness import CheckOutcome, CheckResult
from xe_forge.orbit.patch.inplace import InPlacePatcher, PatchSafetyError
from xe_forge.orbit.policy import PolicyGate, PolicyViolation

# Small on purpose: each trial costs an agent session plus a measurement.
DEFAULT_TRIALS = 3


class TrialVerdict(StrEnum):
    KEPT = "KEPT"
    # Applied, checked, and slower or no better.
    REVERTED_SLOWER = "REVERTED_SLOWER"
    # Applied and numerically wrong. The positive-control case.
    REVERTED_WRONG = "REVERTED_WRONG"
    # Never applied: a gate refused it — the novelty ledger, the sandbox check,
    # the critic, or the policy gate.
    REFUSED = "REFUSED"
    # Applied but nothing could be established. Reverted, and recorded as a gap.
    UNPROVEN = "UNPROVEN"


@dataclass
class Proposal:
    """One bounded change an agent suggested, before anything has been done with it."""

    title: str
    rationale: str
    new_source: bytes | None = None
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass
class TrialRecord:
    """What happened to one proposal, and why."""

    index: int
    proposal: Proposal
    verdict: TrialVerdict
    reason: str
    accuracy: float | None = None
    baseline_us: float | None = None
    candidate_us: float | None = None

    @property
    def delta_percent(self) -> float | None:
        if not self.baseline_us or not self.candidate_us:
            return None
        return (self.baseline_us - self.candidate_us) / self.baseline_us * 100.0

    def format(self) -> str:
        line = f"  [{self.verdict.value:<16}] {self.proposal.title[:52]}"
        delta = self.delta_percent
        if delta is not None:
            line += f"  {delta:+.2f}%"
        return f"{line}\n       {self.reason}"


@dataclass
class LoopResult:
    kernel_id: str
    trials: list[TrialRecord] = field(default_factory=list)
    accepted: TrialRecord | None = None
    baseline_us: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def kept(self) -> bool:
        return self.accepted is not None

    def format(self) -> str:
        lines = [f"optimization loop for {self.kernel_id}", "=" * 72]
        if self.baseline_us:
            lines.append(f"baseline: {self.baseline_us:.2f} us (kernel-level)")
        for trial in self.trials:
            lines.append(trial.format())
        lines.append("=" * 72)
        if self.accepted is not None:
            delta = self.accepted.delta_percent
            lines.append(
                f"ACCEPTED: {self.accepted.proposal.title}"
                + (f" ({delta:+.2f}% at kernel level)" if delta is not None else "")
            )
        else:
            lines.append("NO CANDIDATE ACCEPTED — the tree is unchanged.")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


class OptimizationLoop:
    """Runs proposals through the gates, in cheapest-first order."""

    def __init__(
        self,
        target: Path,
        patcher: InPlacePatcher,
        check: Callable[[], CheckResult],
        measure: Callable[[], float | None],
        critic: Callable[[Proposal, str], tuple[bool, str]] | None = None,
        ledger: NoveltyLedger | None = None,
        min_improvement_percent: float = 1.0,
        measure_samples: Callable[[], list[float]] | None = None,
        policy: PolicyGate | None = None,
    ) -> None:
        self.target = Path(target)
        self.patcher = patcher
        self.check = check
        self.measure = measure
        self.critic = critic
        self.ledger = ledger or NoveltyLedger()
        # Below this, a "win" is indistinguishable from measurement noise.
        self.min_improvement_percent = min_improvement_percent
        # When samples are available, the decision is `stats.compare` (which can say
        # INCONCLUSIVE); the fixed floor is the fallback for single-number callers.
        self.measure_samples = measure_samples
        # Optional; without one, the patcher's sandbox check alone gates writes and
        # no single-writer lock is taken.
        self.policy = policy

    def run(self, proposals: list[Proposal], baseline_us: float | None = None) -> LoopResult:
        result = LoopResult(kernel_id=self.target.name)

        self._baseline_samples: list[float] = []
        if self.measure_samples is not None:
            self._baseline_samples = self.measure_samples() or []
            if self._baseline_samples and baseline_us is None:
                ordered = sorted(self._baseline_samples)
                baseline_us = ordered[len(ordered) // 2]
        if baseline_us is None:
            baseline_us = self.measure()
        result.baseline_us = baseline_us
        if baseline_us is None:
            result.notes.append(
                "no baseline measurement; candidates can be checked for correctness but "
                "not compared, so none can be accepted"
            )

        original = self.target.read_bytes()

        for index, proposal in enumerate(proposals):
            record = self._trial(index, proposal, original, baseline_us)
            result.trials.append(record)
            if record.verdict is TrialVerdict.KEPT:
                result.accepted = record

        # Only the winner stays; anything else is reverted before the loop returns.
        if result.accepted is None:
            self._revert_all()
        return result

    def _trial(
        self,
        index: int,
        proposal: Proposal,
        original: bytes,
        baseline_us: float | None,
    ) -> TrialRecord:
        attempt = Attempt(
            "optimize", str(self.target), dict(proposal.parameters) or {"title": proposal.title}
        )

        # -- gate 1: novelty (free) -----------------------------------------
        verdict, why = self.ledger.classify(attempt)
        if verdict is Verdict.STALL:
            return TrialRecord(index, proposal, TrialVerdict.REFUSED, why)

        if proposal.new_source is None:
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.REFUSED,
                "the proposal carries no source; nothing to apply",
            )

        # -- gate 2: policy / sandbox (free) --------------------------------
        # With a policy gate, the allowlist and write invariants are checked through
        # it; without one, the patcher's own check is the whole gate.
        try:
            if self.policy is not None:
                self.policy.check_action("apply_patch")
                self.policy.check_write(self.target)
            else:
                self.patcher.check(self.target)
        except (PatchSafetyError, PolicyViolation) as exc:
            return TrialRecord(index, proposal, TrialVerdict.REFUSED, str(exc))

        # -- gate 3: critic (one agent call, before anything is written) -----
        if self.critic is not None:
            approved, critique = self.critic(proposal, _diff_summary(original, proposal.new_source))
            if not approved:
                return TrialRecord(index, proposal, TrialVerdict.REFUSED, f"critic: {critique}")

        if self.policy is None:
            return self._apply_and_judge(index, proposal, attempt, baseline_us)
        # The single-writer lock covers exactly the mutating span, so two concurrent
        # loops cannot patch the same target.
        try:
            with self.policy.single_writer(self.target):
                return self._apply_and_judge(index, proposal, attempt, baseline_us)
        except PolicyViolation as exc:
            return TrialRecord(index, proposal, TrialVerdict.REFUSED, str(exc))

    def _apply_and_judge(
        self,
        index: int,
        proposal: Proposal,
        attempt: Attempt,
        baseline_us: float | None,
    ) -> TrialRecord:
        """Apply, verify, measure and decide — the mutating span of one trial."""
        # -- apply -----------------------------------------------------------
        try:
            self.patcher.apply(
                self.target,
                proposal.new_source,
                kernel_id=self.target.stem,
                reason=proposal.title,
            )
        except PatchSafetyError as exc:
            return TrialRecord(index, proposal, TrialVerdict.REFUSED, str(exc))
        self.ledger.record(attempt)

        # -- gate 4: correctness, before paying for a measurement -------------
        check = self.check()
        if check.outcome is CheckOutcome.WRONG:
            self._revert_all()
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.REVERTED_WRONG,
                f"correctness: {check.detail}",
                accuracy=check.accuracy,
            )
        if check.outcome is CheckOutcome.UNCHECKED:
            self._revert_all()
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.UNPROVEN,
                f"could not be checked, so it cannot be kept: {check.detail}",
                accuracy=check.accuracy,
            )

        # -- measure ----------------------------------------------------------
        # The statistical path collects its own samples, so it is checked before the
        # single-value `measure` is called.
        if self.measure_samples is not None and getattr(self, "_baseline_samples", None):
            return self._decide_statistically(index, proposal, check, baseline_us)

        candidate_us = self.measure()
        if candidate_us is None or baseline_us is None:
            self._revert_all()
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.UNPROVEN,
                "correct, but no usable measurement to compare against",
                accuracy=check.accuracy,
            )

        delta = (baseline_us - candidate_us) / baseline_us * 100.0
        if delta < self.min_improvement_percent:
            self._revert_all()
            # "Below the floor" and "a clear regression" are both rejections but not
            # the same finding; the reason must say which.
            if delta <= -self.min_improvement_percent:
                reason = (
                    f"{delta:+.2f}% — a clear regression, well outside the "
                    f"{self.min_improvement_percent:g}% floor"
                )
            else:
                reason = (
                    f"{delta:+.2f}% is inside the {self.min_improvement_percent:g}% "
                    f"floor, so it cannot be distinguished from noise"
                )
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.REVERTED_SLOWER,
                reason,
                accuracy=check.accuracy,
                baseline_us=baseline_us,
                candidate_us=candidate_us,
            )

        # Remember the winner so a later trial's revert can put it back.
        self._accepted_source = proposal.new_source
        return TrialRecord(
            index,
            proposal,
            TrialVerdict.KEPT,
            f"correct and {delta:+.2f}% faster at kernel level",
            accuracy=check.accuracy,
            baseline_us=baseline_us,
            candidate_us=candidate_us,
        )

    def _revert_all(self) -> None:
        """Undo the candidate under trial, and only that one.

        Never raises: a failed revert must not mask the verdict that prompted it.
        """
        try:
            self.patcher.revert_all()
        except Exception:
            pass
        # revert_all restores the pristine file (the patcher keeps one record per
        # target), which undoes the accepted winner too — so re-apply it.
        accepted = getattr(self, "_accepted_source", None)
        if accepted is not None:
            try:
                self.patcher.apply(
                    self.target,
                    accepted,
                    kernel_id=self.target.stem,
                    reason="restore accepted candidate",
                )
            except Exception:
                pass

    def _decide_statistically(
        self,
        index: int,
        proposal: Proposal,
        check: CheckResult,
        baseline_us: float | None,
    ) -> TrialRecord:
        """Decide from an interval, not a threshold: `stats.compare` distinguishes
        faster, slower, and cannot-resolve — only the first is a reason to keep,
        only the second is evidence against the change."""
        from xe_forge.orbit.models import Decision
        from xe_forge.orbit.stats import compare, minimum_detectable_effect

        candidate_samples = self.measure_samples() if self.measure_samples else []
        if not candidate_samples:
            self._revert_all()
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.UNPROVEN,
                "correct, but the candidate produced no usable measurement",
                accuracy=check.accuracy,
            )

        baseline_samples = self._baseline_samples
        mde = minimum_detectable_effect(baseline_samples)
        # Lower is better for a duration, so the sense is inverted relative to throughput.
        decision, detail = compare(
            baseline_samples, candidate_samples, lower_is_better=True, mde_percent=mde
        )
        improvement = float(detail.get("improvement_percent", 0.0) or 0.0)
        ordered = sorted(candidate_samples)
        candidate_us = ordered[len(ordered) // 2]
        reason = str(detail.get("reason", "")) or decision.value

        if decision is Decision.ACCEPT:
            return TrialRecord(
                index,
                proposal,
                TrialVerdict.KEPT,
                f"correct and faster: {improvement:+.2f}%, {reason}",
                accuracy=check.accuracy,
                baseline_us=baseline_us,
                candidate_us=candidate_us,
            )

        self._revert_all()
        if decision is Decision.REJECT:
            verdict, text = TrialVerdict.REVERTED_SLOWER, f"slower: {improvement:+.2f}%, {reason}"
        else:
            # INCONCLUSIVE and INVALID are "not established": revert, but calling it
            # slower would be a claim the data does not support.
            verdict = TrialVerdict.UNPROVEN
            text = f"{decision.value.lower()}: {improvement:+.2f}%, {reason} (MDE {mde:.2f}%)"
        return TrialRecord(
            index,
            proposal,
            verdict,
            text,
            accuracy=check.accuracy,
            baseline_us=baseline_us,
            candidate_us=candidate_us,
        )


def _diff_summary(original: bytes, candidate: bytes) -> str:
    """A unified diff for the critic to review, bounded so a prompt stays affordable."""
    import difflib

    diff = difflib.unified_diff(
        original.decode("utf-8", "replace").splitlines(),
        candidate.decode("utf-8", "replace").splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
        n=3,
    )
    lines = list(diff)[:200]
    return "\n".join(lines)


def interleaved_samples(
    patcher: InPlacePatcher,
    target: Path,
    candidate_source: bytes,
    measure: Callable[[], float | None],
    pairs: int = 3,
    kernel_id: str = "",
) -> tuple[list[float], list[float]]:
    """Collect baseline and candidate samples ABBA-interleaved to cancel drift.

    Only use where switching arms is cheap: a fresh process per switch costs more
    variance than the drift it cancels, so prefer many in-process replicates there.
    A `None` measurement is dropped — a failed run is missing data, not a fast one.
    """
    baseline: list[float] = []
    candidate: list[float] = []

    def sample_into(bucket: list[float]) -> None:
        value = measure()
        if value is not None:
            bucket.append(value)

    for index in range(pairs):
        order = ("B", "C", "C", "B") if index % 2 == 0 else ("C", "B", "B", "C")
        for arm in order:
            if arm == "C":
                patcher.apply(
                    target, candidate_source, kernel_id=kernel_id, reason="interleaved trial"
                )
                sample_into(candidate)
                patcher.revert_all()
            else:
                sample_into(baseline)
    return baseline, candidate
