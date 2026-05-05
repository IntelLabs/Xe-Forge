"""Triton8-style tree walk: extend on improve, branch back on regress,
escalate on plateau.
"""

from typing import TYPE_CHECKING

from xe_forge.trials.search.base import TrialProposal, TrialSearch

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext


def _trial_order(state: dict) -> list[str]:
    """Return trial IDs in insertion order (t0, t1, ...)."""
    return sorted(state["trials"].keys(), key=lambda tid: int(tid[1:]))


def _latest_trial_id(state: dict) -> str | None:
    order = _trial_order(state)
    return order[-1] if order else None


def _speedup_of(state: dict, trial_id: str) -> float | None:
    return state["trials"].get(trial_id, {}).get("speedup")


def _global_best_speedup(state: dict) -> float:
    best_id = state.get("best_trial")
    if best_id is None:
        return -1.0
    return _speedup_of(state, best_id) or -1.0


class TreeWalkSearch(TrialSearch):
    """Default search: walks the tree with backtracking + escalation.

    Decision rules:
    - If no prior trial exists: propose parent=t0, hint_tag=first_try.
    - If latest trial has no recorded speedup (saved/partial): extend from best.
    - If latest trial failed correctness: fix on the same branch (retry from
      the latest's parent with hint_tag=fix_on_same_branch).
    - If latest trial improved vs. its parent AND is the new global best:
      extend from latest.
    - If latest trial regressed vs. its parent: branch back to the global best.
    - If the branch has plateaued (plateau_window trials without a new global
      best): escalate via hint_tag=try_harder (still branching from best).
    """

    def propose(self, state: dict, ctx: "TrialContext") -> TrialProposal:
        trials = state.get("trials", {})
        best_id = state.get("best_trial")
        latest_id = _latest_trial_id(state)

        # Cold start — shouldn't happen since the runner seeds t0, but safe.
        if latest_id is None:
            return TrialProposal(
                parent_id="t0",
                strategy="initial exploration",
                hint_tag="first_try",
            )

        latest = trials[latest_id]

        # First child of t0.
        if latest_id == "t0":
            return TrialProposal(
                parent_id="t0",
                strategy="first optimization pass on t0",
                hint_tag="first_try",
            )

        # Latest trial not yet benchmarked — extend from best anyway.
        if latest["speedup"] is None or latest.get("status") == "saved":
            return TrialProposal(
                parent_id=best_id or "t0",
                strategy="continue from best trial",
                hint_tag="extend",
            )

        # Correctness failure — retry on same branch.
        if latest.get("correctness") == "fail" or latest.get("validation") == "fail":
            retry_parent = latest.get("parent") or "t0"
            return TrialProposal(
                parent_id=retry_parent,
                strategy=f"retry {latest_id}'s branch (fix correctness)",
                hint_tag="fix_on_same_branch",
            )

        # Plateau detection — no new global best for plateau_window trials.
        if self.is_plateau(state, ctx):
            return TrialProposal(
                parent_id=best_id or "t0",
                strategy="plateau — try a fundamentally different approach",
                hint_tag="try_harder",
            )

        parent_id = latest.get("parent")
        parent_speedup = _speedup_of(state, parent_id) if parent_id else None
        latest_speedup = latest["speedup"]

        # Regression vs. parent — backtrack to global best.
        if parent_speedup is not None and latest_speedup < parent_speedup:
            return TrialProposal(
                parent_id=best_id or parent_id or "t0",
                strategy=f"regressed from {parent_id}, branching back to best",
                hint_tag="branch_back",
            )

        # Improvement — extend from latest.
        return TrialProposal(
            parent_id=latest_id,
            strategy=f"extend from {latest_id} ({latest_speedup:.2f}x)",
            hint_tag="extend",
        )

    def should_stop(self, state: dict, ctx: "TrialContext") -> bool:
        trial_cfg = ctx.config.trial
        if len(state.get("trials", {})) >= trial_cfg.max_trials:
            return True
        if _global_best_speedup(state) >= trial_cfg.early_stop_speedup:
            return True
        return False

    def is_plateau(self, state: dict, ctx: "TrialContext") -> bool:
        window = ctx.config.trial.plateau_window
        order = _trial_order(state)
        if len(order) < window + 1:
            return False
        # Best across the last `window` trials vs. best across everything.
        recent_best = max(
            (_speedup_of(state, tid) or -1.0 for tid in order[-window:]),
            default=-1.0,
        )
        all_best = _global_best_speedup(state)
        # Plateau = none of the recent trials set a new global best.
        # Use a small tolerance to avoid float noise.
        return recent_best + 1e-6 < all_best
