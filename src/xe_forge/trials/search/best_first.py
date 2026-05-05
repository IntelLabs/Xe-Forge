"""Best-first search: expand the highest-speedup leaf, round-robin among
leaves of equal speedup, break ties by shallowest depth.
"""

from typing import TYPE_CHECKING

from xe_forge.trials.search.base import TrialProposal, TrialSearch

if TYPE_CHECKING:
    from xe_forge.trials.context import TrialContext


def _children_map(state: dict) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for tid, t in state["trials"].items():
        p = t.get("parent")
        if p is not None:
            children.setdefault(p, []).append(tid)
    return children


def _leaves(state: dict) -> list[str]:
    children = _children_map(state)
    return [tid for tid in state["trials"] if tid not in children]


def _depth(state: dict, trial_id: str) -> int:
    d = 0
    cur = trial_id
    while cur is not None:
        parent = state["trials"][cur].get("parent")
        if parent is None:
            return d
        d += 1
        cur = parent
    return d


class BestFirstSearch(TrialSearch):
    """Priority over leaves by (speedup desc, depth asc, id asc).

    Correct-but-unranked leaves (speedup=None) get score -inf so they never
    block improvement. Failed/saved leaves are skipped entirely so a bad
    child can't lock out its sibling.
    """

    def propose(self, state: dict, ctx: "TrialContext") -> TrialProposal:
        trials = state["trials"]
        leaves = _leaves(state) or list(trials.keys())

        def sort_key(tid: str):
            t = trials[tid]
            s = t.get("speedup")
            score = s if s is not None else float("-inf")
            return (-score, _depth(state, tid), int(tid[1:]))

        # Prefer correct leaves with a speedup; fall back to anything.
        ranked = sorted(leaves, key=sort_key)
        candidates = [
            tid for tid in ranked
            if trials[tid].get("correctness") == "pass" and trials[tid].get("speedup") is not None
        ] or ranked

        parent = candidates[0]
        parent_speedup = trials[parent].get("speedup")
        if parent_speedup is None:
            strategy = f"explore from {parent} (unranked)"
        else:
            strategy = f"expand best leaf {parent} ({parent_speedup:.2f}x)"

        return TrialProposal(
            parent_id=parent,
            strategy=strategy,
            hint_tag="extend",
        )

    def should_stop(self, state: dict, ctx: "TrialContext") -> bool:
        trial_cfg = ctx.config.trial
        if len(state["trials"]) >= trial_cfg.max_trials:
            return True
        best_id = state.get("best_trial")
        if best_id is None:
            return False
        best_speedup = state["trials"][best_id].get("speedup")
        return best_speedup is not None and best_speedup >= trial_cfg.early_stop_speedup
