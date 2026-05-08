"""Trial Tree State Manager for iterative kernel optimization.

Manages a tree of optimization trials for each kernel, tracking parent-child
relationships, strategies, correctness, and speedup results. Supports
branching back to the best ancestor when a trial regresses.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class TrialManager:
    """Persistent tree-structured trial state manager.

    State is stored as JSON at ``{trials_dir}/{kernel_name}/state.json``
    with trial kernel files at ``{trials_dir}/{kernel_name}/t{N}.py``.
    """

    def __init__(self, trials_dir: str | Path = "./trials"):
        self.trials_dir = Path(trials_dir)

    def _state_path(self, kernel_name: str) -> Path:
        return self.trials_dir / kernel_name / "state.json"

    def _trial_dir(self, kernel_name: str) -> Path:
        return self.trials_dir / kernel_name

    def _load_state(self, kernel_name: str) -> dict:
        path = self._state_path(kernel_name)
        if not path.exists():
            raise FileNotFoundError(f"No trial tree found for '{kernel_name}'. Call init() first.")
        state = json.loads(path.read_text())
        state.setdefault("baseline_type", "pytorch")
        for trial in state.get("trials", {}).values():
            if "pytorch_us" in trial and "baseline_us" not in trial:
                trial["baseline_us"] = trial.pop("pytorch_us")
        return state

    def _save_state(self, kernel_name: str, state: dict) -> None:
        path = self._state_path(kernel_name)
        path.write_text(json.dumps(state, indent=2))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def init(
        self,
        kernel_name: str,
        baseline_file: str | Path,
        *,
        triton_baseline: bool = False,
    ) -> None:
        """Initialize a new trial tree for *kernel_name*."""
        trial_dir = self._trial_dir(kernel_name)
        if self._state_path(kernel_name).exists():
            logger.warning("Trial tree for '%s' already exists. Reusing.", kernel_name)
            return
        trial_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "kernel_name": kernel_name,
            "pytorch_file": str(baseline_file),
            "baseline_type": "triton" if triton_baseline else "pytorch",
            "trials": {},
            "best_trial": None,
            "next_id": 0,
            "baseline_us": None,
        }
        self._save_state(kernel_name, state)
        logger.info("Initialized trial tree for '%s'", kernel_name)

    def save_trial(
        self,
        kernel_name: str,
        trial_file: str | Path,
        *,
        parent: str | None = None,
        strategy: str = "",
    ) -> str:
        """Save a trial by copying the kernel file into the trial directory.

        Returns the assigned trial id (e.g. ``"t0"``).
        """
        trial_file = Path(trial_file)
        if not trial_file.exists():
            raise FileNotFoundError(f"Trial file not found: {trial_file}")

        state = self._load_state(kernel_name)

        if parent is not None and parent not in state["trials"]:
            if state["next_id"] == 0:
                parent = None
            else:
                raise ValueError(
                    f"Parent trial '{parent}' not found. Available: {list(state['trials'].keys())}"
                )

        trial_id = f"t{state['next_id']}"
        state["next_id"] += 1

        dest = self._trial_dir(kernel_name) / f"{trial_id}.py"
        try:
            shutil.copy2(trial_file, dest)
        except shutil.SameFileError:
            pass

        state["trials"][trial_id] = {
            "parent": parent,
            "file": f"{trial_id}.py",
            "strategy": strategy,
            "validation": None,
            "correctness": None,
            "speedup": None,
            "baseline_us": None,
            "triton_us": None,
            "status": "saved",
        }
        self._save_state(kernel_name, state)
        logger.info("Saved trial %s: %s", trial_id, strategy)
        return trial_id

    def record_result(
        self,
        kernel_name: str,
        trial_id: str,
        *,
        validation: str | None = None,
        correctness: str | None = None,
        speedup: float | None = None,
        baseline_us: float | None = None,
        triton_us: float | None = None,
    ) -> dict:
        """Record benchmark results for a trial. Returns the trial dict."""
        state = self._load_state(kernel_name)

        if trial_id not in state["trials"]:
            raise KeyError(
                f"Trial '{trial_id}' not found. Available: {list(state['trials'].keys())}"
            )

        trial = state["trials"][trial_id]
        if validation is not None:
            trial["validation"] = validation
        if correctness is not None:
            trial["correctness"] = correctness
        if speedup is not None:
            trial["speedup"] = speedup
        if baseline_us is not None:
            trial["baseline_us"] = baseline_us
        if triton_us is not None:
            trial["triton_us"] = triton_us

        if baseline_us is not None and state.get("baseline_us") is None:
            state["baseline_us"] = [baseline_us]

        if trial["validation"] == "fail" or trial["correctness"] == "fail":
            trial["status"] = "failed"
        elif trial["correctness"] == "pass" and trial["speedup"] is not None:
            trial["status"] = "completed"
        else:
            trial["status"] = "partial"

        best_speedup = -1.0
        best_id = None
        for tid, t in state["trials"].items():
            if t.get("correctness") == "pass" and t.get("speedup") is not None:
                if t["speedup"] > best_speedup:
                    best_speedup = t["speedup"]
                    best_id = tid
        state["best_trial"] = best_id

        self._save_state(kernel_name, state)
        return trial

    def get_status(self, kernel_name: str) -> str:
        """Return an ASCII tree visualization of the trial state."""
        state = self._load_state(kernel_name)
        baseline_label = "Triton" if state.get("baseline_type") == "triton" else "PyTorch"
        lines: list[str] = []
        lines.append(f"Trial tree: {state['kernel_name']}")
        lines.append(f"  Baseline ({baseline_label}): {state['pytorch_file']}")
        lines.append(f"  Best: {state['best_trial'] or 'none'}")
        lines.append(f"  Trials: {len(state['trials'])}")
        lines.append("")

        if not state["trials"]:
            lines.append("  (no trials yet)")
            return "\n".join(lines)

        children: dict[str | None, list[str]] = {}
        roots: list[str] = []
        for tid, t in state["trials"].items():
            p = t["parent"]
            if p is None:
                roots.append(tid)
            else:
                children.setdefault(p, []).append(tid)

        def sort_key(tid: str) -> int:
            return int(tid[1:])

        roots.sort(key=sort_key)
        for k in children:
            children[k].sort(key=sort_key)

        status_icon = {
            "completed": "+",
            "failed": "X",
            "partial": "~",
            "saved": "?",
        }

        def _render(tid: str, prefix: str = "", is_last: bool = True) -> None:
            trial = state["trials"][tid]
            connector = "└── " if is_last else "├── "
            icon = status_icon.get(trial["status"], "?")
            speedup_str = f"{trial['speedup']:.2f}x" if trial["speedup"] is not None else "---"
            runtime = ""
            if trial.get("baseline_us") is not None and trial.get("triton_us") is not None:
                runtime = f" (bl={trial['baseline_us']:.0f}us, tr={trial['triton_us']:.0f}us)"
            best_marker = " <<<< BEST" if tid == state["best_trial"] else ""
            strategy_short = (trial["strategy"] or "")[:60]
            lines.append(
                f"{prefix}{connector}[{icon}] {tid}: {speedup_str}{runtime}"
                f" | {strategy_short}{best_marker}"
            )
            child_prefix = prefix + ("    " if is_last else "│   ")
            kids = children.get(tid, [])
            for i, child in enumerate(kids):
                _render(child, child_prefix, i == len(kids) - 1)

        for i, root in enumerate(roots):
            _render(root, "  ", i == len(roots) - 1)

        return "\n".join(lines)

    def get_best(self, kernel_name: str) -> dict | None:
        """Return the best correct trial record, or None."""
        state = self._load_state(kernel_name)
        best_id = state.get("best_trial")
        if best_id is None:
            return None
        trial = dict(state["trials"][best_id])
        trial["id"] = best_id
        trial["file_path"] = str(self._trial_dir(kernel_name) / trial["file"])
        return trial

    def get_baseline_us(self, kernel_name: str) -> list[float] | None:
        """Return cached baseline time(s) or None."""
        state = self._load_state(kernel_name)
        return state.get("baseline_us")

    def finalize(
        self,
        kernel_name: str,
        output_path: str | Path,
    ) -> str | None:
        """Copy the best correct trial to *output_path*.

        Returns the best trial id, or None if no correct trials exist.
        """
        state = self._load_state(kernel_name)
        best_id = state.get("best_trial")
        if best_id is None:
            logger.warning("No correct trials to finalize for '%s'", kernel_name)
            return None

        best = state["trials"][best_id]
        src = self._trial_dir(kernel_name) / best["file"]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output_path)
        logger.info(
            "Finalized %s (%.2fx) -> %s",
            best_id,
            best.get("speedup", 0),
            output_path,
        )
        return best_id

    def exists(self, kernel_name: str) -> bool:
        """Return True if a trial tree exists for *kernel_name*."""
        return self._state_path(kernel_name).exists()
