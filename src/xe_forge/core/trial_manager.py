"""Trial Tree State Manager for iterative kernel optimization.

Manages a tree of optimization trials for each kernel, tracking parent-child
relationships, strategies, correctness, and speedup results. Supports
branching back to the best ancestor when a trial regresses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _digest(path: Path) -> str | None:
    """sha256 of a trial's sources: one file, or every file under a trial directory."""
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0" + item.read_bytes())
    return digest.hexdigest()


def _rank(trial: dict) -> tuple[float, int, float] | None:
    """How this trial orders against the others, or ``None`` if it does not order.

    A host that gates its comparison reports both arms' times and withholds the ratio
    (``xe_forge.external``): the two arms ran, and the difference between them was
    smaller than what the part can resolve. Ranking on ``speedup`` alone makes such a
    trial invisible, so a *measured regression* outranks it and ``finalize`` ships the
    slower kernel -- which is the opposite of what the gate exists to prevent.

    So the order is taken from the shape of the contract rather than from a verdict
    vocabulary this module would have to keep in step with its hosts: a gated trial
    ranks at parity. The number here is an ordering key and never becomes a reported
    measurement -- ``speedup`` stays ``None`` in the record, because a loop handed a
    ratio will branch on it.

    Returns ``(rank, measured, -time)``; on equal rank a measured result wins, so a kernel
    timed at parity is preferred to one that could not be resolved, and a recorded speedup
    is rounded, so the faster measured time breaks the remaining tie.
    """
    if trial.get("correctness") != "pass" or trial.get("validation") == "fail":
        return None
    us = trial.get("triton_us")
    neg_us = -float(us) if us is not None else float("-inf")
    if trial.get("speedup") is not None:
        return (float(trial["speedup"]), 1, neg_us)
    if trial.get("baseline_us") is not None and us is not None:
        return (1.0, 0, neg_us)
    return None


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
        required_profile_groups: tuple[str, ...] = (),
    ) -> None:
        """Initialize a new trial tree for *kernel_name*."""
        trial_dir = self._trial_dir(kernel_name)
        if set(required_profile_groups) - {"ComputeBasic", "EuStallSampling", "VTune"}:
            raise ValueError("Unsupported required profile group")
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
            "required_profile_groups": list(required_profile_groups),
            "baseline_sha256": _digest(Path(baseline_file)),
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
        """Save a trial by copying the kernel into the trial directory.

        *trial_file* is a single source file, or a directory of them. A kernel split
        across files -- a header, a few translation units, a wrapper -- is one trial and
        has to be stored as one, or a later ``finalize`` hands back an entry point whose
        companions are missing and the tree records a speedup nothing can rebuild. A
        directory is stored whole, under ``t{N}/``, and every reader reaches it through
        the same ``file`` field.

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

        self._require_profiles(kernel_name, state)
        if (
            state.get("required_profile_groups")
            and not strategy.strip()
            and any(t.get("profiles") for t in state["trials"].values())
        ):
            raise ValueError(
                "--strategy must name the profiled measurement this trial targets"
                " (kernel, metric, value, source trial)"
            )
        trial_id = f"t{state['next_id']}"
        state["next_id"] += 1

        if trial_file.is_dir():
            # A directory keeps its own layout: relative paths inside it are how the
            # sources include each other, so flattening would break the build.
            dest = self._trial_dir(kernel_name) / trial_id
            if dest.resolve() != trial_file.resolve():
                shutil.copytree(trial_file, dest, dirs_exist_ok=True)
        else:
            # Keep the trial's own extension: a SYCL trial stored as .py is not a
            # cosmetic problem, since the extension is what a builder and an editor
            # dispatch on.
            suffix = trial_file.suffix or ".py"
            dest = self._trial_dir(kernel_name) / f"{trial_id}{suffix}"
            try:
                shutil.copy2(trial_file, dest)
            except shutil.SameFileError:
                pass

        state["trials"][trial_id] = {
            "parent": parent,
            "file": dest.name,
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

    def profile_source_hash(self, kernel_name: str, trial_id: str, source: str | Path) -> str:
        state = self._load_state(kernel_name)
        trial = state["trials"][trial_id]
        digest = _digest(Path(source))
        if digest is None or _digest(self._trial_dir(kernel_name) / trial["file"]) != digest:
            raise ValueError("Profile source does not match the saved trial")
        return digest

    def recorded_profile(
        self, kernel_name: str, trial_id: str, group: str, source_hash: str
    ) -> dict | None:
        """The attempt already recorded for *group* on this exact source, if any."""
        profile = self._load_state(kernel_name)["trials"][trial_id].get("profiles", {}).get(group)
        return profile if profile and profile.get("source_sha256") == source_hash else None

    def record_profile(
        self,
        kernel_name: str,
        trial_id: str,
        group: str,
        source_hash: str,
        *,
        artifacts_dir: str | None,
        error: str | None,
        warnings: list[str],
    ) -> None:
        state = self._load_state(kernel_name)
        trial = state["trials"][trial_id]
        if _digest(self._trial_dir(kernel_name) / trial["file"]) != source_hash:
            raise ValueError("Saved trial changed during profiling")
        trial.setdefault("profiles", {})[group] = {
            "source_sha256": source_hash,
            "status": "failed" if error else "collected",
            "artifacts_dir": artifacts_dir,
            "error": error,
            "warnings": warnings,
        }
        self._save_state(kernel_name, state)

    def _require_profiles(self, kernel_name: str, state: dict) -> None:
        required = state.get("required_profile_groups", [])
        if not required:
            return
        missing = []
        baseline = state.get("baseline_sha256")
        for trial_id, trial in state["trials"].items():
            if trial.get("correctness") != "pass" or trial.get("validation") == "fail":
                continue
            digest = _digest(self._trial_dir(kernel_name) / trial["file"])
            if digest is not None and digest == baseline:
                continue
            for group in required:
                profile = trial.get("profiles", {}).get(group, {})
                if (
                    digest is None
                    or profile.get("source_sha256") != digest
                    or profile.get("status") not in ("collected", "failed")
                ):
                    missing.append(f"{trial_id}:{group}")
        if missing:
            raise ValueError(
                "Required profiling attempts missing or stale: "
                + ", ".join(missing)
                + ". Run xe-forge-skill profile <trial_file> --kernel-name <name> --trial-id <id>"
                " on the saved trial; a failed attempt is recorded and counts, do not fabricate results."
            )

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
        verdict: str | None = None,
    ) -> dict:
        """Record benchmark results for a trial. Returns the trial dict.

        *speedup* is absent when the host gated the comparison; *verdict* is the gate
        it named. Both are recorded as given -- no ratio is invented for a gate. See
        :func:`_rank` for how a gated trial still orders against the others.
        """
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

        if verdict is not None:
            trial["verdict"] = verdict
            if speedup is None:
                # Gated comparison: drop any stale ratio from an earlier measurement.
                trial["speedup"] = None

        if trial["validation"] == "fail" or trial["correctness"] == "fail":
            trial["status"] = "failed"
        elif _rank(trial) is not None:
            # A gated comparison is a finished measurement, not a half-recorded one:
            # both arms ran and the answer was "not resolvable here".
            trial["status"] = "completed"
        else:
            trial["status"] = "partial"

        best_rank: tuple[float, int, float] | None = None
        best_id = None
        for tid, t in state["trials"].items():
            rank = _rank(t)
            if rank is not None and (best_rank is None or rank > best_rank):
                best_rank = rank
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

    @staticmethod
    def keepable(trial: dict) -> bool:
        """Whether *trial* is something :meth:`finalize` would write out.

        A measured regression is not; parity, measured or gated, is.
        """
        rank = _rank(trial)
        return rank is not None and rank[0] >= 1.0

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

        Returns the best trial id, or None if there is nothing worth finalizing.

        A trial slower than the baseline is never finalized, even when it is the only
        one that ranks: the point of the run is a kernel to keep, and shipping a
        measured regression because nothing else was rankable is a worse outcome than
        reporting that the search found nothing. Parity *is* finalized -- a kernel that
        matches the baseline is a legitimate result, and the caller can see from the
        absent ``speedup`` that it is not a win.
        """
        state = self._load_state(kernel_name)
        best_id = state.get("best_trial")
        if best_id is None:
            logger.warning("No correct trials to finalize for '%s'", kernel_name)
            return None

        self._require_profiles(kernel_name, state)
        best = state["trials"][best_id]
        if not self.keepable(best):
            logger.warning(
                "Best trial %s for '%s' is a regression (%.2fx); nothing finalized",
                best_id,
                kernel_name,
                _rank(best)[0],
            )
            return None

        src = self._trial_dir(kernel_name) / best["file"]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # The old output is removed first, so it must not hold the workspace or the
        # trial tree, nor be a recorded trial or state.json.
        resolved = output_path.parent.resolve() / output_path.name
        trial_dir = self._trial_dir(kernel_name).resolve()
        for holds in (Path.cwd().resolve(), trial_dir):
            if holds == resolved or resolved in holds.parents:
                raise ValueError(
                    f"refusing to replace {output_path}: it contains {holds}; "
                    "name an output path of its own"
                )
        recorded = [trial_dir / t["file"] for t in state["trials"].values() if t.get("file")]
        for kept in (*recorded, self._state_path(kernel_name).resolve()):
            if kept == resolved or kept in resolved.parents:
                raise ValueError(
                    f"refusing to write {output_path}: it is part of the recorded trial "
                    f"{kept}; name an output path of its own"
                )
        if output_path.is_dir() and not output_path.is_symlink():
            shutil.rmtree(output_path)
        elif output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        if src.is_dir():
            # The winner of a multi-file trial is the whole directory. Handing back only
            # its entry point would name a kernel that cannot be rebuilt.
            shutil.copytree(src, output_path)
        else:
            shutil.copy2(src, output_path)
        speedup = best.get("speedup")
        # `%.2f` on the None a gate leaves behind would raise here, at the end of a run
        # that otherwise succeeded. Say which gate decided instead of printing a ratio.
        measured = f"{speedup:.2f}x" if speedup is not None else (best.get("verdict") or "parity")
        logger.info("Finalized %s (%s) -> %s", best_id, measured, output_path)
        return best_id

    def exists(self, kernel_name: str) -> bool:
        """Return True if a trial tree exists for *kernel_name*."""
        return self._state_path(kernel_name).exists()
