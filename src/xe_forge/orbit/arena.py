"""
The agent arena — engine A/B over candidate-directory tasks, outside the
optimization loop. Contestants optimize isolated copies, every pair persists a
resumable `result.json`, and unmeasured speedups are reported as such, never
ranked against measured ones. Design rationale: docs/DESIGN.md
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from xe_forge.orbit.optimize.kernel_dir import (
    KERNEL_FILE,
    SPEC_FILE,
    optimize_kernel_dir,
)

# Per-task arena configuration, next to kernel.py in the candidate directory.
ARENA_CONFIG_FILE = "arena.yaml"
# The persisted outcome of one (contestant, task) pair; its presence is what resume keys on.
RESULT_FILE = "result.json"
DEFAULT_TRAIN_VARIANT = "bench-gpu"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArenaError(RuntimeError):
    """Raised when the arena's inputs are malformed, not when a contestant fails."""


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@dataclass
class ArenaTask:
    """One kernel task: a candidate directory plus the variants that score it."""

    task_id: str
    candidate_dir: Path
    train_variant: str = DEFAULT_TRAIN_VARIANT
    heldout_variants: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def discover_tasks(task_root: Path) -> list[ArenaTask]:
    """Every subdirectory holding a resolvable candidate is a task.

    A subdirectory without `kernel.py` + `spec.yaml` is skipped rather than errored:
    a task root legitimately carries other directories (the arena's own workspace,
    for one). Dot-directories are never tasks.
    """
    root = Path(task_root)
    if not root.is_dir():
        raise ArenaError(f"not a task root: {root}")

    tasks: list[ArenaTask] = []
    for directory in sorted(d for d in root.iterdir() if d.is_dir()):
        if directory.name.startswith("."):
            continue
        if not (directory / KERNEL_FILE).is_file() or not (directory / SPEC_FILE).is_file():
            continue
        tasks.append(_load_task(directory))
    return tasks


def _load_task(directory: Path) -> ArenaTask:
    """Build a task from its directory, reading `arena.yaml` when present."""
    task = ArenaTask(task_id=directory.name, candidate_dir=directory)
    config_path = directory / ARENA_CONFIG_FILE

    if not config_path.is_file():
        task.notes.append(
            f"no {ARENA_CONFIG_FILE}: training on the default {DEFAULT_TRAIN_VARIANT!r} "
            f"variant with no held-out variants, so the generalization gap cannot be "
            f"measured for this task"
        )
        return task

    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ArenaError(f"{config_path} must be a mapping with 'train' and 'heldout' keys")

    train = raw.get("train")
    if train is not None:
        task.train_variant = str(train)
    heldout_raw = raw.get("heldout") or []
    if not isinstance(heldout_raw, list):
        raise ArenaError(f"{config_path}: 'heldout' must be a list of variant names")
    task.heldout_variants = [str(v) for v in heldout_raw]

    if task.train_variant in task.heldout_variants:
        # A variant the contestant trains on cannot also be held out.
        raise ArenaError(
            f"{config_path}: train variant {task.train_variant!r} also listed as held-out; "
            f"a shape the contestant optimizes against cannot measure generalization"
        )
    if not task.heldout_variants:
        task.notes.append(
            f"{ARENA_CONFIG_FILE} declares no held-out variants; the generalization gap "
            f"cannot be measured for this task"
        )
    return task


# ---------------------------------------------------------------------------
# contestants and results
# ---------------------------------------------------------------------------


@dataclass
class ContestantResult:
    """What one contestant did with one task.

    `train_speedup` is None whenever the engine did not measure one (a dry run, or
    the fire-and-forget Claude path) — the report renders that as "unmeasured"
    rather than folding it into a mean. Each held-out variant maps to a speedup or
    to None with the same semantics.
    """

    task_id: str
    contestant: str
    succeeded: bool
    train_speedup: float | None = None
    heldout_speedups: dict[str, float | None] = field(default_factory=dict)
    error: str = ""
    workspace: Path | None = None


def default_runner(
    contestant: Contestant, task: ArenaTask, workspace_dir: Path
) -> ContestantResult:
    """Copy the candidate into the workspace and optimize the copy.

    The shared task directory is read, never written. Held-out variants are
    resolved with `dry_run=True`, so their speedups stay None here; a runner with
    hardware access replaces this callable and fills the numbers in.
    """
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    candidate_copy = workspace_dir / "candidate"
    if candidate_copy.exists():
        shutil.rmtree(candidate_copy)
    shutil.copytree(task.candidate_dir, candidate_copy)

    engine = str(contestant.config.get("engine", contestant.name))
    outcome = optimize_kernel_dir(
        candidate_copy,
        engine=engine,
        device=str(contestant.config.get("device", "xpu")),
        variant=task.train_variant,
        dsl=str(contestant.config.get("dsl", "triton")),
        dry_run=bool(contestant.config.get("dry_run", False)),
    )

    heldout: dict[str, float | None] = {}
    for heldout_variant in task.heldout_variants:
        held = optimize_kernel_dir(
            candidate_copy, engine=engine, variant=heldout_variant, dry_run=True
        )
        heldout[heldout_variant] = held.speedup  # a dry run never measures: stays None

    return ContestantResult(
        task_id=task.task_id,
        contestant=contestant.name,
        succeeded=bool(outcome.success),
        train_speedup=outcome.speedup,
        heldout_speedups=heldout,
        error="" if outcome.success else (outcome.detail or "; ".join(outcome.notes)),
        workspace=workspace_dir,
    )


Runner = Callable[["Contestant", ArenaTask, Path], ContestantResult]


@dataclass
class Contestant:
    """An engine configuration under test. The runner is injectable so CI never
    needs a GPU or an LLM to exercise the arena itself."""

    name: str
    config: dict = field(default_factory=dict)
    runner: Runner = default_runner


def contestant_available(name: str) -> tuple[bool, str]:
    """Check the thing the engine actually needs, not a proxy for it.

    `claude` shells out to the binary; every other engine is routed by
    `optimize_kernel_dir` through Xe-Forge's pipeline, which needs DSPy, ai_bench
    and torch importable.
    """
    if name == "claude":
        if shutil.which("claude") is None:
            return False, "the `claude` binary is not on PATH"
        return True, ""
    try:
        import xe_forge.pipeline  # noqa: F401
    except Exception as exc:
        return False, f"Xe-Forge's pipeline is not importable ({exc})"
    return True, ""


def build_contestants(names: Iterable[str]) -> tuple[list[Contestant], dict[str, str]]:
    """Turn engine names into contestants, skipping the unavailable with a reason."""
    contestants: list[Contestant] = []
    skipped: dict[str, str] = {}
    for name in names:
        ok, reason = contestant_available(name)
        if ok:
            contestants.append(Contestant(name=name, config={"engine": name}))
        else:
            skipped[name] = reason
    return contestants, skipped


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def run_arena(
    tasks: list[ArenaTask],
    contestants: list[Contestant],
    arena_dir: Path,
    *,
    resume: bool = True,
    skipped: dict[str, str] | None = None,
) -> ArenaReport:
    """Run every (contestant, task) pair in its own workspace under `arena_dir`.

    A pair with a persisted, matching `result.json` is loaded instead of re-run;
    the report carries the resumed count. One contestant crashing on one task is
    recorded as that pair's failure and stops nothing else.
    """
    arena_dir = Path(arena_dir)
    names = [c.name for c in contestants]
    if len(set(names)) != len(names):
        raise ArenaError(f"duplicate contestant names: {names}")
    for name in names:
        if not _NAME_RE.match(name):
            raise ArenaError(f"invalid contestant name: {name!r}")

    results: list[ContestantResult] = []
    resumed = 0
    for contestant in contestants:
        for task in tasks:
            pair_dir = arena_dir / contestant.name / task.task_id
            result_path = pair_dir / RESULT_FILE

            if resume and result_path.is_file():
                loaded = _load_result(result_path, task, contestant.name)
                if loaded is not None:
                    results.append(loaded)
                    resumed += 1
                    continue
                # A result that does not verifiably belong to this pair is re-run.

            workspace = pair_dir / "workspace"
            try:
                result = contestant.runner(contestant, task, workspace)
            except Exception as exc:
                result = ContestantResult(
                    task_id=task.task_id,
                    contestant=contestant.name,
                    succeeded=False,
                    heldout_speedups=dict.fromkeys(task.heldout_variants),
                    error=f"{type(exc).__name__}: {exc}",
                    workspace=workspace,
                )
            if result.workspace is None:
                result.workspace = workspace
            _save_result(result_path, result)
            results.append(result)

    return ArenaReport(
        results=results,
        resumed=resumed,
        task_count=len(tasks),
        skipped=dict(skipped or {}),
    )


def _save_result(path: Path, result: ContestantResult) -> None:
    payload = {
        "task_id": result.task_id,
        "contestant": result.contestant,
        "succeeded": result.succeeded,
        "train_speedup": result.train_speedup,
        "heldout_speedups": result.heldout_speedups,
        "error": result.error,
        "workspace": str(result.workspace) if result.workspace else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _as_speedup(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _load_result(path: Path, task: ArenaTask, contestant_name: str) -> ContestantResult | None:
    """Load a persisted pair result, or None when it cannot be trusted.

    The identity check matters: a `result.json` copied or renamed into the wrong
    pair directory would otherwise resume cleanly as somebody else's score.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("task_id") != task.task_id or raw.get("contestant") != contestant_name:
        return None
    heldout_raw = raw.get("heldout_speedups")
    heldout = (
        {str(k): _as_speedup(v) for k, v in heldout_raw.items()}
        if isinstance(heldout_raw, dict)
        else {}
    )
    return ContestantResult(
        task_id=task.task_id,
        contestant=contestant_name,
        succeeded=bool(raw.get("succeeded", False)),
        train_speedup=_as_speedup(raw.get("train_speedup")),
        heldout_speedups=heldout,
        error=str(raw.get("error", "")),
        workspace=Path(str(raw["workspace"])) if raw.get("workspace") else None,
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


@dataclass
class _ContestantStats:
    name: str
    attempted: int
    solved: int
    train_mean: float | None
    heldout_mean: float | None
    gap: float | None
    heldout_slots: int
    heldout_unmeasured: int


@dataclass
class ArenaReport:
    """The arena's outcome: raw results plus an aggregation that refuses to
    compare measured numbers with unmeasured ones."""

    results: list[ContestantResult]
    resumed: int = 0
    task_count: int = 0
    skipped: dict[str, str] = field(default_factory=dict)

    def _stats(self) -> list[_ContestantStats]:
        order: list[str] = []
        by_name: dict[str, list[ContestantResult]] = {}
        for result in self.results:
            if result.contestant not in by_name:
                order.append(result.contestant)
                by_name[result.contestant] = []
            by_name[result.contestant].append(result)

        stats: list[_ContestantStats] = []
        for name in order:
            rows = by_name[name]
            solved = [r for r in rows if r.succeeded]
            train_values = [r.train_speedup for r in solved if r.train_speedup is not None]
            heldout_values = [
                v for r in solved for v in r.heldout_speedups.values() if v is not None
            ]
            heldout_slots = sum(len(r.heldout_speedups) for r in solved)

            # The gap is computed only over tasks where BOTH sides were measured.
            paired = [
                r
                for r in solved
                if r.train_speedup is not None
                and any(v is not None for v in r.heldout_speedups.values())
            ]
            gap: float | None = None
            if paired:
                paired_train = fmean(r.train_speedup for r in paired)  # type: ignore[misc]
                paired_heldout = fmean(
                    v for r in paired for v in r.heldout_speedups.values() if v is not None
                )
                gap = paired_train - paired_heldout

            stats.append(
                _ContestantStats(
                    name=name,
                    attempted=len(rows),
                    solved=len(solved),
                    train_mean=fmean(train_values) if train_values else None,
                    heldout_mean=fmean(heldout_values) if heldout_values else None,
                    gap=gap,
                    heldout_slots=heldout_slots,
                    heldout_unmeasured=heldout_slots - len(heldout_values),
                )
            )
        return stats

    def _ranked(self) -> list[tuple[int | None, _ContestantStats]]:
        """Rank only over commensurable numbers.

        Contestants with a measured mean train speedup are ranked among themselves;
        a contestant whose speedups are unmeasured is listed unranked below, however
        many tasks it solved — an unmeasured mean is not a small mean.
        """
        stats = self._stats()
        measured = [s for s in stats if s.train_mean is not None]
        unmeasured = [s for s in stats if s.train_mean is None]
        measured.sort(key=lambda s: (-s.train_mean, -s.solved, s.name))
        unmeasured.sort(key=lambda s: (-s.solved, s.name))
        ranked: list[tuple[int | None, _ContestantStats]] = [
            (i + 1, s) for i, s in enumerate(measured)
        ]
        ranked.extend((None, s) for s in unmeasured)
        return ranked

    def format(self) -> str:
        """Render the leaderboard."""
        stats = self._stats()
        lines = [
            f"agent arena: {len(stats)} contestant(s) x {self.task_count} task(s) "
            f"({len(self.results)} pairs, {self.resumed} resumed)",
            "",
            f"{'rank':<6}{'contestant':<16}{'solved':<9}{'train mean':<13}"
            f"{'heldout mean':<15}{'gen. gap':<10}",
        ]
        for rank, s in self._ranked():
            if s.heldout_slots == 0:
                heldout = "no heldout"
            elif s.heldout_mean is None:
                heldout = "unmeasured"
            else:
                heldout = f"{s.heldout_mean:.2f}x"
                if s.heldout_unmeasured:
                    heldout += f" ({s.heldout_unmeasured} unmeasured)"
            lines.append(
                f"{rank if rank is not None else '-':<6}"
                f"{s.name:<16}"
                f"{f'{s.solved}/{s.attempted}':<9}"
                f"{f'{s.train_mean:.2f}x' if s.train_mean is not None else 'unmeasured':<13}"
                f"{heldout:<15}"
                f"{f'{s.gap:+.2f}' if s.gap is not None else 'unmeasured':<10}"
            )

        failures = [r for r in self.results if not r.succeeded and r.error]
        if failures:
            lines.append("")
            lines.append("failures:")
            lines.extend(f"  {r.contestant}/{r.task_id}: {r.error[:200]}" for r in failures)

        if self.skipped:
            lines.append("")
            lines.extend(
                f"skipped contestant: {name} ({reason})" for name, reason in self.skipped.items()
            )
        return "\n".join(lines)

    def summary(self) -> dict:
        """JSON-serializable aggregate, mirroring format() without the layout."""
        return {
            "tasks": self.task_count,
            "pairs": len(self.results),
            "resumed": self.resumed,
            "skipped": dict(self.skipped),
            "contestants": {
                s.name: {
                    "attempted": s.attempted,
                    "solved": s.solved,
                    "train_mean_speedup": s.train_mean,
                    "heldout_mean_speedup": s.heldout_mean,
                    "generalization_gap": s.gap,
                    "heldout_slots": s.heldout_slots,
                    "heldout_unmeasured": s.heldout_unmeasured,
                }
                for s in self._stats()
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "contestant": r.contestant,
                    "succeeded": r.succeeded,
                    "train_speedup": r.train_speedup,
                    "heldout_speedups": r.heldout_speedups,
                    "error": r.error,
                    "workspace": str(r.workspace) if r.workspace else None,
                }
                for r in self.results
            ],
        }
