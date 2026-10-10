"""Gather what the Xe-Forge runs produced into ``results/``, by the trial trees' own rule.

For each (kernel, variant) workspace the trial tree is asked for its best *measured*
non-baseline trial (``TrialManager.best_measured``); it is kept when ``keepable`` -- the
test ``finalize`` and ``upstream patch`` apply, so a measured regression is never copied
out. A kept result gets the workspace's ``output/`` (the finalized kernel, and the
port-back patch when the session wrote one) and a ``result.json``; everything else is a
row in SUMMARY.md saying why there is nothing to keep.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from xe_forge.core.trial_manager import TrialManager


def _trees(workspace: Path, name: str) -> TrialManager | None:
    """The trial tree for ``name`` in a workspace (default ``trials/``; any other
    ``trials*`` directory a configuration chose)."""
    for d in [workspace / "trials", *sorted(workspace.glob("trials*"))]:
        mgr = TrialManager(d)
        if d.is_dir() and mgr.exists(name):
            return mgr
    return None


def collect_one(workspace: Path, name: str, variant: str, results: Path) -> dict:
    row = {"kernel": name, "variant": variant, "workspace": str(workspace)}
    status = workspace.with_suffix(".status.json")
    if status.is_file():
        row.update(json.loads(status.read_text()))
    if not workspace.is_dir():
        row.setdefault("outcome", "not run")
        return row
    mgr = _trees(workspace, name)
    if mgr is None:
        row["outcome"] = "no trials"
        return row
    best = mgr.best_measured(name)
    if best is None:
        row["outcome"] = "no measured trial"
        return row
    row.update(
        trial=best["id"],
        speedup=best.get("speedup"),
        baseline_us=best.get("baseline_us"),
        verdict=best.get("verdict"),
        strategy=best.get("strategy"),
    )
    if not mgr.keepable(best):
        row["outcome"] = "no win (best measured trial is a regression)"
        return row
    speedup = best.get("speedup")
    row["outcome"] = "win" if speedup and speedup > 1.0 else "parity"
    dest = results / f"{name}__{variant}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    output = workspace / "output"
    if output.is_dir():
        shutil.copytree(output, dest / "output")
        row["patches"] = sorted(str(p.relative_to(dest)) for p in dest.rglob("*.patch"))
    else:
        # Not finalized by the session (e.g. it ran out of turns): the measured trial's own
        # file is kept, but nothing was ported back and the outcome says so.
        row["outcome"] += " (measured trial, not finalized; no patch)"
        src = Path(best["file_path"])
        if src.is_dir():
            shutil.copytree(src, dest / "output" / src.name)
        elif src.is_file():
            (dest / "output").mkdir()
            shutil.copy2(src, dest / "output" / src.name)
        row["patches"] = []
    row["result"] = str(dest)
    (dest / "result.json").write_text(json.dumps(row, indent=1, default=str) + "\n")
    return row


def collect(run_dir: Path, manifest: dict, runs: list[tuple[str, str]]) -> Path:
    """``runs``: the (kernel, variant) pairs optimize attempted. Writes results/SUMMARY.md
    with one row per manifest entry, attempted or not."""
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    attempted = {}
    for name, variant in runs:
        attempted[(name, variant)] = collect_one(
            run_dir / "workspaces" / f"{name}__{variant}", name, variant, results
        )

    lines = [
        "# Results",
        "",
        "| kernel | repo | variant | decision | outcome | speedup | trial | patches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["kernels"]:
        name = entry["name"]
        mine = [(v, r) for (n, v), r in attempted.items() if n == name]
        if not mine:
            decision = "optimize" if entry.get("enabled") else "skip"
            lines.append(
                f"| `{name}` | {entry.get('repo') or '-'} | - | {decision} | "
                f"not run: {entry.get('reason', '')} | - | - | - |"
            )
            continue
        for variant, r in mine:
            outcome = r.get("outcome", "?")
            if r.get("skipped"):
                outcome = f"skipped: {r['skipped']}"
            elif r.get("exit_code") not in (None, 0) and outcome != "win":
                outcome += f" (xe-forge exit {r['exit_code']})"
            speedup = f"{r['speedup']:.3f}x" if isinstance(r.get("speedup"), (int, float)) else "-"
            patches = ", ".join(r.get("patches") or []) or "-"
            lines.append(
                f"| `{name}` | {entry.get('repo')} | {variant} | optimize | {outcome} | "
                f"{speedup} | {r.get('trial', '-')} | {patches} |"
            )
    summary = results / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n")
    (results / "summary.json").write_text(
        json.dumps(
            [{"kernel": n, "variant": v, **r} for (n, v), r in attempted.items()],
            indent=1,
            default=str,
        )
        + "\n"
    )
    return summary
