"""Copy each workspace's best keepable measured trial into ``results/`` and write SUMMARY.md."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from xe_forge.core.trial_manager import TrialManager


def _trees(workspace: Path, name: str) -> TrialManager | None:
    """The trial tree for ``name`` under ``trials*/`` in a workspace."""
    for d in [workspace / "trials", *sorted(workspace.glob("trials*"))]:
        mgr = TrialManager(d)
        if d.is_dir() and mgr.exists(name):
            return mgr
    return None


def _port_status(port_txt: Path) -> dict:
    """kind, stack verdict (None if unchecked) and mergeability from PORT.txt."""
    text = port_txt.read_text() if port_txt.is_file() else ""
    text = text.split("\nnotes:", 1)[0]  # notes are free text
    kind = next(
        (ln.split(":", 1)[1].split()[0] for ln in text.splitlines() if ln.startswith("kind:")), None
    )
    verdicts = [
        ln.split(":", 1)[1].split()[0] for ln in text.splitlines() if ln.startswith("stack check ")
    ]
    stack = None if not verdicts else ("PASS" if all(v == "PASS" for v in verdicts) else "FAIL")
    return {"kind": kind, "stack": stack, "mergeable": kind == "upstream" and stack == "PASS"}


def collect_one(
    workspace: Path, name: str, variant: str, results: Path, key: str | None = None
) -> dict:
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
    dest = results / f"{key or name}__{variant}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    output = workspace / "output"
    if output.is_dir():
        shutil.copytree(output, dest / "output")
        row["patches"] = sorted(str(p.relative_to(dest)) for p in dest.rglob("*.patch"))
        row.update(_port_status(dest / "output" / "PORT.txt"))
        if row["stack"] == "FAIL":
            for p in dest.rglob("*.patch"):
                p.rename(p.with_suffix(".patch.rejected"))
            row["patches"] = []
            row["outcome"] += " (stack check FAIL; patch rejected)"
    else:
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
    """Collect the attempted (kernel, variant) ``runs``; write results/SUMMARY.md."""
    from xe_forge.frontend.manifest import key as key_of

    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    names = {key_of(e): e["name"] for e in manifest["kernels"]}
    attempted = {}
    for k, variant in runs:
        attempted[(k, variant)] = collect_one(
            run_dir / "workspaces" / f"{k}__{variant}", names.get(k, k), variant, results, k
        )

    lines = [
        "# Results",
        "",
        "| kernel | repo | variant | decision | outcome | speedup | trial | kind | stack | mergeable | patches |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["kernels"]:
        name = entry["name"]
        mine = [(v, r) for (k, v), r in attempted.items() if k == key_of(entry)]
        if not mine:
            decision = "optimize" if entry.get("enabled") else "skip"
            lines.append(
                f"| `{name}` | {entry.get('repo') or '-'} | - | {decision} | "
                f"not run: {entry.get('reason', '')} | - | - | - | - | - | - |"
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
                f"{speedup} | {r.get('trial', '-')} | {r.get('kind') or '-'} | "
                f"{r.get('stack') or '-'} | {'yes' if r.get('mergeable') else 'no'} | {patches} |"
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
