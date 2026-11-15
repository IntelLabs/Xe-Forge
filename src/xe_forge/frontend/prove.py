"""Prove winning patches in the model: an agent builds both stacks, the run compares them."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from xe_forge.frontend import agent


def _why(arm_dir: Path, lines: int = 6) -> str:
    """The reason an arm's capture failed, from whichever log holds it."""
    for name in ("launcher.log", "unitrace.log", "capture_run.log"):
        log = arm_dir / name
        if not log.is_file():
            continue
        hits = [
            ln.strip() for ln in log.read_text(errors="replace").splitlines()
            if any(w in ln for w in ("Error", "error:", "Timeout", "FAIL", "Traceback",
                                     "out of memory", "not become ready"))
        ]
        if hits:
            return f"{name}: " + " | ".join(hits[-lines:])[:600]
    return "no error line in the arm's logs"


def _build_arms(prove_dir: Path, patches: list[Path], python: str, max_turns: int) -> dict:
    arms = agent.run(
        "frontend-build-arms.md.j2", {"patches": patches}, prove_dir, "arms.json", python, max_turns
    )
    if set(arms) != {"A", "B"} or arms["A"] == arms["B"]:
        raise RuntimeError(f"build agent gave no two distinct arms: {arms} (see {prove_dir})")
    for arm, python_ in arms.items():
        if not (
            isinstance(python_, str) and Path(python_).is_file() and os.access(python_, os.X_OK)
        ):
            raise RuntimeError(f"arm {arm}: {python_!r} is not an interpreter (see {prove_dir})")
    return arms


def prove(run_dir: Path, recapture, python: str, repeat: int, max_turns: int) -> str:
    """Build, capture both arms, compare; returns the SUMMARY.md section."""
    from xe_forge.frontend import compare

    rows = json.loads((run_dir / "results" / "summary.json").read_text())
    best: dict[str, dict] = {}  # one per kernel: variants' patches overlap
    for r in rows:
        kernel = r.get("kernel", r.get("result"))
        if str(r.get("outcome", "")).startswith("win") and r.get("patches"):
            if (r.get("speedup") or 0) >= (best.get(kernel, {}).get("speedup") or 0):
                best[kernel] = r
    patches = [Path(r["result"]) / p for r in best.values() for p in r["patches"]]
    if not patches:
        return "## In model\n\nnot run: no winning patch to prove\n"
    prove_dir = run_dir / "prove"
    prove_dir.mkdir(parents=True, exist_ok=True)
    try:
        arms = _build_arms(prove_dir, patches, python, max_turns)
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return f"## In model\n\nnot run: {exc}\n"
    caps: dict[str, list[Path]] = {"A": [], "B": []}
    for i in range(1, repeat + 1):
        for arm in ("A", "B"):  # alternate, so drift falls on both arms
            path = prove_dir / f"{arm}-r{i}" / "capture.json"
            # The run's own cached launcher, under the arm's interpreter: identical work.
            if recapture(run_dir, path.parent, arms[arm]) != 0 or not path.is_file():
                # One retry, and not a blind one: the commonest failure here is a startup
                # timeout loading a large model off NFS from cold, and the attempt that
                # just failed is what warms the page cache. An LTX arm died with
                # "Orchestrator did not become ready within 600s" on an 89 GB model whose
                # own capture had succeeded warm.
                why = _why(path.parent)
                print(f"prove: arm {arm} run {i} failed, retrying once ({why})", flush=True)
                if recapture(run_dir, path.parent, arms[arm]) != 0 or not path.is_file():
                    if arm == "B" and caps["A"]:
                        return (f"## In model\n\nFAIL: with the patches the model does not "
                                f"run where the base does ({path.parent}); the patches are "
                                f"not mergeable\n\n```\n{_why(path.parent)}\n```\n")  # fmt: skip
                    # The reason belongs in the report. A path alone sends the reader
                    # hunting through logs for a line the run already had.
                    return (f"## In model\n\nnot run: capture of arm {arm} failed "
                            f"({path.parent})\n\n```\n{_why(path.parent)}\n```\n")  # fmt: skip
            caps[arm].append(path)
    table = compare.compare(caps["A"], caps["B"])
    out = compare.write(table, prove_dir)
    t = table[0]
    noise = max(t["spread_a"] or 0.0, t["spread_b"] or 0.0)
    verdict = (
        f"x{t['speedup']}"
        if t["speedup"] and abs(t["speedup"] - 1) > noise
        else f"no difference beyond the captures' spread ({noise:.1%})"
    )
    return (
        f"## In model\n\n{len(patches)} patch(es), {repeat} captures per arm: total "
        f"{t['us_a']} -> {t['us_b']} us device time, {verdict}; per workload: "
        f"`{out.relative_to(run_dir)}`\n"
    )
