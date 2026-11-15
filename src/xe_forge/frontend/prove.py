"""Prove winning patches in the model: an agent builds both stacks, the run compares them."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from xe_forge.frontend import agent

PARSE_REFUSED = 3  # recapture: the launcher ran, its capture did not parse


def _why(arm_dir: Path, lines: int = 6) -> str:
    """The reason an arm's capture failed, from whichever log holds it."""
    for name in ("launcher.log", "unitrace.log", "capture_run.log"):
        log = arm_dir / name
        if not log.is_file():
            continue
        hits = [
            ln.strip()
            for ln in log.read_text(errors="replace").splitlines()
            if any(
                w in ln
                for w in (
                    "Error",
                    "error:",
                    "Timeout",
                    "FAIL",
                    "Traceback",
                    "out of memory",
                    "not become ready",
                )
            )
        ]
        if hits:
            return f"{name}: " + " | ".join(hits[-lines:])[:600]
    return "no error line in the arm's logs"


def _build_arms(
    prove_dir: Path,
    patches: list[Path],
    python: str,
    max_turns: int,
    launcher: Path | None,
    stack: Path | None = None,
) -> dict:
    ctx = {"patches": patches, "launcher": launcher if launcher and launcher.is_file() else None,
           "stack": stack if stack and stack.is_file() else None}  # fmt: skip
    arms = agent.run("frontend-build-arms.md.j2", ctx, prove_dir, "arms.json", python, max_turns)
    if set(arms) != {"A", "B"} or arms["A"] == arms["B"]:
        raise RuntimeError(f"build agent gave no two distinct arms: {arms} (see {prove_dir})")
    for arm, python_ in arms.items():
        if not (
            isinstance(python_, str) and Path(python_).is_file() and os.access(python_, os.X_OK)
        ):
            raise RuntimeError(f"arm {arm}: {python_!r} is not an interpreter (see {prove_dir})")
    return arms


def winners(results: Path) -> list[Path]:
    """The winning patches in ``results/``, one kernel's best each: variants' overlap."""
    summary = results / "summary.json"
    rows = json.loads(summary.read_text()) if summary.is_file() else []
    best: dict[str, dict] = {}
    for r in rows:
        kernel = r.get("kernel", r.get("result"))
        if str(r.get("outcome", "")).startswith("win") and r.get("patches"):
            if (r.get("speedup") or 0) >= (best.get(kernel, {}).get("speedup") or 0):
                best[kernel] = r
    return [Path(r["result"]) / p for r in best.values() for p in r["patches"]]


def prove(run_dir: Path, recapture, python: str, repeat: int, max_turns: int) -> str:
    """Build, capture both arms, compare; returns the SUMMARY.md section. The fusions'
    patches (fuse/), made on the kernels' patched repositories, apply after them."""
    from xe_forge.frontend import compare

    fused = winners(run_dir / "fuse" / "results")
    patches = winners(run_dir / "results") + fused
    if not patches:
        return "## In model\n\nnot run: no winning patch to prove\n"
    prove_dir = run_dir / "prove"
    prove_dir.mkdir(parents=True, exist_ok=True)
    try:
        arms = _build_arms(prove_dir, patches, python, max_turns, run_dir / "launcher" / "run.sh",
                           run_dir / "fuse" / "stacked" / "STACK.json" if fused else None)  # fmt: skip
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return f"## In model\n\nnot run: {exc}\n"
    caps: dict[str, list[Path]] = {"A": [], "B": []}
    for i in range(1, repeat + 1):
        for arm in ("A", "B"):  # alternate, so drift falls on both arms
            path = prove_dir / f"{arm}-r{i}" / "capture.json"
            # The run's own cached launcher, under the arm's interpreter: identical work.
            code = recapture(run_dir, path.parent, arms[arm])
            if code not in (0, PARSE_REFUSED) or (code == 0 and not path.is_file()):
                # One retry, and not a blind one: the commonest failure here is a startup
                # timeout loading a large model off NFS from cold, and the attempt that
                # just failed is what warms the page cache. An LTX arm died with
                # "Orchestrator did not become ready within 600s" on an 89 GB model whose
                # own capture had succeeded warm.
                print(f"prove: arm {arm} run {i} failed, retrying once ({_why(path.parent)})",
                      flush=True)  # fmt: skip
                code = recapture(run_dir, path.parent, arms[arm])
            if code == PARSE_REFUSED and arm == "B" and caps["A"]:
                return (f"## In model\n\nnot run: arm B ran but its capture does not parse -- "
                        f"the launcher does not hold on the patched build ({path.parent})\n")  # fmt: skip
            if code != 0 or not path.is_file():
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
