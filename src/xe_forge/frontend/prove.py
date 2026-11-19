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


def applied(row: dict) -> bool:
    """A summary row whose kernel's win is on the stacked repositories: a win with patches."""
    return str(row.get("outcome", "")).startswith("win") and bool(row.get("patches"))


def winners(results: Path) -> list[Path]:
    """The winning patches in ``results/``, one kernel's best each: variants' overlap."""
    summary = results / "summary.json"
    rows = json.loads(summary.read_text()) if summary.is_file() else []
    best: dict[str, dict] = {}
    for r in rows:
        kernel = r.get("kernel", r.get("result"))
        if applied(r):
            if (r.get("speedup") or 0) >= (best.get(kernel, {}).get("speedup") or 0):
                best[kernel] = r
    return [Path(r["result"]) / p for r in best.values() for p in r["patches"]]


def prove(run_dir: Path, recapture, python: str, repeat: int, max_turns: int) -> str:
    """Build, capture both arms, compare; returns the SUMMARY.md section. The fusions'
    patches (fuse/), made on the kernels' patched repositories, apply after them.

    When the patches together do not prove, each is proved alone: one that fails alone is
    marked in the results table and the rest are proved without it."""
    fused = winners(run_dir / "fuse" / "results")
    patches = winners(run_dir / "results") + fused
    if not patches:
        return "## In model\n\nnot run: no winning patch to prove\n"
    stack = run_dir / "fuse" / "stacked" / "STACK.json" if fused else None
    section, got = _prove(run_dir, run_dir / "prove", patches, recapture, python, repeat,
                          max_turns, stack)  # fmt: skip
    # Only the patched side failing is the patches'; a fusion is made on the kernels'
    # patches, so it has no "alone".
    if got != "patched" or fused:
        return section
    note = "not proven in the model: the model does not run with it, see prove/"
    if len(patches) == 1:
        _mark(run_dir / "results", patches[0], note)
        return section
    alone = {}
    for i, p in enumerate(patches, 1):
        alone[p] = _prove(run_dir, run_dir / "prove" / f"alone-{i}", [p], recapture, python,
                          repeat, max_turns)  # fmt: skip
    bad = [p for p, (_, why) in alone.items() if why == "patched"]
    if not bad:  # each holds alone and together they do not: no one patch is to blame
        return section
    for p in bad:
        _mark(run_dir / "results", p, note)
    lines = [f"- `{p.relative_to(run_dir)}` alone: {alone[p][0].split(chr(10), 2)[-1].strip()}"
             for p in bad]  # fmt: skip
    rest = [p for p in patches if p not in bad]
    if not rest:
        tail = "nothing left to prove.\n"
    elif len(rest) == 1:
        tail = alone[rest[0]][0].split("\n", 2)[-1]
    else:
        tail = _prove(run_dir, run_dir / "prove" / "without-failed", rest, recapture, python,
                      repeat, max_turns)[0].split("\n", 2)[-1]  # fmt: skip
    return (
        "## In model\n\nThe patches together did not prove; each was proved alone.\n\n"
        + "\n".join(lines)
        + f"\n\nWithout {'it' if len(bad) == 1 else 'them'}: {tail}"
    )


def _mark(results: Path, patch: Path, note: str) -> None:
    """Say on ``patch``'s row, in summary.json and SUMMARY.md, that it did not prove."""
    rows_path, table = results / "summary.json", results / "SUMMARY.md"
    if not rows_path.is_file():
        return
    rows = json.loads(rows_path.read_text())
    for r in rows:
        if r.get("result") and any(Path(r["result"]) / q == patch for q in r.get("patches") or []):
            r["outcome"], r["mergeable"] = f"{r.get('outcome')} ({note})", False
            if table.is_file():
                head = f"| `{r.get('kernel')}` |"
                lines = []
                for ln in table.read_text().splitlines():
                    cells = ln.split(" | ")
                    if ln.startswith(head) and len(cells) == 11 and cells[2] == r.get("variant"):
                        cells[4], cells[9] = r["outcome"], "no"
                        ln = " | ".join(cells)
                    lines.append(ln)
                table.write_text("\n".join(lines) + "\n")
    rows_path.write_text(json.dumps(rows, indent=1, default=str) + "\n")


def _prove(run_dir: Path, prove_dir: Path, patches: list[Path], recapture, python: str,
           repeat: int, max_turns: int, stack: Path | None = None) -> tuple[str, str]:  # fmt: skip
    """(the SUMMARY.md section, "ok", "patched" -- the patched arm could not be built or did not
    run --, "launcher": it ran but the launcher does not hold on it, or "base": the unpatched
    arm did not run) for one stack of ``patches``."""
    from xe_forge.frontend import compare

    prove_dir.mkdir(parents=True, exist_ok=True)
    try:
        arms = _build_arms(prove_dir, patches, python, max_turns, run_dir / "launcher" / "run.sh",
                           stack)  # fmt: skip
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return f"## In model\n\nnot run: {exc}\n", "patched"
    caps: dict[str, list[Path]] = {"A": [], "B": []}
    for i in range(1, repeat + 1):
        for arm in ("A", "B"):  # alternate, so drift falls on both arms
            path = prove_dir / f"{arm}-r{i}" / "capture.json"
            # The run's own cached launcher, under the arm's interpreter: identical work.
            code = recapture(run_dir, path.parent, arms[arm])
            if code not in (0, PARSE_REFUSED) or (code == 0 and not path.is_file()):
                # One retry, and not a blind one: the commonest failure here is a startup
                # timeout loading a large model off NFS from cold, and the attempt that
                # just failed is what warms the page cache.
                print(f"prove: arm {arm} run {i} failed, retrying once ({_why(path.parent)})",
                      flush=True)  # fmt: skip
                code = recapture(run_dir, path.parent, arms[arm])
            if code == PARSE_REFUSED and arm == "B" and caps["A"]:
                return (f"## In model\n\nnot run: arm B ran but its capture does not parse -- "
                        f"the launcher does not hold on the patched build ({path.parent})\n",
                        "launcher")  # fmt: skip
            if code != 0 or not path.is_file():
                if arm == "B" and caps["A"]:
                    return (f"## In model\n\nFAIL: with the patches the model does not "
                            f"run where the base does ({path.parent}); the patches are "
                            f"not mergeable\n\n```\n{_why(path.parent)}\n```\n", "patched")  # fmt: skip
                # The reason belongs in the report. A path alone sends the reader
                # hunting through logs for a line the run already had.
                return (f"## In model\n\nnot run: capture of arm {arm} failed "
                        f"({path.parent})\n\n```\n{_why(path.parent)}\n```\n",
                        "patched" if arm == "B" else "base")  # fmt: skip
            caps[arm].append(path)
    table = compare.compare(caps["A"], caps["B"])
    out = compare.write(table, prove_dir)
    from xe_forge.config import noise_band

    t = table[0]
    noise = max(t["spread_a"] or 0.0, t["spread_b"] or 0.0, noise_band())
    if t["speedup"] and abs(t["speedup"] - 1) > noise:
        verdict = f"x{t['speedup']}"
    else:
        verdict = f"no difference beyond the captures' spread or the noise band ({noise:.1%})"
    if t.get("unmatched"):
        u = t["unmatched"]
        verdict += (f" ({u['workloads']} workloads holding {u['share_a']:.1%} of A's time did "
                    f"not match; see compare.md)")  # fmt: skip
    return (
        f"## In model\n\n{len(patches)} patch(es), {repeat} captures per arm: total "
        f"{t['us_a']} -> {t['us_b']} us device time, {verdict}; per workload: "
        f"`{out.relative_to(run_dir)}`\n",
        "ok",
    )
