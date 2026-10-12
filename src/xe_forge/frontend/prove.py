"""The run's last stage: the winning patches measured in the model, not in a harness.

Building a stack with a patch is each repository's own business, so an agent does it: a
Claude session gets the patches (each with its ``PORT.txt``: repository, base commit) and
the framework's interpreter, and builds two stacks the same way -- ``A`` at the patches'
base, ``B`` with them -- writing ``arms.json``. Measuring is not delegated: the run
captures the model under each arm ``repeat`` times, in alternation, and ``compare`` joins
the two.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from xe_forge.frontend import agent

BUILD_PROMPT = """\
Build two copies of a serving stack so a patch can be measured in the model.
Patches, each with the PORT.txt beside it naming its repository and base commit:
{patches}

- Arm A is every touched repository at its base commit; arm B is the same with the patches.
  Clone fresh for each arm; never modify the repository PORT.txt names.
- Build both the same way, as the repository builds, into venv-A and venv-B here, each
  layered over the framework's venv and holding only what you rebuilt.
- Check that each arm imports its rebuilt package from its own venv and that the framework
  still imports.
- Write arms.json {{"A": "<venv-A python>", "B": "<venv-B python>"}}, or {{"error": "<why>"}}
  if an arm cannot be built, and in report.md what you built and how.
"""


def _build_arms(prove_dir: Path, patches: list[Path], python: str, max_turns: int) -> dict:
    prompt = BUILD_PROMPT.format(patches="\n".join(f"  {p}" for p in patches))
    arms = agent.run(prompt, prove_dir, "arms.json", python, max_turns)
    if set(arms) != {"A", "B"} or arms["A"] == arms["B"]:
        raise RuntimeError(f"build agent gave no two distinct arms: {arms} (see {prove_dir})")
    return arms


def prove(
    run_dir: Path, capture, capture_args: list[str], python: str, repeat: int, max_turns: int
) -> str:
    """Build, capture both arms, compare. Returns the SUMMARY.md section."""
    from xe_forge.frontend import compare

    rows = json.loads((run_dir / "results" / "summary.json").read_text())
    patches = [
        Path(r["result"]) / p
        for r in rows
        if str(r.get("outcome", "")).startswith("win")
        for p in r.get("patches") or []
    ]
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
            if capture([*capture_args, "-o", str(path)], arms[arm]) != 0 or not path.is_file():
                return f"## In model\n\nnot run: capture of arm {arm} failed ({path.parent})\n"
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
