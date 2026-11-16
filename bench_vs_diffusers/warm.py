"""Warm serving: load once, generate N times. vLLM-Omni's patched pipeline vs diffusers.

`bench.py` times whole processes, so every number it reports carries the model load. That is
the one-shot case, and it is the wrong case for a server: a server loads once and then serves.
The warm cost had to be *derived* there, from the slope `(T100 - T1) / 99`, which cannot see
allocator state, resident weights, or HunyuanVideo's offload swapping on every request.

This measures it instead. Neither arm is re-implemented: each arm's own script is copied and
its single generation statement wrapped in a timed loop, so the code that runs is the code the
patch agents wrote. `--diff` prints the transformation for a given model so the change can be
checked by eye.

    arm A  runs/<tag>/patch-*/scripts/diffusers_ref.py   the `pipe(...)` call
    arm B  runs/<tag>/patch-*/scripts/gen_<model>.py     the `omni.generate(...)` call

Reports generation 1 (cold, straight after load) against the median of generations 2..N (warm).
The seeded generator is re-seeded each iteration where one exists, so every generation is the
same work on the same noise.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[0] / "runs"
OUT = HERE / "results_warm"
STEPS, GENS, TIMEOUT = 20, 4, 7200

# Same four runs and the same fairness fix as bench.py: HunyuanVideo's arm B must be bf16 with
# offload, which is what its capture served and what its diffusers reference uses.
MODELS = [
    {
        "tag": "vision-pixart-sigma",
        "patch": "patch-1",
        "label": "PixArt-Sigma 1024x1024",
        "env": {},
    },
    {
        "tag": "vision-sd15",
        "patch": "patch-2",
        "diffusers_from": "patch-1",
        "label": "SD 1.5 512x512",
        "env": {},
    },
    {
        "tag": "vision-cogvideox-2b",
        "patch": "patch-1",
        "label": "CogVideoX-2B 480x720x49",
        "env": {},
    },
    {
        "tag": "vision-hunyuanvideo",
        "patch": "patch-1",
        "label": "HunyuanVideo 544x960x49",
        "env": {"GEN_QUANTIZATION": "", "GEN_CPU_OFFLOAD": "1"},
    },
]


def _target(tree: ast.AST, arm: str) -> ast.stmt:
    """The one statement that generates. Arm A calls `pipe(...)`, arm B `omni.generate(...)`."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.Expr)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if arm == "A" and isinstance(f, ast.Name) and f.id == "pipe":
                hits.append(node)
            elif (arm == "B" and isinstance(f, ast.Attribute) and f.attr == "generate"
                  and isinstance(f.value, ast.Name)):  # fmt: skip
                hits.append(node)
            break
    if len(hits) != 1:
        raise SystemExit(f"arm {arm}: found {len(hits)} generation statements, need exactly 1")
    return hits[0]


def _loop(src: Path, arm: str) -> str:
    """`src` with its generation statement wrapped in a timed loop. Everything else verbatim."""
    text = src.read_text()
    lines = text.splitlines()
    node = _target(ast.parse(text), arm)
    a, b = node.lineno - 1, node.end_lineno  # the statement, 0-based half-open
    stmt = lines[a:b]
    pad = " " * (len(stmt[0]) - len(stmt[0].lstrip()))
    body = [
        f"{pad}import time as _warm_time, json as _warm_json, os as _warm_os",
        f"{pad}_warm_t = []",
        f"{pad}for _warm_i in range(int(_warm_os.environ.get('WARM_GENS', '4'))):",
        f"{pad}    try:",
        f"{pad}        gen.manual_seed(g.SEED)  # same noise every generation",
        f"{pad}    except Exception:",
        f"{pad}        pass",
        f"{pad}    _warm_0 = _warm_time.perf_counter()",
        *[f"    {ln}" for ln in stmt],  # the original statement, untouched
        f"{pad}    _warm_t.append(_warm_time.perf_counter() - _warm_0)",
        f"{pad}    print(f'[warm] gen {{_warm_i + 1}}: {{_warm_t[-1]:.3f}} s', flush=True)",
        f"{pad}_warm_json.dump(_warm_t, open(_warm_os.environ['WARM_JSON'], 'w'))",
    ]
    return "\n".join([*lines[:a], *body, *lines[b:]]) + "\n"


def _run(model: dict, arm: str, work: Path) -> tuple[list[float], str]:
    run = RUNS / model["tag"]
    bdir = run / model["patch"]
    adir = run / model.get("diffusers_from", model["patch"])
    python = str(bdir / "python.sh")
    if arm == "A":
        src = adir / "scripts" / "diffusers_ref.py"
        looped = work / src.name
        looped.write_text(_loop(src, "A"))
        cmd, cwd = [python, str(looped), str(work / "out"), str(STEPS)], adir
    else:
        src = next(p for p in (bdir / "scripts").glob("gen_*.py"))
        looped = work / src.name
        looped.write_text(_loop(src, "B"))
        # generate.sh runs scripts/gen_*.py; point it at the looped copy by name on PYTHONPATH.
        cmd, cwd = [python, str(looped), str(work / "out")], bdir
    jsonf = work / f"warm-{arm}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
    env.update(model["env"], WARM_GENS=str(GENS), WARM_JSON=str(jsonf),
               FRAMEWORK_PYTHON=python, OUT=str(work / "out"),
               GEN_SCHEDULE_STEPS=str(STEPS), XE_FORGE_CHECK_STEPS=str(STEPS),
               PYTHONPATH=os.pathsep.join(x for x in (str(src.parent), env.get("PYTHONPATH")) if x))  # fmt: skip
    (work / "out").mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    if not jsonf.is_file():
        return [], (p.stderr or p.stdout).strip()[-500:]
    return json.loads(jsonf.read_text()), ""


def main() -> int:
    if "--diff" in sys.argv:
        tag = sys.argv[sys.argv.index("--diff") + 1]
        model = next(m for m in MODELS if m["tag"] == tag)
        run = RUNS / tag
        for arm, d, pat in (("A", model.get("diffusers_from", model["patch"]), "diffusers_ref.py"),
                            ("B", model["patch"], "gen_*.py")):  # fmt: skip
            src = next((run / d / "scripts").glob(pat))
            new = _loop(src, arm)
            print(f"\n----- arm {arm}: {src}")
            import difflib

            print("\n".join(difflib.unified_diff(src.read_text().splitlines(),
                                                 new.splitlines(), lineterm="", n=1)))  # fmt: skip
        return 0
    only = [a for a in sys.argv[1:] if not a.startswith("-")] or None
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for model in MODELS:
        if only and model["tag"] not in only:
            continue
        print(f"\n=== {model['label']}  ({model['tag']}), {GENS} generations of {STEPS} steps",
              flush=True)  # fmt: skip
        r: dict = {"label": model["label"], "gens": GENS, "steps": STEPS}
        for arm in ("A", "B"):
            work = OUT / "scratch" / model["tag"] / arm
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            times, why = _run(model, arm, work)
            print(f"  arm {arm} ({'diffusers' if arm == 'A' else 'omni'}): "
                  f"{[round(x, 2) for x in times] or 'FAILED: ' + why} "
                  f"(process {time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
            if len(times) >= 2:
                r[arm] = {"cold": round(times[0], 3),
                          "warm_median": round(statistics.median(times[1:]), 3),
                          "warm_all": [round(x, 3) for x in times[1:]],
                          "warm_spread": round(max(times[1:]) - min(times[1:]), 3)}  # fmt: skip
            else:
                r[arm] = {"error": why}
            shutil.rmtree(work / "out", ignore_errors=True)
        if "warm_median" in r.get("A", {}) and "warm_median" in r.get("B", {}):
            r["warm_speedup"] = round(r["A"]["warm_median"] / r["B"]["warm_median"], 3)
            r["cold_speedup"] = round(r["A"]["cold"] / r["B"]["cold"], 3)
        report[model["tag"]] = r
        (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n" + "=" * 100)
    print(
        "Generation only, model already loaded. A = diffusers, B = vLLM-Omni. x > 1: omni faster."
    )
    print(f"{'model':26s} {'A cold':>9s} {'A warm':>9s} {'B cold':>9s} {'B warm':>9s} "
          f"{'x warm':>7s} {'x cold':>7s} {'spreadA':>8s} {'spreadB':>8s}")  # fmt: skip
    for tag, r in report.items():
        if "warm_speedup" not in r:
            print(f"{r['label'][:26]:26s}  incomplete: "
                  f"{r.get('A', {}).get('error') or r.get('B', {}).get('error')}")  # fmt: skip
            continue
        print(f"{r['label'][:26]:26s} {r['A']['cold']:7.2f} s {r['A']['warm_median']:7.2f} s "
              f"{r['B']['cold']:7.2f} s {r['B']['warm_median']:7.2f} s {r['warm_speedup']:7.3f} "
              f"{r['cold_speedup']:7.3f} {r['A']['warm_spread']:7.2f}s {r['B']['warm_spread']:7.2f}s")  # fmt: skip
    shutil.rmtree(OUT / "scratch", ignore_errors=True)
    print(f"\n-> {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
