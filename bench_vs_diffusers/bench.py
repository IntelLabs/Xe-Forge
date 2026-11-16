"""vLLM-Omni (patched pipeline) vs diffusers, end to end, for the four missing-model runs.

The question `op_gain` never answered: what did building a native vLLM-Omni pipeline buy,
against just running the model in diffusers? `op_gain` compares the new pipeline to itself
with custom ops off, which says nothing about diffusers.

Per model and arm this times the whole process at 20 steps and at 1 step, so the per-step cost
falls out without editing anyone's script:

    per_step = (T20 - T1) / 19        fixed = T1 - per_step   (load + encode + VAE)

Load cancels in that subtraction only if both runs are equally warm, which is what the first
attempt got wrong: the weights are on NFS, the first read of each arm was ~10 s slower than
its third, and every T1 ran after every T20, so the slope absorbed the warm-up. Hence one
discarded warm-up run per arm, and the step counts interleaved inside each repeat.

A short generation is mostly load, so the slope is only as good as the noise. CogVideoX and
HunyuanVideo spend seconds per step and measure cleanly. SD 1.5 does not: at 20 steps its
slope came out 0.041 s (diffusers) against 0.084 s (omni) with +-0.095 s of noise -- an error
bar wider than either number. More repeats cannot fix a lever that short, so an image model
takes a third, longer point (`slope_steps`) and the slope is measured over that instead. The
served 20-step time is still reported as the end-to-end number. Per-step cost is a property
of the model, so measuring it over 99 steps rather than 19 changes only the resolution.

`spread` and `per_step_err` are reported for exactly this reason -- read them before a ratio.

Both arms run the same prompt, seed, size, schedule length and precision; the arm scripts are
the ones the patch agents wrote (`generate.sh` -> `gen_<model>.py`, and `diffusers_ref.py`),
so neither side is re-implemented here. Runs alternate A/B so drift hits both arms equally.

Arm A is diffusers, arm B is vLLM-Omni. Both run under the patched interpreter: the patch only
adds a new pipeline module and a registry entry, and `diffusers_ref.py` imports from diffusers
directly, so the patch cannot reach arm A.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(__file__).resolve().parent / "results"
STEPS_FULL, STEPS_ONE, TIMEOUT = 20, 1, 5400

# Each model's capture settings, so the comparison measures what was actually served.
# `env` is what gen_<model>.py needs to match diffusers_ref.py's own fixed choices.
MODELS = [
    # `reps`: an image model's per-step cost is a sliver of its load, so it needs more runs.
    {
        "tag": "vision-pixart-sigma",
        "patch": "patch-1",
        "label": "PixArt-Sigma 1024x1024",
        "reps": 5,
        "slope_steps": 100,
        "env": {},
    },
    {
        "tag": "vision-sd15",
        "patch": "patch-2",
        "label": "SD 1.5 512x512",
        "reps": 5,
        "slope_steps": 100,
        # patch-2 holds the winning pipeline; diffusers_ref.py lives in patch-1.
        "diffusers_from": "patch-1",
        "env": {},
    },
    {
        "tag": "vision-cogvideox-2b",
        "patch": "patch-1",
        "label": "CogVideoX-2B 480x720x49",
        "reps": 3,
        "env": {},
    },
    {
        "tag": "vision-hunyuanvideo",
        "patch": "patch-1",
        "label": "HunyuanVideo 544x960x49",
        "reps": 3,
        # The gate used fp8 without offload to fit 32 GB. diffusers_ref.py is bf16 + model CPU
        # offload, which is also what the capture served, so arm B must match it or the
        # comparison is fp8 against bf16.
        "env": {"GEN_QUANTIZATION": "", "GEN_CPU_OFFLOAD": "1"},
    },
]


def _env(model: dict, steps: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
    env.update(model["env"])
    # Both arms: a <steps>-step schedule run to completion, matching each capture's 20 steps.
    env["GEN_SCHEDULE_STEPS"] = str(steps)
    env["XE_FORGE_CHECK_STEPS"] = str(steps)
    return env


def _time(cmd: list[str], cwd: Path, env: dict, out: Path) -> tuple[float, str]:
    """Wall time of one whole process, and why it failed if it did."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, env={**env, "OUT": str(out)},
                       capture_output=True, text=True, timeout=TIMEOUT)  # fmt: skip
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return float("nan"), (p.stderr or p.stdout).strip()[-400:]
    return dt, ""


def _one(model: dict, arm: str, steps: int, tag: str, keep: Path | None) -> tuple[float, str]:
    """One timed process. `keep` copies its artifacts out instead of deleting them."""
    work = RUNS / model["tag"] / model["patch"]
    dwork = RUNS / model["tag"] / model.get("diffusers_from", model["patch"])
    python = str(work / "python.sh")
    o = OUT / "scratch" / model["tag"] / f"{arm}-{tag}"
    if arm == "A":
        cmd, cwd = [python, str(dwork / "scripts" / "diffusers_ref.py"), str(o), str(steps)], dwork
    else:
        cmd, cwd = ["bash", str(work / "generate.sh")], work
    dt, why = _time(cmd, cwd, {**_env(model, steps), "FRAMEWORK_PYTHON": python}, o)
    print(f"    {arm} {tag:>9s} steps={steps:<3d} {dt:8.1f} s"
          f"{'  FAILED: ' + why if why else ''}", flush=True)  # fmt: skip
    if keep:
        shutil.rmtree(keep / arm, ignore_errors=True)
        shutil.copytree(o, keep / arm, dirs_exist_ok=True)
    shutil.rmtree(o, ignore_errors=True)
    return dt, why


def _measure(model: dict, keep: Path) -> dict:
    """Warm up each arm once, then interleave the step points across repeats."""
    slope = model.get("slope_steps", STEPS_FULL)
    points = sorted({STEPS_FULL, STEPS_ONE, slope}, reverse=True)
    times: dict[int, dict[str, list[float]]] = {n: {"A": [], "B": []} for n in points}
    notes: dict[str, str] = {}
    for arm in ("A", "B"):  # discarded: pulls the weights into the page cache
        _one(model, arm, STEPS_ONE, "warm-up", None)
    for rep in range(1, model["reps"] + 1):
        for steps in points:  # interleaved, so warm-up drift cannot bias the slope
            for arm in ("A", "B"):
                first = rep == 1 and steps == STEPS_FULL
                dt, why = _one(model, arm, steps, f"r{rep}", keep if first else None)
                times[steps][arm].append(dt)
                if why:
                    notes[arm] = why
    return {"times": {str(k): v for k, v in times.items()}, "notes": notes,
            "_t": times}  # fmt: skip


def main() -> int:
    only = sys.argv[1:] or None
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for model in MODELS:
        if only and model["tag"] not in only:
            continue
        print(f"\n=== {model['label']}  ({model['tag']})", flush=True)
        keep = OUT / "outputs" / model["tag"]
        keep.mkdir(parents=True, exist_ok=True)
        m = _measure(model, keep)
        r: dict = {"label": model["label"], "reps": model["reps"],
                   "times": m["times"], "notes": m["notes"]}  # fmt: skip
        for arm in ("A", "B"):
            t20 = [x for x in m["_t"][STEPS_FULL][arm] if x == x]
            t1 = [x for x in m["_t"][STEPS_ONE][arm] if x == x]
            slope_n = model.get("slope_steps", STEPS_FULL)
            ts = [x for x in m["_t"][slope_n][arm] if x == x]
            if t20 and t1 and ts:
                m20, m1, ms = statistics.median(t20), statistics.median(t1), statistics.median(ts)
                per = (ms - m1) / (slope_n - STEPS_ONE)
                r[arm] = {"t20_median": round(m20, 1), "t1_median": round(m1, 1),
                          "per_step": round(per, 3), "fixed": round(m1 - per, 1),
                          "slope_steps": slope_n, "t_slope_median": round(ms, 1),
                          "spread": round(max(t20) - min(t20), 1),
                          # The slope is a difference of two noisy medians: this is how much
                          # of it could be noise. Ignore a ratio whose arms overlap here.
                          "per_step_err": round((max(ts) - min(ts) + max(t1) - min(t1))
                                                / 2 / (slope_n - STEPS_ONE), 3)}  # fmt: skip
        if "A" in r and "B" in r:
            r["speedup_end_to_end"] = round(r["A"]["t20_median"] / r["B"]["t20_median"], 3)
            r["speedup_per_step"] = round(r["A"]["per_step"] / r["B"]["per_step"], 3)
        report[model["tag"]] = r
        (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n" + "=" * 108)
    print("A = diffusers, B = vLLM-Omni. 'x' > 1 means the vLLM-Omni pipeline is faster.")
    print(f"{'model':26s} {'A 20st':>9s} {'B 20st':>9s} {'x':>6s} {'A/step':>9s} {'B/step':>9s} "
          f"{'x/step':>7s} {'+-/step':>8s} {'spreadA':>8s} {'spreadB':>8s}")  # fmt: skip
    for tag, r in report.items():
        if "A" not in r or "B" not in r:
            print(f"{r['label'][:26]:26s}  incomplete: {r.get('notes')}")
            continue
        err = max(r["A"]["per_step_err"], r["B"]["per_step_err"])
        print(f"{r['label'][:26]:26s} {r['A']['t20_median']:7.1f} s {r['B']['t20_median']:7.1f} s "
              f"{r['speedup_end_to_end']:6.3f} {r['A']['per_step']:7.3f} s {r['B']['per_step']:7.3f} s "
              f"{r['speedup_per_step']:7.3f} {err:8.3f} {r['A']['spread']:7.1f}s "
              f"{r['B']['spread']:7.1f}s")  # fmt: skip
    shutil.rmtree(OUT / "scratch", ignore_errors=True)
    print(f"\n-> {OUT / 'report.json'}; first-repeat outputs under {OUT / 'outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
