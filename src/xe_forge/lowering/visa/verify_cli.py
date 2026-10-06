"""The session's two commands: the oracle and the ground-truth probe.

    python -m xe_forge.lowering.visa.verify_cli verify <state.json> <kernel.visaasm>
    python -m xe_forge.lowering.visa.verify_cli probe  <state.json> [num_warps=N] [threads_per_warp=16|32] [slm_bytes=B]

``verify`` (the oracle) says whether an attempt computes what the Triton kernel
computes. ``probe`` (ground truth) prints the raw payload facts the compiler and
runtime impose for a launch configuration. Every decision -- algorithm, launch,
SIMD width, shared memory, how to read the payload -- is the model's.

It finalizes the attempt, runs it against the Triton reference on every
verification case, records it (attempt directory, trial tree, metrics), and prints
the structured feedback the model works from, ending with ``VERDICT:`` and
``ATTEMPTS_LEFT:`` lines. Exit status: 0 correct, 1 not correct, 2 budget exhausted
or refused, 3 infrastructure failure (not the model's fault).

The state file and everything it points to live outside the session's workspace;
the session sees only this command's output, which is checked by the artifact
guard before it is printed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from xe_forge.lowering.context import LeakDetected


def _attempt_dirs(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "attempts").glob("a[0-9][0-9][0-9]"))


def summarize_attempts(run_dir: Path) -> list[dict]:
    """The recorded attempts of a run, in order (used by the pipeline after the session)."""
    out = []
    for d in _attempt_dirs(run_dir):
        ev_path = d / "evaluation.json"
        if ev_path.exists():
            ev = json.loads(ev_path.read_text())
            ev["dir"] = str(d)
            out.append(ev)
    return out


def _build_for(st: dict, abi_build, executor, config):
    """The stub build for a launch configuration: the default one, or one built on demand (cached)."""
    from xe_forge.lowering.pipeline import abi_build_from_json, abi_build_to_json, write_json
    from xe_forge.lowering.visa.launch import LaunchConfig

    if config.key() == LaunchConfig.from_json(st["default_config"]).key():
        return abi_build, st["captured"]["stub_py"]
    cache = Path(st["run_dir"]) / "work" / "abi_cache" / f"{config.key()}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return abi_build_from_json(d["build"]), d["stub_py"]
    build, stub_py = executor.build_abi(st["captured"], config)
    write_json(cache, {"build": abi_build_to_json(build), "stub_py": stub_py})
    return build, stub_py


def probe_main(argv: list[str] | None = None) -> int:
    """``abi-probe num_warps=N threads_per_warp=16|32 slm_bytes=B``: raw payload facts."""
    argv = sys.argv[1:] if argv is None else argv
    from xe_forge.lowering.pipeline import load_state
    from xe_forge.lowering.triton_analyzer import LaunchRecord
    from xe_forge.lowering.visa.compiler import probe_report
    from xe_forge.lowering.visa.launch import LaunchConfig, LaunchError, parse_launch

    st, abi_build, _, _, executor = load_state(Path(argv[0]))
    try:
        config = parse_launch("// @launch " + " ".join(argv[1:]), LaunchConfig.from_json(st["default_config"]))
    except LaunchError as e:
        print(f"error: {e}")
        return 1
    build, _ = _build_for(st, abi_build, executor, config)
    print(probe_report(build, LaunchRecord.from_json(st["captured"]["launch"]), config).rstrip())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: visa-verify <kernel.visaasm>")
        return 2
    state_path, visa_path = Path(argv[0]), Path(argv[1])
    if not visa_path.exists():
        print(f"VERDICT: NO_FILE\nno such file: {visa_path.name}")
        return 2

    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.lowering.metrics import append_jsonl
    from xe_forge.lowering.pipeline import load_state, write_json
    from xe_forge.lowering.visa.executor import DeviceUnhealthy
    from xe_forge.lowering.visa.compiler import FinalizerUnavailable

    st, abi_build, cases, guard, executor = load_state(state_path)
    run_dir = Path(st["run_dir"])
    done = _attempt_dirs(run_dir)
    budget = int(st["max_attempts"])
    if len(done) >= budget:
        print(f"VERDICT: BUDGET_EXHAUSTED\nAll {budget} attempts are used; stop.\nATTEMPTS_LEFT: 0")
        return 2
    from xe_forge.lowering.visa.compiler import input_mismatch
    from xe_forge.lowering.visa.launch import LaunchConfig, LaunchError, parse_launch

    visa = visa_path.read_text()
    try:
        config = parse_launch(visa, LaunchConfig.from_json(st["default_config"]))
        build, stub_py = _build_for(st, abi_build, executor, config)
    except LaunchError as e:
        print(f"VERDICT: LAUNCH_ERROR\n{e}\n(not counted as an attempt)\nATTEMPTS_LEFT: {budget - len(done)}")
        return 1
    mismatch = input_mismatch(visa, build)
    if mismatch:
        print(f"VERDICT: ABI_MISMATCH\n{mismatch}\n(not counted as an attempt)\nATTEMPTS_LEFT: {budget - len(done)}")
        return 1
    index = len(done)
    keep = run_dir / "attempts" / f"a{index:03d}"
    keep.mkdir(parents=True)
    (keep / "kernel.visaasm").write_text(visa)
    t0 = time.time()
    try:
        ev = executor.evaluate(
            visa, build, st["captured"], cases, config=config, stub_py=stub_py,
            module_path=st["module_path"], spec_path=st["spec_path"], tolerance=st["tolerance"],
            transcendental=st["transcendental"], measure_perf=bool(st.get("optimize")), keep_dir=keep,
            forbidden_lines=guard._lines,
        )
    except (FinalizerUnavailable, DeviceUnhealthy) as e:
        (keep / "infra_error.txt").write_text(str(e))
        print(f"VERDICT: INFRA\nThe verification infrastructure failed (not your kernel): {e}\nStop.")
        return 3
    write_json(keep / "evaluation.json", {**ev.to_dict(), "elapsed_s": round(time.time() - t0, 2)})

    trials = TrialManager(run_dir / "trials")
    tree = st["trial_tree"]
    tid = trials.save_trial(tree, keep / "kernel.visaasm", strategy=f"visa:a{index}")
    perf = ev.performance
    trials.record_result(
        tree, tid,
        validation="pass" if ev.compile.success else "fail",
        correctness="pass" if ev.category.is_correct else "fail",
        speedup=perf.speedup if perf else None,
        baseline_us=perf.baseline_us if perf else None,
        triton_us=perf.candidate_us if perf else None,
        verdict=None if perf or not ev.category.is_correct else "correct-unmeasured",
    )
    c = ev.compile
    append_jsonl(run_dir / "metrics.jsonl", {
        "run_id": st["run_id"], "index": index, "category": ev.category.value, "t": time.time(),
        "compile_ok": c.success, "runtime_ok": ev.runtime.success if ev.runtime else None,
        "correct": ev.correctness.success if ev.correctness else None,
        "error": (c.error or (ev.runtime.error if ev.runtime else "") or "").splitlines()[0][:200] if (
            c.error or (ev.runtime and ev.runtime.error)) else "",
        "max_abs_error": ev.correctness.max_abs_error if ev.correctness else None,
        "failing_elements": ev.correctness.failing_elements if ev.correctness else None,
        "grf": c.grf_count, "spills": c.spill_size, "binary_size": c.binary_size,
        "speedup": round(perf.speedup, 4) if perf else None,
    })

    feedback = ev.feedback_yaml()
    try:
        guard.check(feedback)
    except LeakDetected as e:
        (keep / "leak.txt").write_text(str(e))
        print("VERDICT: INFRA\nFeedback withheld by the artifact guard. Stop.")
        return 3
    left = budget - index - 1
    print(feedback.rstrip())
    if ev.category.is_correct:
        n = ev.correctness.shapes_checked if ev.correctness else 0
        print(f"\nThe kernel is correct on all {n} verification cases.")
    print(f"VERDICT: {ev.category.value.upper()}")
    print(f"ATTEMPTS_LEFT: {left}")
    return 0 if ev.category.is_correct else 1


def cli() -> int:
    """``verify <state> <kernel.visaasm>`` | ``probe <state> [num_warps=..] [threads_per_warp=..] [slm_bytes=..]``"""
    cmd, *rest = sys.argv[1:] or [""]
    if cmd == "verify":
        return main(rest)
    if cmd == "probe":
        return probe_main(rest)
    print(cli.__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(cli())
