"""
Run any Python workload under torch.profiler and export the Chrome trace.

    python -m xe_forge.orbit.profiling.wrap --out trace.json -- python -m my.workload args...
    python -m xe_forge.orbit.profiling.wrap --out trace.json -- python script.py args...

This is what makes `xe-orbit trace --wrap -- <command>` a point-and-start operation:
the workload needs no profiler code of its own — the wrapper runs it in-process (the
profiler must live in the process that launches the kernels) and writes the trace
where the trace stage expects it.

Scope, stated honestly: in-process wrapping covers single-process torch workloads —
`orbit_mini`, training scripts, plain inference. A framework that moves GPU work into
its own subprocess (vLLM's EngineCore, SGLang's scheduler) is NOT reachable this way;
those are profiled through the framework's own hook (vLLM: `profiler_config` /
`/start_profile`), which is per-framework knowledge the adapter carries. The wrapper
refuses gracefully rather than emitting an empty trace as though it measured
something: a trace with zero device events is reported as such.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m xe_forge.orbit.profiling.wrap",
        description="Run a Python workload under torch.profiler; export a Chrome trace.",
    )
    parser.add_argument("--out", required=True, help="Chrome trace output path")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the workload, after `--`: python [-m module | script.py] [args...]",
    )
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    # The leading interpreter token is accepted and ignored: the workload runs in
    # THIS process (the profiler must), so `python -m mod` and `-m mod` mean the
    # same thing here.
    if command and (command[0].endswith("python") or command[0].endswith("python3")):
        command = command[1:]
    if not command:
        parser.error("no workload given after --")

    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        activities.append(ProfilerActivity.XPU)
    elif torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    if command[0] == "-m":
        if len(command) < 2:
            parser.error("-m requires a module name")
        target = ("module", command[1])
        workload_argv = command[1:]
    else:
        target = ("path", command[0])
        workload_argv = command

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv
    sys.argv = list(workload_argv)
    try:
        with profile(activities=activities, record_shapes=True) as prof:
            try:
                if target[0] == "module":
                    runpy.run_module(target[1], run_name="__main__")
                else:
                    runpy.run_path(target[1], run_name="__main__")
            except SystemExit as exc:
                # A workload calling sys.exit(0) has finished, not failed.
                if exc.code not in (None, 0):
                    raise
    finally:
        sys.argv = old_argv

    prof.export_chrome_trace(str(out))

    device_events = sum(
        1
        for e in prof.events()
        if e.device_type is not None and "cpu" not in str(e.device_type).lower()
    )
    print(f"wrote {out} ({device_events} device-side events)")
    if device_events == 0:
        print(
            "note: zero device events. Either the workload ran on CPU, or its GPU "
            "work happens in a subprocess this wrapper cannot reach (vLLM/SGLang "
            "engines) — use the framework's own profiler hook for those.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
