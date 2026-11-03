"""``capture run`` / ``capture parse``: the instruments and the checks around a launcher an agent
wrote. The launcher owns the framework; these own what is measured and whether it holds.

A launcher runs one profiled batch and writes into ``$RAW_DIR``:
  meta.json   {"framework", "framework_version", "model": {"name", "config"}, "runtime": {...},
               "device": {...}, "repeats": <times the process ran the identical workload,
               warm-up included; unitrace times the whole process>}
  and, for a torch framework, ``trace*.json.gz`` (torch.profiler, record_shapes=True; see
  ``xe_forge.frontend.profile.profile_region``), or, for any other framework, ``ops.json``:
  {"<op key>": {"framework_op", "kernels": ["<regex over unitrace kernel names>", ...],
                "args": [{"name", "shape", "dtype"}], "call_site", "calls", "meta": {...}}}

Exit codes: 0 ok; 1 the launcher or a check failed (the reason is printed); 2 nothing ran on
the device.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

_SIMD = re.compile(r"\[SIMD\d+ \{[^}]*\} \{[^}]*\}\]$")
_SECRET = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(raw: Path) -> dict:
    return {
        str(p.relative_to(raw)): _sha256(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }


def _env() -> dict:
    """The launcher's environment: this one without the API credentials."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}


def _launch(
    launcher: Path, raw: Path, prefix: list[str], log: Path, base: dict | None = None
) -> int:
    env = dict(base or _env(), RAW_DIR=str(raw))
    with open(log, "w") as f:
        return subprocess.call(
            [*prefix, str(launcher)], env=env, stdout=f, stderr=subprocess.STDOUT
        )


def _tail(log: Path, n: int = 25) -> str:
    return "\n".join(log.read_text(errors="replace").splitlines()[-n:])


def _unitrace_kernels(reports: list[Path]) -> dict:
    """{kernel: {"us", "calls"}} from unitrace's device-timing tables, SIMD geometry folded."""
    from xe_forge.core.profiler import UnitraceXPUProfiler

    out: dict = {}
    for path in reports:
        timings = UnitraceXPUProfiler._parse_device_timing(None, path.read_text(errors="replace"))
        for name, t in timings.items():
            k = out.setdefault(_SIMD.sub("", name).strip(), {"us": 0.0, "calls": 0})
            k["us"] += t["time_ns"] / 1e3
            k["calls"] += int(t["calls"])
    return {k: {"us": round(v["us"], 3), "calls": v["calls"]} for k, v in out.items()}


def run_capture(args) -> int:
    launcher, raw = Path(args.launcher).resolve(), Path(args.raw).resolve()
    if not os.access(launcher, os.X_OK):
        print(f"capture run: {launcher} is not an executable file")
        return 1
    if shutil.which(args.unitrace) is None and not Path(args.unitrace).is_file():
        print(f"capture run: no unitrace at {args.unitrace!r} (set --unitrace or UNITRACE_BIN)")
        return 1
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    scratch = raw.parent / f".{raw.name}-unitrace"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "raw").mkdir(parents=True)
    try:
        # Pass 1, plain: the launcher's own trace, timed without unitrace in the way.
        code = _launch(launcher, raw, [], raw.parent / "launcher.log")
        if code != 0:
            print(
                f"capture run: the launcher failed (exit {code}):\n{_tail(raw.parent / 'launcher.log')}"
            )
            return 1
        if not (raw / "meta.json").is_file():
            print("capture run: the launcher wrote no $RAW_DIR/meta.json")
            return 1
        # Pass 2, under unitrace: every kernel the device ran, by its real name, with counters.
        report = scratch / "unitrace.txt"
        cmd = [args.unitrace, "--device-timing", "--opencl", "--metric-sampling",
               "--group", "ComputeBasic", "-o", str(report)]  # fmt: skip
        from xe_forge.core.profiler import UnitraceXPUProfiler

        # The profiler's own setup: its Metrics Library on LD_LIBRARY_PATH, a flat hierarchy.
        env = UnitraceXPUProfiler(unitrace_bin=args.unitrace)._environment()
        env = {k: v for k, v in env.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
        code = _launch(launcher, scratch / "raw", cmd, raw.parent / "unitrace.log", env)
        reports = [p for p in scratch.glob("unitrace*") if ".metrics." not in p.name]
        kernels = _unitrace_kernels(reports)
        if not kernels:
            print(f"capture run: unitrace saw no device kernel (exit {code}):\n"
                  f"{_tail(raw.parent / 'unitrace.log')}")  # fmt: skip
            return 2
        (raw / "unitrace_kernels.json").write_text(json.dumps(kernels, indent=1) + "\n")
        from xe_forge.core.compute_metrics import summarize_compute_metrics

        records = []
        for path in sorted(scratch.glob("unitrace*.metrics.*")):
            with open(path, errors="replace") as f:
                records += summarize_compute_metrics(f)["records"]
        if records:
            (raw / "counters.json").write_text(json.dumps({"records": records}, indent=1) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    run = json.loads((raw / "meta.json").read_text())
    run.update(
        launcher=str(launcher),
        launcher_sha256=_sha256(launcher),
        env={k: v for k, v in sorted(_env().items()) if not any(w in k for w in _SECRET)
             and k.startswith(("VLLM_", "SGLANG_", "OV_", "ONEAPI_", "SYCL_", "ZE_", "IGC_"))},
    )  # fmt: skip
    (raw / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    (raw / "MANIFEST.json").write_text(json.dumps(_manifest(raw), indent=1) + "\n")
    total = sum(k["us"] for k in kernels.values())
    print(f"capture run: {len(kernels)} kernels, {total / 1e3:.1f} ms device time (unitrace), "
          f"counters for {len(records)} records -> {raw}")  # fmt: skip
    return 0


def _ops_summary(ops_map: dict, kernels: dict):
    """An agent's op -> kernel map, timed by unitrace; problems where it does not hold."""
    from xe_forge.frontend.ir import CapturedOp, TensorArg
    from xe_forge.frontend.torch_trace import TraceSummary

    problems, owner, ops = [], {}, []
    for key, e in ops_map.items():
        pats = e.get("kernels") or []
        hits = set()
        for pat in pats:
            try:
                found = {k for k in kernels if re.search(pat, k)}
            except re.error as exc:
                problems.append(f"{key}: kernel pattern {pat!r} is not a regex ({exc})")
                continue
            if not found:
                problems.append(f"{key}: kernel pattern {pat!r} matches no kernel unitrace saw")
            hits |= found
        for k in hits:
            if k in owner:
                problems.append(f"kernel {k!r} claimed by both {owner[k]!r} and {key!r}")
            owner[k] = key
        arg_list = e.get("args") or []
        args = [TensorArg(list(a.get("shape") or []), a.get("dtype")) for a in arg_list]
        meta = dict(e.get("meta") or {}, arg_names=[a.get("name") for a in arg_list])
        op = CapturedOp(id="", framework_op=e.get("framework_op") or key, args=args,
                        call_site=e.get("call_site") or "", framework_meta=meta)  # fmt: skip
        op.calls = int(e.get("calls") or max((kernels[k]["calls"] for k in hits), default=0))
        op.kernel_symbols = {k: kernels[k]["us"] for k in sorted(hits)}
        op.device_us = round(sum(op.kernel_symbols.values()), 3)
        ops.append(op)
    ops.sort(key=lambda o: -o.device_us)
    for i, op in enumerate(ops):
        op.id = f"op-{i:04d}"
    total = sum(k["us"] for k in kernels.values())
    attributed = sum(kernels[k]["us"] for k in owner)
    summary = TraceSummary(ops=ops, total_us=total, attributed_us=attributed,
                           kernels={k: v["us"] for k, v in kernels.items()},
                           launches=sum(v["calls"] for v in kernels.values()))  # fmt: skip
    return summary, problems


def parse_capture(args) -> int:
    from xe_forge.frontend.capture import build

    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    want = raw / "MANIFEST.json"
    if not want.is_file():
        print(f"capture parse: {raw} has no MANIFEST.json; write it with `capture run`")
        return 1
    changed = sorted(f for f, h in json.loads(want.read_text()).items()
                     if not (raw / f).is_file() or _sha256(raw / f) != h)  # fmt: skip
    if changed:
        print(f"capture parse: raw/ changed since `capture run`: {changed[:5]}")
        return 1
    kernels = json.loads((raw / "unitrace_kernels.json").read_text())
    # unitrace times the whole process; a launcher that ran the identical workload N times
    # (its warm-up included) says so, and one run's share is 1/N of it.
    repeats = int(json.loads((raw / "run.json").read_text()).get("repeats") or 1)
    if repeats > 1:
        kernels = {k: {"us": v["us"] / repeats, "calls": round(v["calls"] / repeats)}
                   for k, v in kernels.items()}  # fmt: skip
    truth = sum(k["us"] for k in kernels.values())
    names = json.loads(Path(args.names).read_text()) if args.names else None
    summary, problems = None, []
    if (raw / "ops.json").is_file():
        summary, problems = _ops_summary(json.loads((raw / "ops.json").read_text()), kernels)
    elif not list(raw.glob("trace*.json*")):
        print("capture parse: raw/ has neither a torch trace nor ops.json")
        return 1
    capture = build(raw, out, names, summary=summary)
    r = capture.run
    # The trace's kernels must be ones the device ran: share of trace time unitrace knows.
    traced = {}
    for op in capture.ops:
        for k, us in op.kernel_symbols.items():
            traced[k] = traced.get(k, 0.0) + us
    seen = sum(
        us for k, us in traced.items() if k in kernels or any(u.startswith(k) for u in kernels)
    )
    known_pct = 100 * seen / sum(traced.values()) if traced else 0.0
    print(f"capture parse: {len(capture.ops)} ops, {len(capture.workloads)} workloads, "
          f"{r['total_device_us'] / 1e3:.1f} ms device time, {r['attributed_pct']}% attributed; "
          f"unitrace saw {truth / 1e3:.1f} ms; {known_pct:.1f}% of op time is in kernels unitrace "
          f"names -> {out}")  # fmt: skip
    if not capture.ops:
        problems.append("no ops")
    if known_pct < 90:
        problems.append(f"only {known_pct:.1f}% of op time is in kernels unitrace saw")
    for p in problems:
        print(f"capture parse: PROBLEM {p}")
    return 1 if problems else 0


def run(args) -> int:
    if args.capture_command == "run":
        return run_capture(args)
    return parse_capture(args)
