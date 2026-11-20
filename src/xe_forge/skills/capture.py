"""``capture run`` / ``capture parse``: the instruments and the checks around a launcher an agent
wrote. The launcher owns the framework; these own what is measured and whether it holds.

A launcher runs one profiled batch and writes into ``$RAW_DIR``:
  meta.json   {"framework", "framework_version", "model": {"name", "config"}, "runtime": {...},
               "device": {...}, "repeats": <times the process ran the identical workload,
               warm-up included; unitrace times the whole process, and first-use work
               (model load, a prepack) stays before those runs>}
  and, for a torch framework, ``trace*.json.gz`` (torch.profiler, record_shapes=True; see
  ``xe_forge.frontend.profile.profile_region``), or, for any other framework, ``ops.json``:
  {"<op key>": {"framework_op", "kernels": ["<regex over unitrace kernel names>", ...],
                "args": [{"name", "shape", "dtype"}], "call_site", "calls", "meta": {...}}}

Exit codes: 0 ok; 1 the launcher or a check failed (the reason is printed); 2 nothing ran on
the device.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from xe_forge.frontend import canonical, ir, naming
from xe_forge.frontend.dtypes import STORAGE_BYTES, compute_dtype, observed, parse_quantization
from xe_forge.frontend.dtypes import canonical as canonical_dtype
from xe_forge.frontend.torch_trace import parse_traces

_SIMD = re.compile(r"\[SIMD\d+ \{[^}]*\} \{[^}]*\}\]$")


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
    """The launcher's environment: this one without the API credentials.

    XE_FORGE_TRACE_STEPS goes with them. It is set for the unitrace pass only, and an
    exported one reaching the timed pass would shorten the measurement itself.
    """
    drop = ("ANTHROPIC_", "OPENAI_", "XE_FORGE_TRACE_STEPS")
    return {k: v for k, v in os.environ.items() if not k.startswith(drop)}


def _launch(
    launcher: Path,
    raw: Path,
    prefix: list[str],
    log: Path,
    base: dict | None = None,
    cwd: Path | None = None,
) -> int:
    env = dict(base or _env(), RAW_DIR=str(raw))
    with open(log, "w") as f:
        return subprocess.call(
            [*prefix, str(launcher)], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=cwd
        )


def _same_work(a: list, b: list) -> float:
    """Share of ``a``'s kernel time that ``b`` ran too: 1.0 when both did the same work.

    The launcher says it ran the workload N times and one run is the last 1/N of the
    timeline, so that claim has to be checked. Counting launches checks it badly: a single
    stray launch -- a lazy allocation, a one-off init -- leaves the total indivisible while
    both runs did the same work, and at a few hundred thousand launches that decides the
    verdict by chance. So compare the work itself: time per kernel name in one window against
    the other.
    """

    def by_name(xs: list) -> dict:
        out: dict = {}
        for name, us in xs:
            out[name] = out.get(name, 0.0) + us
        return out

    x, y = by_name(a), by_name(b)
    total = sum(x.values())
    if not total:
        return 0.0
    return min(1.0, sum(min(us, y.get(k, 0.0)) for k, us in x.items()) / total)


def _timeline(scratch: Path) -> list[list]:
    """[[kernel, us], ...] of every launch, in submission order, from --chrome-kernel-logging.

    Each process's launches are ordered by unitrace's launch id, which counts submissions: their
    device start times can be skewed enough to swap neighbours, and
    the order would then differ from run to run. Without ids, by start time. Processes follow
    one another by their first launch."""
    procs = []
    for path in scratch.glob("*.json"):
        try:
            data = json.loads(path.read_text(errors="replace"), strict=False)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        ev = [e for e in data.get("traceEvents") or []
              if e.get("ph") == "X" and e.get("cat") == "gpu_op"]  # fmt: skip
        if not ev:
            continue
        if all(str((e.get("args") or {}).get("id", "")).isdigit() for e in ev):
            ev.sort(key=lambda e: int(e["args"]["id"]))
        else:
            ev.sort(key=lambda e: float(e["ts"]))
        procs.append(ev)
    procs.sort(key=lambda ev: min(float(e["ts"]) for e in ev))
    return [[_SIMD.sub("", e["name"]).strip(), round(float(e["dur"]), 3)]
            for ev in procs for e in ev]  # fmt: skip


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


_SAMPLING_SHM = Path("/dev/shm/uctrlunitrace_metric_sampling_control")  # unitrace prefixes /uctrl


# torch and unitrace name the same device copy differently, and a workload that streams
# weights is mostly copies: torch calls them "Memcpy H2D (...)" while unitrace calls the
# very same transfers "zeCommandListAppendMemoryCopy(H2D)[bytes]", so unmapped they count
# against the coverage check.
#
# Mapped rather than excluded. A copy is real device time: dropping it from the check would
# pass the capture and then hide a fifth of the run from the ranking, which is the opposite
# of what this file is for.
_MEMCPY = re.compile(r"^Mem(cpy|set)\s*([A-Z0-9]+)?", re.I)


def _is_seen(name: str, kernels: dict) -> bool:
    """Whether unitrace saw the thing the trace calls ``name``."""
    if name in kernels or any(u.startswith(name) for u in kernels):
        return True
    m = _MEMCPY.match(name)
    if not m:
        return False
    # "Memcpy H2D (Host -> Device)" -> unitrace's "...AppendMemoryCopy(H2D)[bytes]";
    # a direction-less "Memset" matches any of unitrace's fill entries.
    kind = "Copy" if m.group(1).lower() == "cpy" else "Fill"
    direction = (m.group(2) or "").upper()
    return any(
        f"Memory{kind}" in u and (not direction or f"({direction})" in u.upper()) for u in kernels
    )


def _sampling_free() -> bool:
    """Whether unitrace can open its metric-sampling session here.

    The session has one fixed name for every user: a killed unitrace leaves it behind and every
    later one fails ("Failed to create shared memory ... File exists"). Ours is removed when no
    unitrace of ours runs; another user's cannot be, and the counters are then not measured.
    """
    if not _SAMPLING_SHM.exists():
        return True
    alive = subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", "unitrace"],
                           capture_output=True).returncode == 0  # fmt: skip
    if not alive:
        try:
            _SAMPLING_SHM.unlink()
            print(f"capture run: removed {_SAMPLING_SHM}, left by a unitrace that no longer runs")
            return True
        except OSError:
            pass
    print(
        f"capture run: {_SAMPLING_SHM} is held by another unitrace session on this node: "
        "no counters this time (bounds take them as unknown)"
    )
    return False


def _has_group(unitrace: str, group: str, env: dict) -> bool:
    """Whether this device exposes a metric group.

    Asking for one it does not have is not a missing counter: unitrace exits 255 before it
    runs the launcher, so nothing is captured at all and the device-timing pass -- which the
    whole capture rests on -- is lost with it. So ask the device first.

    A device that lists EuStallSampling and nothing else is usually not a device without
    counters: reading them needs metrics discovery (libigdmd, which the GPU stacks ship) and
    the metrics library (libigdml, which none of them ships). Build intel/metrics-library and
    put its lib64 on LD_LIBRARY_PATH -- it is user-space code and needs no root; the steps are
    in docs/PROFILING.md -- and the device then lists its full set of groups,
    ComputeBasic among them.
    """
    try:
        out = subprocess.run([unitrace, "--metric-list"], capture_output=True, text=True,
                             env=env, timeout=180)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"capture run: cannot list this device's metric groups ({exc}): no counters "
              "this time (bounds take them as unknown)")  # fmt: skip
        return False
    if group in out.stdout or group in out.stderr:
        return True
    print(f"capture run: this device lists no {group} metric group: no counters this time "
          "(bounds take them as unknown). If it lists EuStallSampling only, the metrics "
          "library (libigdml) is probably not on LD_LIBRARY_PATH: build "
          "intel/metrics-library (no root needed) and add its lib64, as "
          "docs/PROFILING.md describes")  # fmt: skip
    return False


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
        # Pass 2, under unitrace: every kernel the device ran, by its real name, with
        # counters. This pass may do less work than the one above -- see --trace-steps.
        report = scratch / "unitrace.txt"
        cmd = [args.unitrace, "--device-timing", "--chrome-kernel-logging", "--opencl",
               "-o", str(report)]  # fmt: skip
        from xe_forge.core.profiler import UnitraceXPUProfiler

        # The profiler's own setup: its Metrics Library on LD_LIBRARY_PATH, a flat hierarchy.
        env = UnitraceXPUProfiler(unitrace_bin=args.unitrace)._environment()
        env = {k: v for k, v in env.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
        sampling = _sampling_free() and _has_group(args.unitrace, "ComputeBasic", env)
        if sampling:
            # The interval decides whether the counters are usable at all, in both
            # directions. Metric sampling is time-based: a kernel shorter than one tick is
            # never sampled, and analyze drops a kernel whose counters cover under half its
            # measured time -- so too coarse gives nothing usable. Too fine gives nothing at
            # all: unitrace stops partway and writes a truncated metrics file that parses to
            # zero records, with no error and exit 0.
            #
            # 25 us is the default. Another device, or a much shorter or longer run, may
            # need a different interval -- `--sampling-interval-us`, env
            # XE_FORGE_SAMPLING_INTERVAL_US. Sampling barely moves device time, so this
            # shares the pass with --device-timing.
            cmd[-2:-2] = ["--metric-sampling", "--group", "ComputeBasic",
                          "--sampling-interval", str(args.sampling_interval)]  # fmt: skip
        if args.trace_steps:
            # The launcher's contract: run this many steps in place of the workload's own
            # count, changing nothing else. Only this pass is cut; the timed pass above keeps
            # the whole workload, its trace and its output.
            env["XE_FORGE_TRACE_STEPS"] = str(args.trace_steps)
        else:
            # 0 is "measure the whole workload under unitrace too", and it has to hold
            # against an exported variable: this env is built from os.environ, and the flag
            # is the decision while the variable is only where its default came from.
            env.pop("XE_FORGE_TRACE_STEPS", None)
        code = _launch(launcher, scratch / "raw", cmd, raw.parent / "unitrace.log", env, scratch)
        # Whether the launcher read it. A launcher written before the variable existed runs
        # the full workload under unitrace, which is what it did before and still correct --
        # but then nothing downstream may say the pass was short, and the launch count that
        # loses the pass is still there. Its own meta.json is the evidence, while it exists.
        short = False
        if args.trace_steps:
            ran = scratch / "raw" / "meta.json"
            if ran.is_file():
                before = json.loads((raw / "meta.json").read_text()).get("runtime")
                short = json.loads(ran.read_text()).get("runtime") != before
            if not short:
                print(f"capture run: the launcher ran the full workload under unitrace; it "
                      f"does not read XE_FORGE_TRACE_STEPS (asked for {args.trace_steps}). "
                      f"That is the behaviour it had before the variable existed, and the "
                      f"capture is sound; a launcher that reads it makes this pass far "
                      f"cheaper and cannot run the device out of timing events.")  # fmt: skip
        reports = [p for p in scratch.glob("unitrace*") if ".metrics." not in p.name]
        kernels = _unitrace_kernels(reports)
        if not kernels:
            print(f"capture run: unitrace saw no device kernel (exit {code}):\n"
                  f"{_tail(raw.parent / 'unitrace.log')}")  # fmt: skip
            return 2
        (raw / "unitrace_kernels.json").write_text(json.dumps(kernels, indent=1) + "\n")
        # The measured run's launches: the last 1/repeats of the process's timeline.
        launches = _timeline(scratch)
        repeats = int(json.loads((raw / "meta.json").read_text()).get("repeats") or 1)
        if launches:
            # Whether the launcher's "repeats" holds is decided here, where the whole
            # timeline is in memory: the window against the run before it. Only the window
            # is written -- the full timeline runs to tens of megabytes.
            n = len(launches) // repeats
            window = launches[len(launches) - n :]
            prev = launches[len(launches) - 2 * n : len(launches) - n] if repeats > 1 else []
            (raw / "unitrace_launches.json").write_text(json.dumps({
                "repeats": repeats, "total": len(launches),
                "stray": len(launches) - n * repeats,
                "match": round(_same_work(window, prev), 4) if prev else None,
                "trace_steps": args.trace_steps if short else None,
                "launches": window,
            }) + "\n")  # fmt: skip
        from xe_forge.core.compute_metrics import summarize_compute_metrics

        records = []
        metrics = sorted(scratch.glob("unitrace*.metrics.*"))
        for path in metrics:
            with open(path, errors="replace") as f:
                records += summarize_compute_metrics(f)["records"]
        # A record is a counter only with a time to weight it by and a counter measured.
        usable = [
            r
            for r in _usable(records)
            if any(v is not None for v in (r.get("counters") or {}).values())
        ]
        if usable:
            (raw / "counters.json").write_text(json.dumps({"records": records}, indent=1) + "\n")
        elif sampling:
            # Sampling was asked for and nothing usable came back. This is the failure that
            # hides: unitrace exits 0, the capture succeeds, every bound silently falls back
            # to arithmetic, and the metrics files go with the scratch directory below -- so
            # say it here, with the interval, while the evidence still exists. Keep the
            # files for a look: they are the only way to tell a truncated collection from
            # one that never started, or one that sampled without reading a counter.
            kept = raw.parent / "metrics-no-records"
            shutil.rmtree(kept, ignore_errors=True)
            kept.mkdir(parents=True, exist_ok=True)
            for path in metrics:
                shutil.copy2(path, kept / path.name)
            print(f"capture run: PROBLEM metric sampling at {args.sampling_interval} us "
                  f"produced no usable counter records from {len(metrics)} metrics file(s) "
                  f"({len(records)} record(s) parsed, none with a time to weight it by). "
                  f"Every bound will fall back to arithmetic. Too fine an interval does "
                  f"this -- unitrace stops partway and writes a truncated file with exit 0. "
                  f"Try a coarser --sampling-interval-us (25 works on every launcher measured "
                  f"here). The files are kept in {kept}")  # fmt: skip
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # Pass 1's meta.json: the timed pass is the one the capture describes.
    run = json.loads((raw / "meta.json").read_text())
    # Settings that change the run are the launcher's, in its meta.json runtime.
    # trace_steps: what the instrumented pass actually ran, not what was asked for.
    run.update(launcher=str(launcher), launcher_sha256=_sha256(launcher),
               trace_steps=args.trace_steps if short else None)  # fmt: skip

    (raw / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    (raw / "MANIFEST.json").write_text(json.dumps(_manifest(raw), indent=1) + "\n")
    total = sum(k["us"] for k in kernels.values())
    # Usable records, not records: the count is read as "the capture has counters".
    print(f"capture run: {len(kernels)} kernels, {total / 1e3:.1f} ms device time (unitrace), "
          f"counters for {len(usable)} of {len(records)} records -> {raw}")  # fmt: skip
    return 0


def _ops_summary(ops_map: dict, window: list, by_launch: bool):
    """An agent's op map over ``window`` ([kernel, us, calls] per launch, or per kernel name
    without a timeline), timed by unitrace; problems where it does not hold.

    An op claims every item whose kernel matches one of its ``kernels`` patterns, or, given
    ``launches`` ([[start, end], ...] into the timeline), exactly those launches -- each of
    which must then match one of its patterns, when it gives any.
    """
    from xe_forge.frontend.ir import CapturedOp, TensorArg
    from xe_forge.frontend.torch_trace import TraceSummary

    problems, owner, ops, unknown, first = [], {}, [], {}, {}
    for key, e in ops_map.items():
        regs = []
        for pat in e.get("kernels") or []:
            try:
                regs.append(re.compile(pat))
            except re.error as exc:
                problems.append(f"{key}: kernel pattern {pat!r} is not a regex ({exc})")
        matches = {i for i, item in enumerate(window) if any(r.search(item[0]) for r in regs)}
        for r in regs:
            if not any(r.search(item[0]) for item in window):
                problems.append(
                    f"{key}: kernel pattern {r.pattern!r} matches no kernel unitrace saw"
                )
        # Bounds size tensors by dtype: a name they do not know leaves the op unbounded.
        for a in e.get("args") or []:
            if isinstance(a, dict) and a.get("dtype") and a["dtype"] not in STORAGE_BYTES:
                unknown.setdefault(a["dtype"], key)
        ranges = e.get("launches")
        if ranges is None:
            idx = matches
        elif not by_launch:
            problems.append(f"{key}: launches need unitrace's per-launch timeline; none recorded")
            idx = set()
        else:
            idx = set()
            for rg in ranges if isinstance(ranges, list) else [ranges]:
                ok = (isinstance(rg, list) and len(rg) == 2 and all(type(v) is int for v in rg)
                      and 0 <= rg[0] < rg[1] <= len(window))  # fmt: skip
                if not ok:
                    problems.append(f"{key}: launch range {rg!r} is not [start, end) within "
                                    f"the {len(window)} launches")  # fmt: skip
                    continue
                idx |= set(range(*rg))
            stray = sorted(i for i in idx if regs and i not in matches)
            if stray:
                problems.append(f"{key}: launch {stray[0]} ({window[stray[0]][0]!r}) matches "
                                f"none of its kernel patterns")  # fmt: skip
        for i in sorted(idx):
            if i in owner and owner[i] != key:
                problems.append(f"kernel {window[i][0]!r} (launch {i}) claimed by both "
                                f"{owner[i]!r} and {key!r}")  # fmt: skip
            owner[i] = key
        symbols: dict = {}
        counts: dict = {}
        for i in idx:
            name, us, calls = window[i]
            symbols[name] = round(symbols.get(name, 0.0) + us, 3)
            counts[name] = counts.get(name, 0) + calls
        arg_list = e.get("args") or []
        args = [TensorArg(list(a.get("shape") or []), a.get("dtype")) for a in arg_list]
        meta = dict(e.get("meta") or {}, arg_names=[a.get("name") for a in arg_list])
        # names.json describes an op type once (its "gemm" names arguments): its nodes must
        # name their arguments alike, by role, not by the tensor each node happens to read.
        fop = e.get("framework_op") or key
        k0, n0 = first.setdefault((fop, len(arg_list)), (key, meta["arg_names"]))
        if meta["arg_names"] != n0 and k0 is not None:
            problems.append(f"{fop}: nodes name their arguments differently ({n0} at {k0!r}, "
                            f"{meta['arg_names']} at {key!r}); name each by its role on the "
                            f"op, the same on every node")  # fmt: skip
            first[(fop, len(arg_list))] = (None, n0)  # once per type
        op = CapturedOp(id="", framework_op=e.get("framework_op") or key, args=args,
                        call_site=e.get("call_site") or "", framework_meta=meta)  # fmt: skip
        op.calls = int(e.get("calls") or max(counts.values(), default=0))
        op.kernel_symbols = dict(sorted(symbols.items()))
        op.device_us = round(sum(symbols.values()), 3)
        ops.append(op)
    ops.sort(key=lambda o: -o.device_us)
    for i, op in enumerate(ops):
        op.id = f"op-{i:04d}"
    kernels: dict = {}
    for name, us, _ in window:
        kernels[name] = round(kernels.get(name, 0.0) + us, 3)
    summary = TraceSummary(ops=ops, total_us=sum(item[1] for item in window),
                           attributed_us=sum(window[i][1] for i in owner), kernels=kernels,
                           launches=sum(item[2] for item in window))  # fmt: skip
    for name, key in unknown.items():
        problems.append(f"dtype {name!r} (first at {key}) is not one of {', '.join(STORAGE_BYTES)}: "
                        "the bounds size tensors by torch's names")  # fmt: skip
    return summary, problems


def parse_capture(args) -> int:
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
    problems = []
    # unitrace times the whole process; a launcher that ran the identical workload N times
    # (its warm-up included) says so, and one run is the last 1/N of the timeline.
    timeline = raw / "unitrace_launches.json"
    if timeline.is_file():
        t = json.loads(timeline.read_text())
        match = t.get("match")
        if match is not None:
            if match < 0.9:
                problems.append(
                    f"the last 1/{t['repeats']} of the timeline's {t['total']} launches is "
                    f"not the work the run before it did ({match:.0%} of its kernel time "
                    f"matches); meta.json's repeats says the launcher ran the identical "
                    f"workload {t['repeats']} times"
                )
        elif t["total"] % t["repeats"]:
            # raw/ from before "match" was recorded: the old divisibility test, which is a
            # parity test at repeats=2 and fails on one stray launch.
            problems.append(f"the timeline's {t['total']} launches are not {t['repeats']} "
                            "identical runs (repeats in meta.json)")  # fmt: skip
        window, by_launch = [[n, us, 1] for n, us in t["launches"]], True
    else:
        repeats = int(json.loads((raw / "run.json").read_text()).get("repeats") or 1)
        per_name = json.loads((raw / "unitrace_kernels.json").read_text())
        window = [[k, v["us"] / repeats, round(v["calls"] / repeats)] for k, v in per_name.items()]
        by_launch = False
    kernels = {}
    for name, us, _ in window:
        kernels[name] = kernels.get(name, 0.0) + us
    truth = sum(kernels.values())
    names = json.loads(Path(args.names).read_text()) if args.names else None
    summary = None
    if (raw / "ops.json").is_file():
        ops_map = json.loads((raw / "ops.json").read_text())
        summary, more = _ops_summary(ops_map, window, by_launch)
        problems += more
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
    seen = sum(us for k, us in traced.items() if _is_seen(k, kernels))
    known_pct = 100 * seen / sum(traced.values()) if traced else 0.0
    # unitrace's total is this run's only when both passes did the same work; with
    # --trace-steps it is a shorter run of the same model, so say so rather than invite the
    # two numbers to be compared.
    saw = (f"unitrace saw {truth / 1e3:.1f} ms over {r['trace_steps']} step(s)"
           if r.get("trace_steps") else f"unitrace saw {truth / 1e3:.1f} ms")  # fmt: skip
    print(f"capture parse: {len(capture.ops)} ops, {len(capture.workloads)} workloads, "
          f"{r['total_device_us'] / 1e3:.1f} ms device time, {r['attributed_pct']}% attributed; "
          f"{saw}; {known_pct:.1f}% of op time is in kernels unitrace "
          f"names -> {out}")  # fmt: skip
    # The inputs the launcher says it ran on (runtime.inputs: name -> {"file", "sha256"}) are
    # the ones it was given: each file still hashes to its recorded sha256.
    for name, inp in ((r.get("runtime") or {}).get("inputs") or {}).items():
        f = Path(inp.get("file") or "")
        if not f.is_file() or _sha256(f) != inp.get("sha256"):
            problems.append(f"input {name!r}: {f} is missing or not the file the run was given")
    problems += _precision_problems(r, named=names is not None, ops=capture.ops)
    if not capture.ops:
        problems.append("no ops")
    least = getattr(args, "min_known", 90.0)
    if known_pct < least:
        problems.append(f"only {known_pct:.1f}% of op time is in kernels unitrace saw "
                        f"(--min-known {least:g})")  # fmt: skip
    for p in problems:
        print(f"capture parse: PROBLEM {p}")
    return 1 if problems else 0


def kv_args(ops) -> dict[tuple, set[str]]:
    """{(op key, argument): its dtypes over the op's calls} of every argument the namer names
    the stored KV cache (``kv_args``) that has a dtype."""
    from xe_forge.frontend.canonical import arg_label

    out: dict[tuple, set[str]] = {}
    for o in ops:
        key = o.framework_meta.get("name_key", o.framework_op)
        want = (o.framework_meta.get("named") or {}).get("kv_args") or []
        for i, a in enumerate(o.args):
            if a is not None and a.dtype and arg_label(o, i) in want:
                out.setdefault((key, arg_label(o, i)), set()).add(a.dtype)
    return out


def _kv_problems(run: dict, ops: list) -> list[str]:
    """The declared ``runtime.kv_cache`` against the named KV-cache arguments: each holds it
    in some call -- as that dtype or a byte container (uint8/int8) of its size or of a sub-byte
    dtype packed into it; a generic op
    (an indexed store) may write other state in its other calls. None named: no check."""
    declared = (run.get("runtime") or {}).get("kv_cache")
    if declared in (None, "auto"):
        return []
    try:
        want = canonical_dtype(str(declared))
    except ValueError as exc:
        return [f"runtime.kv_cache: {exc}"]

    def holds(d):
        return d == want or (d in ("uint8", "int8") and STORAGE_BYTES[want] <= 1)

    return [f"declared KV cache {want}, {key!r} holds it in {arg} as {', '.join(sorted(ds))}"
            for (key, arg), ds in kv_args(ops).items() if not any(holds(d) for d in ds)]  # fmt: skip


def _precision_problems(run: dict, named: bool = False, ops: list = ()) -> list[str]:
    """The declared ``runtime.precision`` / ``runtime.quantization`` against what the trace ran;
    both must name a dtype. Once ``named``, the GEMMs' weights are known: declared weights
    none of them read is a problem too, and GEMMs reading more bytes than stored are one
    unless the namer names the op that widens them (``model.weights_widened_by``); a named op
    must be one of ``ops`` reading the stored dtype. The namer decides; this checks its claim."""
    rt, seen = run.get("runtime") or {}, run["precision"]
    problems = _kv_problems(run, ops) if named else []
    try:
        want = canonical_dtype(str(rt.get("precision")))
        _, w_dtype = parse_quantization(rt.get("quantization"))
    except ValueError as exc:
        return [*problems, f"runtime.precision: {exc}"]
    if w_dtype is None:
        return [*problems, f"runtime.quantization {rt.get('quantization')!r} names no stored dtype: write "
                f"'<scheme>:<dtype>' (uint8 for packed sub-byte formats)"]  # fmt: skip
    got = seen["compute"]
    if want and got != want:
        share = f"{100 * seen['dtypes'][got]:.0f}% in {got}" if got else "no float tensor"
        problems.append(f"declared compute {want}, the trace ran {share}")
    w_bytes = seen["w_bytes"]
    if named and w_dtype and w_bytes is None:
        problems.append(f"declared weights {w_dtype}, but no GEMM's weight was captured: "
                        f"the launcher must record each node's weights among its args")  # fmt: skip
    ratio = round(w_bytes / STORAGE_BYTES[w_dtype], 3) if w_bytes is not None else 1
    # uint8 is also the byte container of packed sub-byte formats: no lower bound for it.
    if ratio < 1.0 and w_dtype != "uint8":
        problems.append(f"declared weights {w_dtype} ({STORAGE_BYTES[w_dtype]} B/element), the "
                        f"GEMMs read {w_bytes} B per weight element")  # fmt: skip
    claim = (run.get("model_facts") or {}).get("weights_widened_by")
    # Read with the weight: its scales and zero points. A quarter more, or for sub-byte weights
    # 0.125 B per element (an fp16 scale and zero per 32): int2 at 0.375, not int4's 0.5.
    stored = STORAGE_BYTES[w_dtype]
    if w_bytes is not None and w_bytes > max(1.25 * stored, stored + 0.125) and not claim:
        problems.append(f"declared weights {w_dtype} ({STORAGE_BYTES[w_dtype]} B/element), the "
                        f"GEMMs read {w_bytes} B per weight element: loaded wide, or widened on "
                        f"the fly by an op -- then names.json's model.weights_widened_by names "
                        f"it")  # fmt: skip
    if named and claim:  # every claim is checked, needed or not
        reads = any(o.framework_meta.get("name_key", o.framework_op) == claim
                    and any(a is not None and a.dtype == w_dtype for a in o.args) for o in ops)  # fmt: skip
        if not reads:
            problems.append(f"model.weights_widened_by {claim!r}: no such op reads {w_dtype}")
    return problems


def run(args) -> int:
    if args.capture_command == "run":
        return run_capture(args)
    return parse_capture(args)


# ── raw/ -> capture.json ───────────────────────────────────────────────────


def build(raw_dir: Path, out: Path, names: dict | None = None, summary=None) -> ir.CaptureRun:
    """``raw/`` (run.json + traces) -> capture.json, with ops named from ``names``."""
    run = json.loads((raw_dir / "run.json").read_text())
    traces = sorted(raw_dir.glob("trace*.json*"))
    ranks: dict[int, list[Path]] = {}
    for t in traces:
        found = re.search(r"rank(\d+)", t.name)
        ranks.setdefault(int(found.group(1)) if found else 0, []).append(t)
    if summary is None and traces:
        summary = parse_traces(ranks[min(ranks)])  # TP ranks run the same shapes: rank 0
    elif summary is None:
        raise SystemExit(f"no trace in {raw_dir}")

    if not summary.launches:
        raise SystemExit(f"capture: no device kernel in {raw_dir}; nothing ran on the device")
    run["total_device_us"] = round(summary.total_us, 1)
    run["attributed_device_us"] = round(summary.attributed_us, 1)
    run["attributed_pct"] = round(summary.attributed_pct, 2)
    run["kernel_invocations"] = summary.launches
    _annotate_schemas(summary.ops, raw_dir / "schemas.json")
    summary.ops = apply_names(summary.ops, names)
    if names is not None:
        run["model_facts"] = names.get("model") or {}
    if (raw_dir / "counters.json").is_file():
        run["counters"] = _counters(json.loads((raw_dir / "counters.json").read_text()))
    # The same pass the counters came from, per kernel name: what their coverage is judged
    # against. The trace's time for a kernel is the timed pass's, which --trace-steps makes
    # a longer run, so the two are not comparable.
    if (raw_dir / "unitrace_kernels.json").is_file():
        run["unitrace_kernels"] = json.loads((raw_dir / "unitrace_kernels.json").read_text())
    capture = ir.CaptureRun(run=run, ops=summary.ops)
    capture.workloads = canonical.workloads(capture.ops, summary.total_us, naming.name_op)
    run["precision"] = precision_record(capture)
    ir.dump(capture, out)
    return capture


def precision_record(capture: ir.CaptureRun) -> dict:
    """What the run computed in, and the time-weighted bytes per weight element of its GEMMs."""
    gemms = [(w.attrs["w_bytes"], w.device_us) for w in capture.workloads if "w_bytes" in w.attrs]
    gemm_us = sum(us for _, us in gemms)
    shares = observed(capture.ops)
    return {"compute": compute_dtype(list(shares)), "dtypes": shares,
            "w_bytes": round(sum(b * us for b, us in gemms) / gemm_us, 4) if gemm_us else None,
            "kv_args": {f"{k}.{a}": sorted(ds) for (k, a), ds in kv_args(capture.ops).items()}}  # fmt: skip


def apply_names(ops: list, names: dict | None) -> list:
    """``ops`` with names.json's entry on each (split per kernel group it names)."""
    entries = (names or {}).get("ops") or {}
    out = []
    for op in ops:
        op.framework_meta["name_key"] = naming.key(op)
        if naming.key(op) in entries:
            op.framework_meta["named"] = entries[naming.key(op)]
        out.extend(_split(op))
    return out


def _usable(records: list) -> list:
    """The records that carry a counter at all.

    ``_counters`` weights every rate by ``time_ns`` and skips a record without one, so such
    a record is no counter, however many of them come back. Counting all records instead
    would hide a capture with no usable counters from the no-records branch.
    """
    return [r for r in records if r.get("time_ns")]


def _counters(summary: dict) -> dict:
    """unitrace ComputeBasic records -> {kernel: {gb_per_s, util_pct, us, gb_us, util_us}},
    time-weighted.

    compute_metrics reports an incomplete counter as None (unknown); a rate is averaged over
    the records that measured it, and stays None when none did -- never 0. ``gb_us`` and
    ``util_us`` are the time each rate was measured over, so a rate a few records measured
    is not read as the kernel's.
    """
    acc: dict = {}
    for r in summary.get("records") or []:
        time_ns = r.get("time_ns")
        if not time_ns:
            continue
        name = _SIMD.sub("", r["kernel"]).strip()
        a = acc.setdefault(
            name, {"ns": 0.0, "bytes": 0.0, "bytes_ns": 0.0, "util": 0.0, "util_ns": 0.0}
        )
        a["ns"] += time_ns
        moved = [((r.get("memory") or {}).get(d) or {}).get("bytes") for d in ("read", "write")]
        if None not in moved:
            a["bytes"] += sum(moved)
            a["bytes_ns"] += time_ns
        utils = [v for k, v in (r.get("counters") or {}).items()
                 if k.endswith("_UTILIZATION[%]") and v is not None]  # fmt: skip
        if utils:
            a["util"] += max(utils) * time_ns
            a["util_ns"] += time_ns
    return {
        k: {"gb_per_s": round(a["bytes"] / a["bytes_ns"], 3) if a["bytes_ns"] else None,
            "util_pct": round(a["util"] / a["util_ns"], 2) if a["util_ns"] else None,
            "us": round(a["ns"] / 1e3, 3), "gb_us": round(a["bytes_ns"] / 1e3, 3),
            "util_us": round(a["util_ns"] / 1e3, 3)}
        for k, a in acc.items()
    }  # fmt: skip


def _split(op) -> list:
    """Split an op into one row per kernel group in ``named["kernels"]``."""
    named = op.framework_meta.get("named") or {}
    groups = named.get("kernels")
    if not isinstance(groups, dict) or not groups:
        return [op]
    by: dict = {}
    for sym, us in op.kernel_symbols.items():
        sub = next((g for g in groups if g in sym), None)
        by.setdefault(sub, {})[sym] = us
    parts = []
    for sub, syms in by.items():
        part = copy.deepcopy(op)
        part.kernel_symbols = syms
        part.device_us = round(sum(syms.values()), 3)
        entry = {k: v for k, v in named.items() if k != "kernels"}
        if sub is not None and isinstance(groups[sub], dict):
            entry.update(groups[sub])
        part.framework_meta["named"] = entry
        parts.append(part)
    return parts


def _schema_args(op_name: str, nargs: int) -> list[str] | None:
    """Argument names of the overload of ``ns::op`` taking ``nargs`` arguments."""
    try:
        import torch

        ns, name = op_name.split("::", 1)
        packet = getattr(getattr(torch.ops, ns), name)
        for overload in packet.overloads():
            args = getattr(packet, overload)._schema.arguments
            if len(args) == nargs:
                return [a.name for a in args]
    except Exception:
        return None
    return None


def _annotate_schemas(ops: list[ir.CapturedOp], cache: Path) -> None:
    """Name each op's arguments from its schema, cached in ``cache`` for a re-parse elsewhere."""
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    for op in ops:
        if "::" not in op.framework_op or op.framework_op.startswith("kernel::"):
            continue
        key = f"{op.framework_op}/{len(op.args)}"
        if key not in known:
            names = _schema_args(op.framework_op, len(op.args))
            if names is None:
                continue
            known[key] = names
        op.framework_meta["arg_names"] = known[key]
    if known:
        cache.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
