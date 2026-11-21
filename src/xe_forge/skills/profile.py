"""xe-forge-skill profile: hardware counter profiling (unitrace or VTune on a GPU; perf,
or VTune, on the host)."""

_HOST_ONLY = ("events", "metrics", "system_wide", "collector_args", "window")


def _host_profilers(args):
    """perf when it is there (or asked for), else VTune. Event, metric and analysis names
    are passed through as given: which to ask for is the agent's choice."""
    import shutil

    from xe_forge.core.profiler import PerfCPUProfiler, VTuneCPUProfiler

    if args.tool == "unitrace":
        raise SystemExit("unitrace profiles a GPU; on cpu use --tool perf, vtune or auto.")
    if getattr(args, "metric_group", "ComputeBasic") != "ComputeBasic":
        raise SystemExit(f"--metric-group {args.metric_group} is a unitrace group; on cpu use "
                         "--analysis.")  # fmt: skip
    perf_bin = getattr(args, "perf_bin", "perf")
    tool = args.tool
    if tool == "auto":
        tool = "perf" if shutil.which(perf_bin) else "vtune"
    common = {"top": getattr(args, "top", 15),
              "collector_args": getattr(args, "collector_args", None) or ()}  # fmt: skip
    if getattr(args, "window", None):
        common["window"] = args.window
    if tool == "vtune":
        for name in ("events", "metrics", "system_wide"):
            if getattr(args, name, None):
                flag = "--" + name.replace("_", "-")
                raise SystemExit(f"{flag} is a perf option; VTune takes --analysis and "
                                 "--collector-arg.")  # fmt: skip
        return [VTuneCPUProfiler(vtune_bin=args.vtune_bin, analysis=args.analysis or "hotspots",
                                 **common)]  # fmt: skip
    analysis = args.analysis or "stat"
    analyses = ("stat", "record") if analysis == "all" else (analysis,)
    if any(a not in ("stat", "record") for a in analyses):
        raise SystemExit(f"--analysis {analysis}: perf runs stat, record or all.")
    events, metrics = _names(args, "events"), _names(args, "metrics")
    try:
        return [PerfCPUProfiler(perf_bin=perf_bin, analysis=a, events=events,
                                metrics=metrics if a == "stat" else (),
                                system_wide=bool(getattr(args, "system_wide", False)), **common)
                for a in analyses]  # fmt: skip
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _names(args, name):
    return [n for part in (getattr(args, name, None) or []) for n in part.split(",") if n]


def _gpu_profilers(args, metric_group):
    from xe_forge.core.profiler import UnitraceXPUProfiler, XPUProfiler

    if args.tool == "perf":
        raise SystemExit("perf profiles the host; on a GPU use --tool unitrace, vtune or auto.")
    for name in ("analysis", *_HOST_ONLY):
        if getattr(args, name, None):
            flag = "--" + name.replace("_", "-").replace("collector-args", "collector-arg")
            raise SystemExit(f"{flag} is a host profiling option; on a GPU use --tool and "
                             "--metric-group.")  # fmt: skip
    if metric_group != "ComputeBasic" and args.tool == "vtune":
        raise SystemExit(f"{metric_group} requires --tool unitrace (or auto).")
    profilers = []
    if args.tool in ("auto", "unitrace"):
        groups = ("ComputeBasic", "EuStallSampling") if metric_group == "all" else (metric_group,)
        for group in groups:
            profilers.append(
                UnitraceXPUProfiler(
                    unitrace_bin=args.unitrace_bin,
                    parent_profile=getattr(args, "parent_profile", None),
                    capture_assembly=getattr(args, "assembly", False),
                    sampling_interval_us=getattr(args, "sampling_interval_us", None),
                    metric_group=group,
                )
            )
    if args.tool in ("auto", "vtune") and metric_group == "ComputeBasic":
        profilers.append(XPUProfiler(vtune_bin=args.vtune_bin))
    return profilers


def run(args):
    from xe_forge.core.profiler import ProfileResult

    metric_group = getattr(args, "metric_group", "ComputeBasic")
    device = getattr(args, "device", None)
    if not device:
        from xe_forge.frontend.profile import device_type

        device = device_type()
    host = device.split(":")[0] == "cpu"
    profilers = _host_profilers(args) if host else _gpu_profilers(args, metric_group)
    # Several collections of one source, each printed and recorded (unitrace's groups, or
    # perf's stat and record).
    every = metric_group == "all" or (host and len(profilers) > 1)

    # With a trial named, every attempt -- collected, failed or unavailable -- is recorded
    # against the saved trial's source; a final attempt (collected, or failed twice) is not repeated.
    kernel_name, trial_id = getattr(args, "kernel_name", None), getattr(args, "trial_id", None)
    manager = digest = None
    if kernel_name or trial_id:
        if not (kernel_name and trial_id):
            raise SystemExit("--kernel-name and --trial-id must be given together.")
        from xe_forge.core.trial_manager import TrialManager

        manager = TrialManager(getattr(args, "trials_dir", "./trials"))
        try:
            digest = manager.profile_source_hash(kernel_name, trial_id, args.kernel_file)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"Profiling refused: {exc}") from exc

    first_result = None
    for profiler in profilers:
        group = getattr(profiler, "metric_group", "VTune")
        previous = (
            manager.recorded_profile(kernel_name, trial_id, group, digest) if manager else None
        )
        if previous is not None:
            print(
                f"== Collection: {group} == final on this source "
                f"({previous['status']}); not re-collected. Artifacts: {previous['artifacts_dir']}"
            )
            result = ProfileResult(error=previous["error"], artifacts_dir=previous["artifacts_dir"])
        else:
            if not profiler.available():
                if manager is None:
                    continue
                result = ProfileResult(error=f"{type(profiler).__name__} not available")
            else:
                result = profiler.profile(
                    args.kernel_file,
                    spec_path=args.spec,
                    variant=args.variant,
                    warmup=args.warmup,
                    iters=args.iters,
                    reference_path=getattr(args, "reference", None),
                )
            if manager is not None:
                manager.record_profile(
                    kernel_name,
                    trial_id,
                    group,
                    digest,
                    artifacts_dir=result.artifacts_dir,
                    error=result.error,
                    warnings=[r.message for r in result.recommendations],
                )
        if every:
            if previous is None:
                print(f"== Collection: {group} ==")
                print(result.format_for_llm())
            if result.error is not None:
                first_result = first_result or result
            continue
        if result.error is None:
            if previous is None:
                print(result.format_for_llm())
            return
        first_result = first_result or result

    if every and (manager is not None or any(p.available() for p in profilers)):
        if first_result is not None:
            if manager is not None:
                print(
                    "A failed collection is re-run once on the same source; after that it is "
                    "final. Report it with its artifacts; do not fabricate results."
                )
            raise SystemExit(1)
        return

    if first_result is None:
        tried = " or ".join(type(p).__name__ for p in profilers) or "no profiler"
        print(f"Profiling error: none available ({tried}).")
        raise SystemExit(1)

    print(first_result.format_for_llm())
    raise SystemExit(1)
