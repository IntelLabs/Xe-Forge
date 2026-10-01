"""xe-forge-skill profile: GPU hardware counter profiling (unitrace or VTune)."""


def run(args):
    from xe_forge.core.profiler import ProfileResult, UnitraceXPUProfiler, XPUProfiler

    metric_group = getattr(args, "metric_group", "ComputeBasic")
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

    # With a trial named, every attempt -- collected, failed or unavailable -- is recorded
    # against the saved trial's source, and an attempt already recorded is not repeated.
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
                f"== Collection: {group} == already attempted on this source "
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
        if metric_group == "all":
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

    if metric_group == "all" and (manager is not None or any(p.available() for p in profilers)):
        if first_result is not None:
            if manager is not None:
                print(
                    "Failed attempts are recorded for this trial and satisfy its profiling gate; "
                    "do not retry them on this source."
                )
            raise SystemExit(1)
        return

    if first_result is None:
        tried = " or ".join(type(p).__name__ for p in profilers) or "no profiler"
        print(f"Profiling error: none available ({tried}).")
        raise SystemExit(1)

    print(first_result.format_for_llm())
    raise SystemExit(1)
