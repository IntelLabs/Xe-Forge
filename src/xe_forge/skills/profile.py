"""xe-forge-skill profile: VTune GPU profiling."""


def run(args):
    from xe_forge.core.profiler import XPUProfiler

    profiler = XPUProfiler(vtune_bin=args.vtune_bin)
    result = profiler.profile(
        args.kernel_file,
        spec_path=args.spec,
        variant=args.variant,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(result.format_for_llm())
