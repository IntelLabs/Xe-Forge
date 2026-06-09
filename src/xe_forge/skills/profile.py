"""xe-forge-skill profile: VTune GPU profiling.

Dispatched on ``args.dsl``:
  * ``_profile_triton`` — PyTorch/Triton kernels via ``XPUProfiler`` (gpu-offload
    around a generated Python runner).
  * ``_profile_sycl``   — compiled CUTLASS SYCL ``.cpp`` via ``SyclProfiler``
    (gpu-hotspots characterization on the binary, no Python runner).
"""


def run(args):
    if getattr(args, "dsl", "triton") == "sycl":
        return _profile_sycl(args)
    return _profile_triton(args)


def _profile_triton(args):
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


def _profile_sycl(args):
    from xe_forge.core.spec_loader import load_spec
    from xe_forge.core.sycl_profiler import KernelType, SyclProfiler

    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    dims = spec.get_dims(variant)
    dtype = spec.get_dtype(variant)

    profiler = SyclProfiler(
        vtune_bin=args.vtune_bin,
        kernel_type=KernelType.GEMM,
        iterations=args.iters,
    )
    result = profiler.profile(kernel_path=args.kernel_file, dims=dims, dtype=dtype)
    print(result.format_for_llm())
