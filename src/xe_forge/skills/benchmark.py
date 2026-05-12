"""xe-forge-skill benchmark: Correctness + performance comparison."""


def run(args):
    from pathlib import Path

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.spec_loader import load_spec

    baseline_code = Path(args.baseline).read_text()
    optimized_code = Path(args.optimized).read_text()

    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    input_shapes = spec.get_input_shapes(variant)
    flop = spec.get_flop(variant)
    dtype_name = spec.get_dtype(variant)
    init_args = spec.get_init_args(variant)

    dtype = None
    if dtype_name is not None:
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(str(dtype_name))

    executor = KernelBenchExecutor(device=args.device)

    if args.baseline_us is not None:
        baseline_us = [float(v) for v in str(args.baseline_us).split(",")]
        print(f"Using cached baseline: {baseline_us} us")
        optimized_result = executor.execute(
            optimized_code,
            None,
            input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
        )
        if optimized_result.success:
            baseline_ms = sum(baseline_us) / len(baseline_us) / 1000.0
            opt_ms = optimized_result.execution_time_ms
            speedup = baseline_ms / opt_ms if opt_ms > 0 else 0
            print(f"Correctness: {'PASSED' if optimized_result.success else 'FAILED'}")
            print(
                f"Performance: baseline_us={baseline_ms * 1000:.2f}, "
                f"triton_us={opt_ms * 1000:.2f}, speedup={speedup:.2f}x"
            )
        else:
            print("Correctness: FAILED")
            print(f"Error: {optimized_result.error_message}")
    else:
        result = executor.compare_kernels(
            original_code=baseline_code,
            optimized_code=optimized_code,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
        )
        print(f"Correctness: {'PASSED' if result.optimized_correct else 'FAILED'}")
        if result.original_time_us and result.optimized_time_us:
            print(
                f"Performance: baseline_us={result.original_time_us:.2f}, "
                f"triton_us={result.optimized_time_us:.2f}, speedup={result.speedup:.2f}x"
            )
        if result.feedback_message:
            print(f"Feedback: {result.feedback_message}")
