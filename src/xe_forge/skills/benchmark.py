"""xe-forge-skill benchmark: Correctness + performance comparison.

Two DSL paths, dispatched on ``args.dsl``:
  * ``_run_triton`` — PyTorch/Triton kernels via ``KernelBenchExecutor``
    (original-vs-optimized correctness).
  * ``_run_sycl``   — CUTLASS SYCL ``.cpp`` kernels via ``SyclExecutor``,
    correctness checked against a **golden PyTorch/numpy reference** computed on
    the same (bit-identical) inputs.

Both print the same lines so ``tool-runner`` / ``trial result --triton-us``
parsing stays uniform across DSLs (the ``triton_us=`` token is kept verbatim).
"""


def run(args):
    if getattr(args, "dsl", "triton") == "sycl":
        return _run_sycl(args)
    return _run_triton(args)


def _perf_line(baseline_us, opt_us, speedup, tflops=None, peak_tflops=None):
    """Build the uniform Performance: line, appending TFLOPS + utilization.

    ``tflops`` is the optimized kernel's achieved throughput; ``peak_tflops`` is
    the device's theoretical peak (config.device_config.peak_tflops) used to
    report utilization as a percentage. Both are optional — the us/speedup part
    is always emitted, so the line stays parseable when TFLOPS is unavailable.
    """
    line = (
        f"Performance: baseline_us={baseline_us:.2f}, "
        f"triton_us={opt_us:.2f}, speedup={speedup:.2f}x"
    )
    if tflops is not None:
        line += f", tflops={tflops:.2f}"
        if peak_tflops:
            line += f", util={tflops / peak_tflops * 100:.1f}%"
    return line


def _peak_tflops():
    """Theoretical peak TFLOPS from the active device config (for utilization)."""
    from xe_forge.config import get_config

    return get_config().device_config.peak_tflops


def _run_triton(args):
    from pathlib import Path

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.spec_loader import load_spec

    baseline_code = Path(args.baseline).read_text()
    optimized_code = Path(args.optimized).read_text()

    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    input_shapes = spec.get_input_shapes(variant)
    flop = spec.get_flop(variant)
    dtype = spec.get_dtype(variant)
    input_dtypes = spec.get_input_dtypes(variant)
    init_args = spec.get_init_args(variant)

    executor = KernelBenchExecutor(device=args.device)
    peak = _peak_tflops()

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
            input_dtypes=input_dtypes,
        )
        if optimized_result.success:
            baseline_ms = sum(baseline_us) / len(baseline_us) / 1000.0
            opt_ms = optimized_result.execution_time_ms
            speedup = baseline_ms / opt_ms if opt_ms > 0 else 0
            print(f"Correctness: {'PASSED' if optimized_result.success else 'FAILED'}")
            print(
                _perf_line(
                    baseline_ms * 1000,
                    opt_ms * 1000,
                    speedup,
                    tflops=optimized_result.tflops,
                    peak_tflops=peak,
                )
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
            input_dtypes=input_dtypes,
        )
        print(f"Correctness: {'PASSED' if result.optimized_correct else 'FAILED'}")
        if result.original_time_us and result.optimized_time_us:
            print(
                _perf_line(
                    result.original_time_us,
                    result.optimized_time_us,
                    result.speedup,
                    tflops=result.optimized_tflops,
                    peak_tflops=peak,
                )
            )
        if result.feedback_message:
            print(f"Feedback: {result.feedback_message}")


# ---------------------------------------------------------------------------
# SYCL path
# ---------------------------------------------------------------------------


def _bf16_bin_to_torch(path: str, rows: int, cols: int, dtype):
    """Read a tensor .bin written by SyclExecutor._save_tensor back into torch.

    bfloat16 is stored as raw int16 bits (NumPy has no bf16); other dtypes are
    stored as their native numpy bytes. Layout is row-major [rows, cols].
    """
    import numpy as np
    import torch

    if dtype == torch.bfloat16:
        raw = np.fromfile(path, dtype=np.int16)
        t = torch.from_numpy(raw).view(torch.bfloat16)
    else:
        np_dtype = {
            torch.float16: np.float16,
            torch.float32: np.float32,
        }.get(dtype, np.float32)
        raw = np.fromfile(path, dtype=np_dtype)
        t = torch.from_numpy(raw)
    return t.reshape(rows, cols)


def _compute_golden(reference_path, input_dir, m, n, k, dtype):
    """Run the PyTorch golden reference on the bit-identical .bin inputs.

    Reads A.bin [M,K], B0.bin [K,N], B1.bin [K,N] (the GEMM-shaped inputs
    SyclExecutor.generate_inputs emits) and feeds (A, B0[, B1]) by position to
    the reference Model. Returns the result as a float32 numpy array.
    """
    import inspect
    import os

    from ai_bench.utils import import_from_path

    module = import_from_path("sycl_golden_ref", reference_path)
    if not hasattr(module, "Model"):
        raise ValueError(f"Reference {reference_path} has no Model class")

    # Instantiate the reference. GEMM goldens take no init args; fall back to
    # get_init_inputs() when the reference declares it (matches executor).
    if hasattr(module, "get_init_inputs"):
        model = module.Model(*module.get_init_inputs())
    else:
        model = module.Model()

    A = _bf16_bin_to_torch(os.path.join(input_dir, "A.bin"), m, k, dtype)
    B0 = _bf16_bin_to_torch(os.path.join(input_dir, "B0.bin"), k, n, dtype)

    # Pass B1 only when the reference's forward expects a third positional arg.
    forward_params = [
        p
        for p in inspect.signature(model.forward).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(forward_params) >= 3:
        B1 = _bf16_bin_to_torch(os.path.join(input_dir, "B1.bin"), k, n, dtype)
        out = model(A, B0, B1)
    else:
        out = model(A, B0)

    return out.float().detach().cpu().numpy()


def _run_sycl(args):
    import os

    from xe_forge.config import get_config
    from xe_forge.core.spec_loader import load_spec
    from xe_forge.core.sycl_executor import KernelType, SyclExecutor

    config = get_config()
    spec = load_spec(args.spec)
    variant = spec.resolve_variant(args.variant)
    dims = spec.get_dims(variant)
    dtype = spec.get_dtype(variant)

    m = int(dims.get("M", dims.get("N", 1024)))
    n = int(dims.get("N", m))
    k = int(dims.get("K", m))

    # Spec-driven tolerances, falling back to config defaults.
    rtol = spec.get_rtol(variant)
    atol = spec.get_atol(variant)
    if rtol is None:
        rtol = config.optimization.correctness_rtol
    if atol is None:
        atol = config.optimization.correctness_atol

    executor = SyclExecutor(kernel_type=KernelType.GEMM, verify=False)
    input_dir = executor.get_or_create_inputs(dims, seed=42, dtype=dtype)

    # Golden reference: {baseline_stem}_pytorch.py next to the baseline .cpp.
    baseline_stem = os.path.splitext(args.baseline)[0]
    reference_path = f"{baseline_stem}_pytorch.py"
    golden = None
    if os.path.exists(reference_path):
        try:
            golden = _compute_golden(reference_path, input_dir, m, n, k, dtype)
        except Exception as e:
            print(f"Warning: golden reference failed ({e}); correctness will be skipped")
    else:
        print(f"No PyTorch golden reference at {reference_path}; correctness will be skipped")

    # Correctness + optimized timing vs the golden array.
    opt_tflops = None
    if golden is not None:
        result = executor.compare_with_reference(
            golden_output=golden,
            optimized_path=args.optimized,
            dims=dims,
            rtol=rtol,
            atol=atol,
            input_dir=input_dir,
        )
        opt_ms = result.optimized_time_ms
        opt_tflops = result.optimized_tflops
        correct = result.optimized_correct
        feedback = result.feedback_message
    else:
        # No golden — just run the kernel for timing.
        run = executor.execute(
            kernel_path=args.optimized,
            dims=dims,
            output_name="optimized_sycl",
            input_dir=input_dir,
        )
        opt_ms = run.execution_time_ms if run.success else None
        opt_tflops = run.tflops if run.success else None
        correct = run.success
        feedback = run.error_message if not run.success else ""

    if not correct or opt_ms is None:
        print("Correctness: FAILED")
        if feedback:
            print(f"Feedback: {feedback}")
        return

    opt_us = opt_ms * 1000.0
    peak = config.device_config.peak_tflops

    # Baseline caching, uniform with Triton: t0 times the baseline .cpp; t1+
    # reuses the cached baseline_us and only computes the speedup.
    if args.baseline_us is not None:
        baseline_us_list = [float(v) for v in str(args.baseline_us).split(",")]
        baseline_us = sum(baseline_us_list) / len(baseline_us_list)
        print(f"Using cached baseline: {baseline_us:.2f} us")
    else:
        baseline_run = executor.execute(
            kernel_path=args.baseline,
            dims=dims,
            output_name="baseline_sycl",
            input_dir=input_dir,
        )
        if not baseline_run.success or not baseline_run.execution_time_ms:
            print("Correctness: PASSED")
            print(
                f"Warning: baseline kernel did not produce a timing: {baseline_run.error_message}"
            )
            print(_perf_line(0.0, opt_us, 0.0, tflops=opt_tflops, peak_tflops=peak))
            return
        baseline_us = baseline_run.execution_time_ms * 1000.0

    speedup = baseline_us / opt_us if opt_us > 0 else 0.0
    print("Correctness: PASSED")
    print(_perf_line(baseline_us, opt_us, speedup, tflops=opt_tflops, peak_tflops=peak))
    if feedback:
        print(f"Feedback: {feedback}")
