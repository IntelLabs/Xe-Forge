"""xe-forge-skill benchmark: Correctness + performance comparison.

Two paths. The built-in one measures with :class:`KernelBenchExecutor` against
tensors generated from the spec's shapes. The other delegates to a command the
host supplied (``--external-benchmark``, ``EXTERNAL_BENCHMARK``, or
``external.benchmark`` in config) -- see :mod:`xe_forge.external` for the
contract and for why a host that owns real workload data and a calibrated timer
should be the one answering.

Both print the same two lines, because the trial loop branches on them:

    Correctness: PASSED|FAILED
    Performance: baseline_us=..., kernel_us=..., speedup=...x

The external path adds a ``VERDICT:`` line naming the gate that decided, and
omits the speedup when the host refused to compute one -- a comparison the host
could not distinguish from noise is not a regression, and must not be reported
as a number the loop can branch away from.
"""

import os
import sys

BUILTIN_ENV = "XE_FORGE_BUILTIN_BENCHMARK"


def _external_template(args) -> str | None:
    """The host-supplied benchmark command, if there is one."""
    explicit = getattr(args, "external_benchmark", None)
    if explicit:
        return explicit
    from xe_forge.config import get_config

    return get_config().external.benchmark


def _run_external(args, template: str) -> int:
    from pathlib import Path

    from xe_forge.config import get_config
    from xe_forge.external import ExternalCommandError, run_external

    optimized = Path(args.optimized)
    spec_path = Path(args.spec) if args.spec else None
    kernel_name = spec_path.stem if spec_path else optimized.stem

    try:
        result = run_external(
            template,
            timeout=get_config().external.timeout,
            kernel=kernel_name,
            baseline=str(Path(args.baseline).resolve()) if args.baseline else "",
            trial=str(optimized.resolve()),
            optimized=str(optimized.resolve()),
            spec=str(spec_path.resolve()) if spec_path else "",
            variant=args.variant or "",
            device=args.device or "",
            dsl=args.dsl or "",
            workspace=str(Path.cwd()),
        )
    except ExternalCommandError as exc:
        print(f"Correctness: FAILED\nVERDICT: EXTERNAL_ERROR\nError: {exc}")
        return 1

    if result.correctness is None:
        # The host measured something but never said whether it was right.
        # Reading that as a pass would let a fast wrong kernel become the best
        # trial, so it is refused here rather than downstream.
        print("Correctness: FAILED")
        print("VERDICT: NO_CORRECTNESS_VERDICT")
        print("Error: external benchmark reported no CORRECTNESS line")
        return 1

    print(f"Correctness: {'PASSED' if result.correctness else 'FAILED'}")
    if result.correctness and result.measured:
        print(
            f"Performance: baseline_us={result.baseline_us:.2f}, "
            f"kernel_us={result.trial_us:.2f}, speedup={result.speedup:.2f}x"
        )
    elif result.correctness:
        # No ratio came back. Say what is known without implying a comparison.
        parts = []
        if result.baseline_us is not None:
            parts.append(f"baseline_us={result.baseline_us:.2f}")
        if result.trial_us is not None:
            parts.append(f"kernel_us={result.trial_us:.2f}")
        print(f"Performance: {', '.join(parts) if parts else 'not measured'}, speedup=none")
    else:
        # A failed trial is the one case where the host's own words are worth
        # more than the parsed fields: the reason it failed -- a compiler
        # diagnostic, a mismatched element, a shape -- is what the next trial
        # is written from, and none of it survives the contract.
        print("Output:")
        print(result.raw.rstrip())
    if result.timer:
        print(f"TIMER: {result.timer}")
    print(f"VERDICT: {result.verdict or 'UNSPECIFIED'}")
    return result.returncode


def _run_builtin(args) -> int:
    from pathlib import Path

    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.spec_loader import load_spec

    reference_path = getattr(args, "reference", None)
    if not args.spec and (not reference_path or args.variant):
        print("Correctness: FAILED")
        print("Error: without --spec, supply --reference and omit --variant")
        return 1

    # One file each: Xe-Forge's own executor compiles a single source, so a split kernel
    # has no route through it. Say so rather than letting read_text raise -- a host-supplied
    # benchmark command is what carries a directory, and that is the actionable answer.
    for role, path in (("baseline", args.baseline), ("optimized", args.optimized)):
        if Path(path).is_dir():
            print(
                f"VERDICT: UNSUPPORTED\n"
                f"The built-in executor compiles one source file; {role} is a directory "
                f"({path}). A kernel split across files needs a host-supplied benchmark "
                f"command (external.benchmark)."
            )
            return 1

    baseline_code = Path(args.baseline).read_text()
    optimized_code = Path(args.optimized).read_text()

    executor = KernelBenchExecutor(device=args.device)
    reference_code = Path(reference_path).read_text() if reference_path else None
    # A spec that declares no inputs cannot build tensors; the reference builds them
    # from the variant's dims. A spec with inputs keeps its own path below.
    spec = load_spec(args.spec) if args.spec else None
    if reference_path and (spec is None or not spec.inputs):
        spec_workload = flop = nbytes = None
        if spec is not None:
            variant = spec.resolve_variant(args.variant)
            try:
                spec_workload = spec.get_reference_workload(variant)
            except ValueError as exc:
                print(f"Correctness: FAILED\nError: {exc}")
                return 1
            for name, value in (("rtol", spec.get_rtol(variant)), ("atol", spec.get_atol(variant))):
                if value is not None:
                    setattr(executor, name, value)
            flop, nbytes = spec.get_flop(variant), spec.get_bytes(variant)
            print(f"Variant: {variant}")
        result = executor.compare_reference_workload(
            reference_code,
            baseline_code,
            optimized_code,
            baseline_us=args.baseline_us,
            spec_workload=spec_workload,
        )
        correct = result.original_correct and result.optimized_correct
        print(f"Correctness: {'PASSED' if correct else 'FAILED'}")
        if not correct:
            print(f"Error: {result.feedback_message}")
            return 1
        gpu = args.device.split(":")[0] in ("xpu", "cuda")
        print(
            f"TIMER: {'device_forward_after_untimed_reset' if gpu else 'device_buffer_reset_plus_forward'}"
        )
        print(f"Feedback: {result.feedback_message}")
        print(
            f"Performance: baseline_us={result.original_time_us:.2f}, "
            f"kernel_us={result.optimized_time_us:.2f}, speedup={result.speedup:.2f}x"
        )
        rates = []
        if flop:
            rates += [
                f"baseline_tflops={flop / result.original_time_us / 1e6:.3f}",
                f"kernel_tflops={flop / result.optimized_time_us / 1e6:.3f}",
            ]
        if nbytes:
            rates += [
                f"baseline_gbs={nbytes / result.original_time_us / 1e3:.1f}",
                f"kernel_gbs={nbytes / result.optimized_time_us / 1e3:.1f}",
            ]
        if rates:
            print(f"Throughput: {', '.join(rates)}")
        return 0

    variant = spec.resolve_variant(args.variant)
    input_shapes = spec.get_input_shapes(variant)
    flop = spec.get_flop(variant)
    dtype = spec.get_dtype(variant)
    input_dtypes = spec.get_input_dtypes(variant)
    init_args = spec.get_init_args(variant)

    if reference_code is not None:
        for candidate in (baseline_code, optimized_code):
            if not executor._check_correctness(
                original_code=reference_code,
                optimized_code=candidate,
                kernel_name="Model",
                input_shapes=input_shapes,
                dtype=dtype,
                init_args=init_args,
                input_dtypes=input_dtypes,
            ):
                print("Correctness: FAILED")
                print("Error: baseline or trial differs from the semantic reference")
                return 1

    if args.baseline_us is not None:
        baseline_us = [float(v) for v in str(args.baseline_us).split(",")]
        print(f"Using cached baseline: {baseline_us} us")
        outputs_match = executor._check_correctness(
            original_code=baseline_code,
            optimized_code=optimized_code,
            kernel_name="Model",
            input_shapes=input_shapes,
            dtype=dtype,
            init_args=init_args,
            input_dtypes=input_dtypes,
        )
        if not outputs_match:
            print("Correctness: FAILED")
            print("Error: optimized kernel did not pass reference correctness validation")
            return 1
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
            print("Correctness: PASSED")
            print(
                f"Performance: baseline_us={baseline_ms * 1000:.2f}, "
                f"kernel_us={opt_ms * 1000:.2f}, speedup={speedup:.2f}x"
            )
        else:
            print("Correctness: FAILED")
            print(f"Error: {optimized_result.error_message}")
            return 1
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
        correct = result.original_correct and result.optimized_correct
        print(f"Correctness: {'PASSED' if correct else 'FAILED'}")
        if not result.original_correct or not result.optimized_correct:
            if result.feedback_message:
                print(f"Feedback: {result.feedback_message}")
            return 1
        if result.original_time_us and result.optimized_time_us:
            print(
                f"Performance: baseline_us={result.original_time_us:.2f}, "
                f"kernel_us={result.optimized_time_us:.2f}, speedup={result.speedup:.2f}x"
            )
        if result.feedback_message:
            print(f"Feedback: {result.feedback_message}")
    return 0


def _builtin_opted_in(args) -> bool:
    if getattr(args, "builtin_benchmark", False):
        return True
    return os.getenv(BUILTIN_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def run(args):
    template = _external_template(args)
    if template:
        sys.exit(_run_external(args, template))
    if not _builtin_opted_in(args):
        print("Correctness: FAILED")
        print("VERDICT: NO_BENCHMARK_CONFIGURED")
        print(
            "Error: no host benchmark command (--external-benchmark, EXTERNAL_BENCHMARK, "
            "or external.benchmark in config) and the built-in executor was not asked for "
            f"(--builtin-benchmark or {BUILTIN_ENV}=1). The built-in one times random "
            "tensors at the spec's shapes; it is a measurement of something, and saying "
            "which is the caller's to state."
        )
        sys.exit(1)
    sys.exit(_run_builtin(args))
