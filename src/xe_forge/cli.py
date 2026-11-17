#!/usr/bin/env python3
"""
CLI for Xe-Forge kernel optimization pipeline

Usage:
    python -m xe_forge.cli --input kernel.py --name my_kernel
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from xe_forge.config import Config, get_config, override_config
from xe_forge.models import DSL, OptimizationStage

logger = logging.getLogger(__name__)


def _setup_dspy(config: Config) -> None:
    """Configure DSPy LM from the shared config. Used by all LLM-driven paths.

    Supports OpenAI-compatible endpoints and AWS Bedrock (model ids prefixed
    with ``bedrock/``). See :mod:`xe_forge.llm_setup`.
    """
    from xe_forge.llm_setup import configure_dspy

    configure_dspy(config.llm)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Xe-Forge - Multi-stage kernel optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a Triton kernel for Intel XPU (default)
  python -m xe_forge.cli --input kernel.py --name gemm_kernel

  # Optimize for CUDA
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --device cuda --dsl triton

        """,
    )

    # Input
    parser.add_argument("--input", "-i", type=str, help="Input kernel file")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Explicit path to the semantic reference (e.g. PyTorch) this kernel must "
        "match; overrides the sibling <name>_pytorch.py lookup next to --input",
    )
    parser.add_argument(
        "--name", "-n", type=str, default="kernel", help="Kernel function name (default: kernel)"
    )

    # Output
    parser.add_argument("--output", "-o", type=str, help="Output file for optimized kernel")

    # Spec file for testing
    parser.add_argument(
        "--spec", "-s", type=str, help="YAML spec file for test configuration (KernelBench format)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant name from spec (default: bench-gpu, overridden by default_variant in spec)",
    )

    # Stage selection
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated stages to apply (e.g., dtype_fix,fusion,device_specific)",
    )

    # LLM configuration
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--api-base", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")

    # Device and DSL configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["xpu", "cuda", "cpu"],
        default=None,
        help="Target device (default: xpu)",
    )
    parser.add_argument(
        "--dsl",
        type=str,
        choices=["triton", "gluon", "sycl", "cuda", "opencl"],
        default=None,
        help="Kernel DSL (default: triton)",
    )

    # GPU tuning configuration
    parser.add_argument("--num-warps", type=int, help="Default number of warps")
    parser.add_argument("--tile-size", type=int, help="Preferred tile size (M=N)")

    # Dtype options
    parser.add_argument(
        "--target-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Target dtype for kernel optimization (e.g., float16, bfloat16)",
    )

    # Correctness options
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip correctness validation (performance-only mode)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance override for correctness check (overrides spec and config values)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance override for correctness check (overrides spec and config values)",
    )

    # Engine selection
    parser.add_argument(
        "--engine",
        type=str,
        choices=["dspy", "claude", "codex", "dspy-agent"],
        default=None,
        help="Optimization engine (default: dspy). claude and codex drive the agent "
        "workspace with that CLI; dspy-agent runs a DSPy tool-calling agent in it",
    )
    parser.add_argument(
        "--target",
        choices=["kernel", "library"],
        default=None,
        help="Workspace engines: optimize a kernel (default) or tune how the repository "
        "calls a library such as oneDNN",
    )

    # Trial management
    parser.add_argument("--max-trials", type=int, help="Max optimization trials (default: 10)")
    parser.add_argument("--trials-dir", type=str, help="Directory for trial state")
    parser.add_argument("--no-trials", action="store_true", help="Disable trial tracking")

    # GPU profiling: unitrace (device timing, ComputeBasic counters and EU stall
    # sampling) and/or VTune (gpu-offload, needs /dev/dri/card0 access)
    parser.add_argument(
        "--unitrace", action="store_true", default=None, help="Enable unitrace profiling"
    )
    parser.add_argument("--no-unitrace", action="store_true", help="Disable unitrace profiling")
    parser.add_argument("--unitrace-bin", type=str, help="Path to unitrace binary")
    parser.add_argument("--vtune", action="store_true", default=None, help="Enable VTune profiling")
    parser.add_argument("--no-vtune", action="store_true", help="Disable VTune profiling")
    parser.add_argument("--vtune-bin", type=str, help="Path to VTune binary")

    # Claude Code specific
    parser.add_argument(
        "--workspace", type=str, help="Workspace dir of the claude, codex and dspy-agent engines"
    )
    parser.add_argument(
        "--auto-launch",
        action="store_true",
        help="Run the agent in the workspace headless (claude, codex; dspy-agent always does)",
    )
    parser.add_argument(
        "--max-turns", type=int, help="Turn limit for one headless claude session (default: 80)"
    )
    parser.add_argument(
        "--compiler-flags",
        type=str,
        default=None,
        help="Extra flags for the target compiler (e.g. icpx for SYCL); rendered into "
        "the generated workspace for Claude's own build step, not consumed by xe_forge",
    )

    # Host-supplied toolchain and measurement (see xe_forge.core.build_backend
    # and xe_forge.core.external). Unset, all three keep the built-in ai_bench build and measurement.
    host_group = parser.add_argument_group(
        "host integration",
        "Delegate building, correctness and timing to a project that owns them",
    )
    host_group.add_argument(
        "--build-backend",
        type=str,
        help="Build backend: a registered name, an entry point, or 'module:attr'",
    )
    host_group.add_argument(
        "--external-benchmark",
        type=str,
        help="Command template run instead of the built-in benchmark; placeholders: "
        "{kernel} {baseline} {trial} {spec} {variant} {device} {dsl} {workspace}",
    )
    host_group.add_argument(
        "--external-validate",
        type=str,
        help="Command template run instead of the built-in validator; placeholders: "
        "{kernel} {trial} {dsl} {stage} {workspace}",
    )
    host_group.add_argument(
        "--stack-python",
        type=str,
        help="Interpreter of the stack the kernel serves; the port-back agent then builds its "
        "port into that stack (beside it, never into it), tests it there and records the verdict",
    )
    host_group.add_argument(
        "--dataset-record",
        type=str,
        help="JSON record of the dataset behind the spec's benchmark variants; its "
        "facts are rendered into the workspace so the session can inspect the real "
        "workload distribution (see xe_forge.core.dataset_record)",
    )
    host_group.add_argument(
        "--lessons",
        type=str,
        metavar="DIR",
        help="Directory, outside the workspace, where sessions record what they "
        "measured on this part; a session reads every file in it and appends to "
        "its own (see xe_forge.core.lessons)",
    )
    host_group.add_argument(
        "--kernel-repo",
        type=str,
        metavar="PATH",
        help="Path to an external repository holding a kernel this workspace did not "
        "write; the kernel-locator agent explores it once and writes findings before "
        "the first trial",
    )
    parser.add_argument(
        "--integration-repo",
        action="append",
        default=None,
        metavar="PATH",
        help="With --kernel-repo: a repository the kernel is wired into (e.g. vLLM). A winner "
        "may then change the op's interface and call sites, ported as a serving-only patch "
        "to each repository. Repeatable",
    )

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser, parser.parse_args()


def _load_config(args) -> Config:
    """Shared config loading: env vars, overrides, dotenv — used by all paths."""
    # Build configuration overrides
    overrides = {}

    if args.model:
        overrides["llm_model"] = args.model
    if args.api_base:
        overrides["llm_api_base"] = args.api_base
    if args.api_key:
        overrides["llm_api_key"] = args.api_key
    if args.num_warps:
        overrides["device_config_default_num_warps"] = args.num_warps
    if args.tile_size:
        overrides["device_config_preferred_tile_m"] = args.tile_size
        overrides["device_config_preferred_tile_n"] = args.tile_size
    if args.target_dtype:
        overrides["optimization_target_dtype"] = args.target_dtype
    if args.debug:
        overrides["logging_level"] = "DEBUG"
    if args.no_correctness:
        overrides["optimization_require_correctness"] = False

    # Set device/dsl env vars before config loading
    if args.device:
        os.environ["DEVICE_TYPE"] = args.device
    if args.dsl:
        os.environ["DSL"] = args.dsl

    # Engine/trial/profiler env var overrides
    if args.engine:
        os.environ["ENGINE"] = args.engine
    if getattr(args, "target", None):
        os.environ["TARGET"] = args.target
    if args.max_trials is not None:
        os.environ["MAX_TRIALS"] = str(args.max_trials)
    if args.trials_dir:
        os.environ["TRIALS_DIR"] = args.trials_dir
    if args.no_trials:
        os.environ["TRIALS_ENABLED"] = "false"
    if args.vtune:
        os.environ["VTUNE_ENABLED"] = "true"
    if args.no_vtune:
        os.environ["VTUNE_ENABLED"] = "false"
    if args.vtune_bin:
        os.environ["VTUNE_BIN"] = args.vtune_bin
    if args.unitrace:
        os.environ["UNITRACE_ENABLED"] = "true"
    if args.no_unitrace:
        os.environ["UNITRACE_ENABLED"] = "false"
    if args.unitrace_bin:
        os.environ["UNITRACE_BIN"] = args.unitrace_bin
    if args.workspace:
        os.environ["WORKSPACE"] = args.workspace
        os.environ.setdefault(
            "TORCH_EXTENSIONS_DIR", str(Path(args.workspace).resolve() / ".torch_extensions")
        )
    if args.auto_launch:
        os.environ["AUTO_LAUNCH"] = "true"
    if getattr(args, "max_turns", None) is not None:
        os.environ["MAX_TURNS"] = str(args.max_turns)
    if getattr(args, "build_backend", None):
        os.environ["BUILD_BACKEND"] = args.build_backend
    if getattr(args, "external_benchmark", None):
        os.environ["EXTERNAL_BENCHMARK"] = args.external_benchmark
    if getattr(args, "external_validate", None):
        os.environ["EXTERNAL_VALIDATE"] = args.external_validate
    if getattr(args, "stack_python", None):
        os.environ["STACK_PYTHON"] = args.stack_python
    if getattr(args, "dataset_record", None):
        os.environ["DATASET_RECORD"] = args.dataset_record
    if getattr(args, "lessons", None):
        os.environ["LESSONS_DIR"] = args.lessons
    if getattr(args, "kernel_repo", None):
        os.environ["KERNEL_REPO"] = args.kernel_repo
    if getattr(args, "integration_repo", None):
        os.environ["INTEGRATION_REPOS"] = os.pathsep.join(args.integration_repo)
    if getattr(args, "compiler_flags", None):
        os.environ["COMPILER_FLAGS"] = args.compiler_flags

    config = get_config()
    if overrides:
        config = override_config(**overrides)

    return config


def main():
    sys.stdout.reconfigure(line_buffering=True)
    from xe_forge import load_env

    load_env()  # before the modules that read the environment when imported

    from xe_forge.frontend.cli import COMMANDS

    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from xe_forge.frontend.cli import main as frontend_main

        return frontend_main(sys.argv[1:])

    parser, args = _parse_args()

    # ── Shared setup: config + LLM ──────────────────────────────
    config = _load_config(args)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return _run_optimize(parser, args, config)


def _run_optimize(parser, args, config: Config) -> int:
    """Run the optimization pipeline (original path)."""
    # Validate input. Only the workspace engines can locate a kernel in --kernel-repo
    # at runtime; the pipeline would be handed an empty kernel.
    workspace_engine = config.engine.engine in ("claude", "codex", "dspy-agent")
    if not args.input and not (args.kernel_repo and workspace_engine):
        parser.error(
            "--input is required (or use --kernel-repo with --engine claude|codex|dspy-agent)"
        )
    if args.input and not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    dsl = config.device_config.dsl

    # Parse stages
    stages = None
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]
        stages = []
        for name in stage_names:
            try:
                stages.append(OptimizationStage(name))
            except ValueError:
                print(f"Warning: Unknown stage '{name}', skipping")

    # Print header
    print("=" * 60)
    print("XE-FORGE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Kernel: {args.name}")
    print(f"Device: {config.device_config.device}")
    print(f"DSL: {config.device_config.dsl}")
    print(f"Model: {config.llm.model}")
    if args.spec:
        variant_display = args.variant or "(auto-resolved from spec)"
        print(f"Spec: {args.spec} (variant: {variant_display})")
    if args.target_dtype:
        print(f"Target dtype: {args.target_dtype}")
    print(f"Stages: {[s.value for s in stages] if stages else 'all'}")

    # Print correctness settings
    if config.optimization.require_correctness:
        tol_source = []
        if args.rtol is not None:
            tol_source.append(f"rtol={args.rtol} (CLI)")
        if args.atol is not None:
            tol_source.append(f"atol={args.atol} (CLI)")
        if tol_source:
            print(f"Correctness: enabled, {', '.join(tol_source)}")
        elif args.spec:
            print("Correctness: enabled (tolerances from spec, fallback to config defaults)")
        else:
            print(
                f"Correctness: enabled (rtol={config.optimization.correctness_rtol}, atol={config.optimization.correctness_atol})"
            )
    else:
        print("Correctness: disabled (performance-only)")

    print("=" * 60)

    # Load spec if provided
    input_shapes = None
    flop = None
    if args.spec:
        from xe_forge.core import load_spec

        spec = load_spec(args.spec)
        # SYCL's family default is bench-xpu, after the spec's own default_variant.
        args.variant = spec.resolve_variant(
            args.variant, "bench-xpu" if dsl == "sycl" else "bench-gpu"
        )
        input_shapes = spec.get_input_shapes(args.variant)
        flop = spec.get_flop(args.variant)
        spec_rtol = spec.get_rtol(args.variant)
        spec_atol = spec.get_atol(args.variant)
        print("\nTest configuration from spec:")
        print(f"  Variant: {args.variant}")
        print(f"  Input shapes: {input_shapes}")
        print(f"  FLOP: {flop:,.0f}" if flop else "  FLOP: N/A")
        if spec_rtol is not None or spec_atol is not None:
            print(f"  Spec tolerances: rtol={spec_rtol}, atol={spec_atol}")
        print()

    # Create executor if spec provided (let pipeline auto-create for SYCL/CUDA)
    executor = None
    if args.spec and dsl not in ("sycl", "cuda"):
        from xe_forge.core import KernelBenchExecutor

        executor = KernelBenchExecutor(
            device=config.device_config.device,
            require_correctness=config.optimization.require_correctness,
            rtol=config.optimization.correctness_rtol,
            atol=config.optimization.correctness_atol,
        )
        print(f"Executor: KernelBenchExecutor (device={config.device_config.device})")

    try:
        target_ext = DSL(str(dsl)).kernel_ext
    except ValueError:
        target_ext = ".py"

    kernel_code = ""
    reference_code = ""
    if args.input:
        input_path = Path(args.input)
        with open(input_path) as f:
            input_code = f.read()

        # The input is a baseline to optimize only if it is already written in the target
        # language; a file in another language cannot compile as one, so it is instead the
        # semantic reference, and Claude writes the target file itself.
        is_target_language = input_path.suffix == target_ext
        kernel_code = input_code if is_target_language else ""

        if args.reference:
            try:
                with open(args.reference) as f:
                    reference_code = f.read()
            except FileNotFoundError:
                parser.error(f"--reference file not found: {args.reference}")
        elif is_target_language:
            sibling = input_path.with_name(f"{input_path.stem}_pytorch.py")
            try:
                with open(sibling) as f:
                    reference_code = f.read()
            except FileNotFoundError:
                print(f"No PyTorch reference file found at {sibling}")
        else:
            reference_code = input_code
    elif args.reference:
        # No --input at all (e.g. --kernel-repo): the reference can still be named
        # explicitly even with no local baseline file to optimize yet.
        try:
            with open(args.reference) as f:
                reference_code = f.read()
        except FileNotFoundError:
            parser.error(f"--reference file not found: {args.reference}")

    # Create engine and optimize
    from xe_forge.engines import create_engine

    engine = create_engine(config)
    engine_name = config.engine.engine
    print(f"Engine: {engine_name}")
    if config.trial.enabled:
        print(f"Trials: enabled (max={config.trial.max_trials}, dir={config.trial.trials_dir})")
    if config.profiler.unitrace_enabled:
        print(f"Unitrace: enabled (bin={config.profiler.unitrace_bin})")
    if config.profiler.vtune_enabled:
        print(f"VTune: enabled (bin={config.profiler.vtune_bin})")

    # For DSPy engine, pass executor if available
    if engine_name == "dspy" and executor is not None:
        engine.executor = executor

    result = engine.optimize(
        kernel_code=kernel_code,
        reference_code=reference_code,
        kernel_name=args.name if args.name != "kernel" else None,
        input_shapes=input_shapes,
        stages=stages,
        spec_path=args.spec,
        variant_type=args.variant,
        target_dtype=args.target_dtype,
        rtol=args.rtol,
        atol=args.atol,
    )

    # Save output if requested
    if args.output and result.optimized_code:
        with open(args.output, "w") as f:
            f.write(result.optimized_code)
    elif args.output and result.optimized_path and Path(result.optimized_path).is_dir():
        error = _copy_winner_dir(Path(result.optimized_path), Path(args.output))
        if error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {'✓' if result.success else '✗'}")

    if result.analysis:
        print(f"Issues found: {len(result.analysis.detected_issues)}")

    print(f"Stages applied: {len(result.stages_applied)}")
    for stage_result in result.stages_applied:
        status = "✓" if stage_result.success else "✗"
        print(f"  {status} {stage_result.stage.value}")
        if stage_result.speedup:
            print(f"      Speedup: {stage_result.speedup:.2f}x")
        if stage_result.changes_made:
            for change in stage_result.changes_made[:3]:
                print(f"      - {change}")

    if result.total_speedup:
        print(f"\nTotal Speedup: {result.total_speedup:.2f}x")

    if result.original_tflops and result.optimized_tflops:
        print(f"Performance: {result.original_tflops:.2f} → {result.optimized_tflops:.2f} TFLOPS")

    if result.original_ms and result.optimized_ms:
        print(f"Execution Time: {result.original_ms:.3f} ms → {result.optimized_ms:.3f} ms")

    if args.output:
        print(f"\nOptimized kernel saved to: {args.output}")

    print("=" * 60)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
