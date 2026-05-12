#!/usr/bin/env python3
"""
CLI for Xe-Forge kernel optimization pipeline

Usage:
    python -m xe_forge.cli --input kernel.py --name my_kernel
"""

import argparse
import os
import sys
from pathlib import Path

from xe_forge.config import get_config, override_config
from xe_forge.models import OptimizationStage


def main():
    sys.stdout.reconfigure(line_buffering=True)

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

  # Optimize with specific stages
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --stages dtype_fix,fusion,device_specific

  # Override model
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --model openai/gpt-4o

  # Optimize with target dtype
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --target-dtype float16

  # Skip correctness checking (performance-only)
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --no-correctness

  # Use more lenient tolerances (overrides spec values)
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --rtol 0.1 --atol 1e-3
        """,
    )

    # Required arguments
    parser.add_argument("--input", "-i", type=str, required=True, help="Input kernel file")
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
        help="Comma-separated stages to apply (e.g., dtype_fix,fusion,xpu_specific)",
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
        choices=["triton", "gluon", "sycl", "cuda"],
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

    parser.add_argument(
        "--best-k",
        type=int,
        help="Number of candidates to evaluate (default: 1)",
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
        choices=["dspy", "claude"],
        default=None,
        help="Optimization engine (default: dspy)",
    )

    # Trial management
    parser.add_argument("--max-trials", type=int, help="Max optimization trials (default: 10)")
    parser.add_argument("--trials-dir", type=str, help="Directory for trial state")
    parser.add_argument("--no-trials", action="store_true", help="Disable trial tracking")

    # VTune profiling
    parser.add_argument("--vtune", action="store_true", default=None, help="Enable VTune profiling")
    parser.add_argument("--no-vtune", action="store_true", help="Disable VTune profiling")
    parser.add_argument("--vtune-bin", type=str, help="Path to VTune binary")

    # Claude Code specific
    parser.add_argument("--workspace", type=str, help="Workspace dir for Claude Code engine")
    parser.add_argument("--auto-launch", action="store_true", help="Auto-launch claude CLI")

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

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
    if args.best_k:
        overrides["optimization_best_k"] = args.best_k
    if args.debug:
        overrides["logging_level"] = "DEBUG"

    # Correctness overrides
    if args.no_correctness:
        overrides["optimization_require_correctness"] = False
    # Note: rtol/atol are passed directly to pipeline.optimize() for
    # proper priority resolution (CLI > spec > config default), not
    # via config overrides.

    # Set device/dsl env vars before config loading
    if args.device:
        os.environ["DEVICE_TYPE"] = args.device
    if args.dsl:
        os.environ["DSL"] = args.dsl

    # Engine/trial/profiler env var overrides
    if args.engine:
        os.environ["ENGINE"] = args.engine
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
    if args.workspace:
        os.environ["WORKSPACE"] = args.workspace
    if args.auto_launch:
        os.environ["AUTO_LAUNCH"] = "true"

    # Load and override config
    config = get_config()
    if overrides:
        config = override_config(**overrides)

    dsl = config.device_config.dsl

    # Default to bench-xpu variant for SYCL
    if args.variant is None and dsl in ("sycl",):
        args.variant = "bench-xpu"

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
    print(f"Best@k: {config.optimization.best_k}")

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
        args.variant = spec.resolve_variant(args.variant)
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

    # Read input file
    with open(args.input) as f:
        kernel_code = f.read()

    # Read reference implementation (Python DSLs only)
    reference_code = ""
    if dsl not in ("sycl", "cuda"):
        try:
            with open(f"{os.path.splitext(args.input)[0]}_pytorch.py") as f:
                reference_code = f.read()
        except FileNotFoundError:
            print(
                f"No PyTorch reference file found at {os.path.splitext(args.input)[0]}_pytorch.py"
            )

    # Create engine and optimize
    from xe_forge.engines import create_engine

    engine = create_engine(config)
    engine_name = config.engine.engine
    print(f"Engine: {engine_name}")
    if config.trial.enabled:
        print(f"Trials: enabled (max={config.trial.max_trials}, dir={config.trial.trials_dir})")
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
