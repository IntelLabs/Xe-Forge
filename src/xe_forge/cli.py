#!/usr/bin/env python3
"""
CLI for Xe-Forge kernel optimization pipeline

Usage:
    python -m xe_forge.cli --input kernel.py --name my_kernel
    python -m xe_forge.cli --dsl sycl --tune-config tune_gemm_shapes.yaml
    python -m xe_forge.cli --dsl sycl --tile-tune --m 4096 --gemm-n 4096 --k 4096
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from xe_forge.config import Config, get_config, override_config
from xe_forge.models import OptimizationStage

logger = logging.getLogger(__name__)


def _setup_dspy(config: Config) -> None:
    """Configure DSPy LM from the shared config. Used by all LLM-driven paths."""
    import dspy
    import httpx
    import litellm

    if config.llm.api_base:
        os.environ["OPENAI_API_BASE"] = config.llm.api_base
    if config.llm.api_key:
        os.environ["OPENAI_API_KEY"] = config.llm.api_key

    litellm.client_session = httpx.Client(verify=False)
    lm = dspy.LM(
        model=config.llm.model,
        api_base=config.llm.api_base,
        model_type="responses",
        api_key=config.llm.api_key or "",
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        cache=False,
    )
    dspy.configure(lm=lm, warn_on_type_mismatch=False)


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

  # Tile tuning (single GEMM shape)
  python -m xe_forge.cli --dsl sycl --tile-tune --m 8192 --gemm-n 4096 --k 4096

  # Tile tuning (YAML config with multiple workloads)
  python -m xe_forge.cli --dsl sycl --tune-config tune_gemm_shapes.yaml
        """,
    )

    # Input (required for optimization pipeline, not for tile tuning)
    parser.add_argument("--input", "-i", type=str, help="Input kernel file")
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

    # Tile tuning options
    tune_group = parser.add_argument_group("tile tuning", "Tile search / auto-tuning options")
    tune_group.add_argument(
        "--tile-tune",
        action="store_true",
        help="Run tile tuning for a single GEMM shape (requires --m, --gemm-n, --k)",
    )
    tune_group.add_argument(
        "--tune-config",
        type=str,
        help="YAML config file for multi-workload tile tuning (see TILE.md)",
    )
    tune_group.add_argument("--m", type=int, default=4096, help="GEMM M dimension")
    tune_group.add_argument("--gemm-n", type=int, default=4096, help="GEMM N dimension")
    tune_group.add_argument("--k", type=int, default=4096, help="GEMM K dimension")
    tune_group.add_argument("--max-rounds", type=int, default=5, help="Max LLM proposal rounds")
    tune_group.add_argument(
        "--gemm-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "f16", "tf32", "f32", "int8"],
        help="Data type for tile tuning (default: bf16)",
    )
    tune_group.add_argument(
        "--tune-output",
        type=str,
        default="tile_tuning_results.json",
        help="Output JSON file for tuning results",
    )

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
    if args.best_k:
        overrides["optimization_best_k"] = args.best_k
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

    config = get_config()
    if overrides:
        config = override_config(**overrides)

    return config


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser, args = _parse_args()

    # ── Shared setup: config + LLM ──────────────────────────────
    config = _load_config(args)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # ── Route: tile tuning or optimization pipeline ─────────────
    if args.tune_config:
        return _run_tune_config(args, config)
    if args.tile_tune:
        return _run_tile_tune(args, config)
    return _run_optimize(parser, args, config)


def _run_tile_tune(args, config: Config) -> int:
    """Run tile tuning for a single GEMM shape."""
    from xe_forge.core.sycl_executor import KernelType, SyclExecutor
    from xe_forge.core.tile_search import GEMMStrategy, TileTuningAgent, export_results_json

    _setup_dspy(config)

    print("=" * 60)
    print("XE-FORGE TILE TUNING")
    print("=" * 60)
    print("Mode: GEMM single shape")
    print(f"Shape: M={args.m}, N={args.gemm_n}, K={args.k}")
    print(f"Dtype: {args.gemm_dtype}")
    print(f"Model: {config.llm.model}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Output: {args.tune_output}")
    print("=" * 60)

    executor = SyclExecutor(kernel_type=KernelType.GEMM, verify=False)
    strategy = GEMMStrategy()
    agent = TileTuningAgent(executor, strategy, dtype=args.gemm_dtype)

    workload = {"M": args.m, "N": args.gemm_n, "K": args.k}
    result = agent.tune(workload, max_rounds=args.max_rounds)

    output_data = {
        "workloads": [json.loads(result.model_dump_json())],
        "tile_shapes": json.loads(export_results_json([result])),
    }
    with open(args.tune_output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.tune_output}")

    if result.best_tflops:
        print(
            f"Best: {result.best_config.wg} -> {result.best_tflops:.2f} TFLOPS ({result.best_time_ms:.4f} ms)"
        )
    else:
        print("No successful configuration found.")

    return 0


def _run_tune_config(args, config: Config) -> int:
    """Run tile tuning from a YAML config file."""
    from xe_forge.core.sycl_executor import KernelType, SyclExecutor
    from xe_forge.core.tile_search import (
        FAStrategy,
        GEMMStrategy,
        GroupedGEMMStrategy,
        MoEGEMMStrategy,
        TileTuningAgent,
        export_results_json,
        load_tune_config,
    )

    _setup_dspy(config)

    cfg = load_tune_config(args.tune_config)

    print("=" * 60)
    print("XE-FORGE TILE TUNING")
    print("=" * 60)
    print(f"Config: {args.tune_config}")
    print(f"Mode: {cfg.mode}")
    print(f"Dtype: {cfg.dtype}")
    print(f"Model: {config.llm.model}")
    print(f"Max rounds: {cfg.max_rounds}")
    print(f"Output: {cfg.output}")
    print(f"Workloads: {len(cfg.workloads)}")
    for w in cfg.workloads:
        print(f"  - {w.get('name', '?')}: {w}")
    print("=" * 60)

    mode_to_kernel_type = {
        "fa": KernelType.FA,
        "gemm": KernelType.GEMM,
        "grouped_gemm": KernelType.GROUPED_GEMM,
        "moe": KernelType.MOE_GEMM,
    }
    mode_to_strategy = {
        "fa": FAStrategy,
        "gemm": GEMMStrategy,
        "grouped_gemm": GroupedGEMMStrategy,
        "moe": MoEGEMMStrategy,
    }
    kernel_type = mode_to_kernel_type.get(cfg.mode, KernelType.GEMM)
    executor = SyclExecutor(kernel_type=kernel_type, verify=False)
    strategy = mode_to_strategy.get(cfg.mode, GEMMStrategy)()

    agent = TileTuningAgent(executor, strategy, dtype=cfg.dtype)
    all_results = []

    for i, wl in enumerate(cfg.workloads):
        wl = dict(wl)  # copy so pop doesn't mutate config
        name = wl.pop("name", f"workload_{i}")
        print(f"\n{'─' * 40}")
        print(f"Tuning: {name} ({i + 1}/{len(cfg.workloads)})")
        print(f"{'─' * 40}")
        result = agent.tune(wl, max_rounds=cfg.max_rounds)
        all_results.append(result)
        if result.best_tflops:
            print(f"  Best: {result.best_config.wg} -> {result.best_tflops:.2f} TFLOPS")
        else:
            print("  No successful configuration found.")

    output_data = {
        "workloads": [json.loads(r.model_dump_json()) for r in all_results],
        "tile_shapes": json.loads(export_results_json(all_results)),
    }
    output_path = cfg.output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 60}")

    return 0


def _run_optimize(parser, args, config: Config) -> int:
    """Run the optimization pipeline (original path)."""
    # Validate input (required for optimization pipeline)
    if not args.input:
        parser.error("--input is required (or use --tile-tune / --tune-config)")
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

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
