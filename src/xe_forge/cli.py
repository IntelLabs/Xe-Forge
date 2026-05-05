#!/usr/bin/env python3
"""
CLI for Triton Optimizer

Usage:
    python -m xe_forge.cli --input kernel.py --name my_kernel
"""

import argparse
import os
import sys
from pathlib import Path

from xe_forge.config import get_config, override_config
from xe_forge.models import OptimizationStage
from xe_forge.pipeline import XeForgePipeline


def _looks_like_pytorch(source: str) -> bool:
    """Heuristic: Triton kernels import triton and use @triton.jit.

    If neither appears, assume the file is a PyTorch baseline and the user
    wants us to generate a Triton kernel for it.
    """
    if "@triton.jit" in source:
        return False
    if "import triton" in source or "from triton" in source:
        return False
    return True


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Triton Optimizer - Multi-stage optimization for Intel XPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a Triton kernel
  python -m xe_forge.cli --input kernel.py --name gemm_kernel

  # Optimize with specific stages
  python -m xe_forge.cli --input kernel.py --name kernel \\
      --stages dtype_fix,fusion,xpu_specific

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
    parser.add_argument("--input", "-i", type=str, required=True, help="Input Triton kernel file")
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

    # XPU configuration
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

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # Trial-tree exploration overrides (all default to values in .env / config).
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Trial search strategy: tree_walk (default), best_first, beam, mcts",
    )
    parser.add_argument(
        "--writer",
        type=str,
        default=None,
        help="Trial writer: stage_sequence (default), cover, react, explorer",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Max trials in the tree (overrides TRIAL_MAX_TRIALS)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Treat --input as PyTorch and synthesize the initial Triton kernel "
        "via GeneratorAgent (auto-detected when the file has no @triton.jit)",
    )

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
        overrides["xpu_default_num_warps"] = args.num_warps
    if args.tile_size:
        overrides["xpu_preferred_tile_m"] = args.tile_size
        overrides["xpu_preferred_tile_n"] = args.tile_size
    if args.target_dtype:
        overrides["optimization_target_dtype"] = args.target_dtype
    if args.best_k:
        overrides["optimization_best_k"] = args.best_k
    if args.debug:
        overrides["logging_level"] = "DEBUG"
    if args.search:
        overrides["trial_search"] = args.search
    if args.writer:
        overrides["trial_writer"] = args.writer
    if args.max_trials is not None:
        overrides["trial_max_trials"] = args.max_trials

    # Correctness overrides
    if args.no_correctness:
        overrides["optimization_require_correctness"] = False
    # Note: rtol/atol are passed directly to pipeline.optimize() for
    # proper priority resolution (CLI > spec > config default), not
    # via config overrides.

    # Load and override config
    config = get_config()
    if overrides:
        config = override_config(**overrides)

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
    print("TRITON OPTIMIZER")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Kernel: {args.name}")
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

    # Create executor if spec provided
    executor = None
    if args.spec:
        from xe_forge.core import KernelBenchExecutor

        executor = KernelBenchExecutor(
            device=config.xpu.device,
            require_correctness=config.optimization.require_correctness,
            rtol=config.optimization.correctness_rtol,
            atol=config.optimization.correctness_atol,
        )
        print(f"Executor: KernelBenchExecutor (device={config.xpu.device})")

    # Create pipeline and optimize
    pipeline = XeForgePipeline(config=config, executor=executor)  # type: ignore

    # Read input file
    with open(args.input) as f:
        input_code = f.read()

    # Decide whether the input is a PyTorch baseline (needs generation) or
    # an already-written Triton kernel.
    is_pytorch_input = args.generate or _looks_like_pytorch(input_code)

    triton_code: str = ""
    pytorch_code: str = ""
    pytorch_path: str | None = None

    if is_pytorch_input:
        pytorch_code = input_code
        pytorch_path = args.input
        print(f"Detected PyTorch input — will generate initial Triton kernel via GeneratorAgent.")
    else:
        triton_code = input_code
        # Pick up the _pytorch.py sibling if present (used by the analyzer
        # for the algorithmic stage).
        sibling = f"{os.path.splitext(args.input)[0]}_pytorch.py"
        if os.path.exists(sibling):
            with open(sibling) as f:
                pytorch_code = f.read()
            pytorch_path = sibling
        else:
            print(f"No PyTorch reference file found at {sibling}")

    result = pipeline.optimize(
        triton_code=triton_code,
        pytorch_code=pytorch_code,
        pytorch_path=pytorch_path,
        kernel_name=args.name if args.name != "kernel" else None,
        input_shapes=input_shapes,
        stages=stages,
        spec_path=args.spec,
        variant_type=args.variant,
        target_dtype=args.target_dtype,
        rtol=args.rtol,
        atol=args.atol,
        output_path=args.output,
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
