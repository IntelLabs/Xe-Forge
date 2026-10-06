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
import shutil
import sys
from pathlib import Path

from xe_forge.config import Config, get_config, override_config
from xe_forge.models import DSL, OptimizationStage

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

    litellm.client_session = httpx.Client()
    litellm.aclient_session = httpx.AsyncClient()
    litellm.ssl_verify = True
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

    # GPU profiling: unitrace (Level Zero EU stall sampling, no GPU OA counter
    # permissions needed) and/or VTune (gpu-offload, needs /dev/dri/card0 access)
    parser.add_argument(
        "--unitrace", action="store_true", default=None, help="Enable unitrace profiling"
    )
    parser.add_argument("--no-unitrace", action="store_true", help="Disable unitrace profiling")
    parser.add_argument("--unitrace-bin", type=str, help="Path to unitrace binary")
    parser.add_argument("--vtune", action="store_true", default=None, help="Enable VTune profiling")
    parser.add_argument("--no-vtune", action="store_true", help="Disable VTune profiling")
    parser.add_argument("--vtune-bin", type=str, help="Path to VTune binary")

    # Claude Code specific
    parser.add_argument("--workspace", type=str, help="Workspace dir for Claude Code engine")
    parser.add_argument("--auto-launch", action="store_true", help="Auto-launch claude CLI")
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

    # Experimental AI lowering (see xe_forge.lowering and docs/VISA_LOWERING.md)
    lower_group = parser.add_argument_group(
        "AI lowering", "Lower a Triton kernel to Intel vISA with an LLM (experimental)"
    )
    lower_group.add_argument("--lower", choices=["visa"], default=None,
                             help="Lower the kernel instead of optimizing it as source; with --stages, "
                                  "source optimization runs first and its result is lowered")
    lower_group.add_argument("--lower-target", choices=["xe2"], default=None, help="Lowering target (default: xe2)")
    lower_group.add_argument("--lower-max-trials", type=int, default=None,
                             help="Attempt budget: finalize-and-verify calls (default: 12)")
    lower_group.add_argument("--lower-optimize", action="store_true", default=None,
                             help="After a correct kernel, continue with performance rounds")
    lower_group.add_argument("--lower-knowledge", default=None,
                             choices=["none", "contracts", "docs", "docs+examples", "docs+corpus", "all"],
                             help="vISA knowledge in the prompt: experimental modes A, B', B, C, D, C+D "
                                  "(default: docs+examples)")
    lower_group.add_argument("--lower-no-compiler-seed", action="store_true", default=None,
                             help="Forbid compiler-generated lowering of the target kernel in any prompt "
                                  "(the default and the only mode implemented)")
    lower_group.add_argument("--lower-allow-compiler-seed", action="store_true",
                             help="Reserved control arm; not implemented")
    lower_group.add_argument("--lower-kernel", default=None,
                             help="Which @triton.jit kernel to lower when the module launches several")
    lower_group.add_argument("--lower-out", default=None, help="Directory for lowering runs (default: ./lowering_runs)")

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
        help="YAML config file for multi-workload tile tuning (see docs/TILE.md)",
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
    if getattr(args, "dataset_record", None):
        os.environ["DATASET_RECORD"] = args.dataset_record
    if getattr(args, "lessons", None):
        os.environ["LESSONS_DIR"] = args.lessons
    if getattr(args, "kernel_repo", None):
        os.environ["KERNEL_REPO"] = args.kernel_repo
    if getattr(args, "compiler_flags", None):
        os.environ["COMPILER_FLAGS"] = args.compiler_flags

    for flag, env in (("lower", "LOWER"), ("lower_target", "LOWER_TARGET"),
                      ("lower_max_trials", "LOWER_MAX_TRIALS"), ("lower_knowledge", "LOWER_KNOWLEDGE"),
                      ("lower_kernel", "LOWER_KERNEL")):
        value = getattr(args, flag, None)
        if value is not None:
            os.environ[env] = str(value)
    if getattr(args, "lower_optimize", None):
        os.environ["LOWER_OPTIMIZE"] = "true"

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
    if config.lowering.target:
        return _run_lower(parser, args, config)
    return _run_optimize(parser, args, config)


def _run_lower(parser, args, config: Config) -> int:
    """Experimental: lower a Triton kernel to Intel vISA (see xe_forge.lowering)."""
    if args.lower_allow_compiler_seed:
        parser.error("--lower-allow-compiler-seed is a reserved control arm and is not implemented")
    if not config.lowering.no_compiler_seed:
        parser.error("only --lower-no-compiler-seed is implemented")
    if not args.input or not Path(args.input).exists():
        parser.error("--lower needs --input pointing at a Triton kernel module")
    if config.device_config.dsl != "triton":
        parser.error("--lower currently lowers Triton kernels only (--dsl triton)")

    module_path = args.input
    if args.stages:
        # Composition: optimize as source first, then lower the optimized kernel.
        out_dir = Path(args.lower_out or "lowering_runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = args.output or str(out_dir / f"{Path(args.input).stem}_source_optimized.py")
        rc = _run_optimize(parser, args, config)
        if rc != 0 or not Path(args.output).exists():
            print("Source optimization produced no kernel; lowering the original instead.")
        else:
            module_path = args.output

    from xe_forge.lowering.pipeline import LoweringPipeline

    lw = config.lowering
    print("=" * 60)
    print("XE-FORGE AI LOWERING (experimental)")
    print("=" * 60)
    print(f"Kernel module: {module_path}")
    print(f"Target: {lw.target} / {lw.lower_target}; knowledge: {lw.knowledge}; budget: {lw.max_trials}")
    print(f"Agent: Claude Code (headless); model: {config.llm.model}")
    pipeline = LoweringPipeline(
        module_path, args.spec, variant=args.variant, kernel=lw.kernel, target=lw.lower_target,
        knowledge=lw.knowledge, max_attempts=lw.max_trials, optimize=lw.optimize,
        char_budget=lw.char_budget, finalizer_igc=lw.finalizer_igc, timeout_s=lw.timeout_s,
        max_turns=config.engine.max_turns, out_dir=args.lower_out, llm=config.llm,
        rtol=args.rtol, atol=args.atol,
    )
    result = pipeline.run()
    r = result.record
    print("=" * 60)
    print(f"Correct: {'yes' if r.correct else 'no'}  attempts: {r.attempts}  stopped: {r.stopped}")
    if r.attempts_to_valid:
        print(f"First correct attempt: #{r.attempts_to_valid} after {r.time_to_valid_s}s")
    if r.best_speedup:
        print(f"Speedup vs Triton: {r.best_speedup:.3f}x")
    print(f"Tokens: {r.tokens}  cost: {r.cost_usd}  turns: {r.num_turns}")
    if r.error:
        print(f"Error: {r.error}")
    print(f"Run directory: {result.run_dir}")
    if result.visa_path:
        print(f"Kernel: {result.visa_path}")
        if args.output and not args.stages:
            shutil.copy(result.visa_path, args.output)
    return 0 if result.success else 1


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
    if cfg.mode == "fa":
        print(f"FA mode: {cfg.fa_mode}, causal: {cfg.causal}, persistent: {cfg.persistent}")
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
    if cfg.mode == "fa":
        strategy = FAStrategy(causal=cfg.causal, mode=cfg.fa_mode, persistent=cfg.persistent)
    else:
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


def _copy_winner_dir(src: Path, out: Path) -> str | None:
    """Replace *out* with a copy of the directory winner *src*; return an error or None."""
    import shutil

    dst = out.parent.resolve() / out.name
    cwd = Path.cwd().resolve()
    if dst == cwd or dst in cwd.parents:
        return f"refusing to replace {out}: it contains the working directory"
    # out is removed before the copy, so it must not overlap the winner.
    src = src.resolve()
    if src == dst or dst in src.parents or src in dst.parents:
        return f"refusing to replace {out}: it is, holds or is inside the winner {src}"
    if out.is_dir() and not out.is_symlink():
        shutil.rmtree(out)
    elif out.exists() or out.is_symlink():
        out.unlink()
    shutil.copytree(src, out)
    return None


def _run_optimize(parser, args, config: Config) -> int:
    """Run the optimization pipeline (original path)."""
    # Validate input. Only the Claude engine can locate a kernel in --kernel-repo at
    # runtime; every other engine would be handed an empty kernel.
    if not args.input and not (args.kernel_repo and config.engine.engine == "claude"):
        parser.error(
            "--input is required (or use --kernel-repo with --engine claude, "
            "--tile-tune / --tune-config)"
        )
    if args.input and not Path(args.input).exists():
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
