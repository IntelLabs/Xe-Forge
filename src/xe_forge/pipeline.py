"""
Triton Optimizer Pipeline - Multi-stage optimization for Intel XPU

Stage ordering:
1. Analysis           - Identify all issues (LLM-based, no local KB)
2. Algorithmic        - Mathematical / algorithmic rewrites (NEW)
3. Dtype Fix          - Fix float64 to float32
4. Fusion             - Fuse multiple kernels
5. Memory Access      - Coalescing, TMA, register pressure
6. Block Pointers     - Convert to block pointer API
7. Persistent Kernel  - Persistent kernel transforms
8. XPU Specific       - Intel XPU optimizations

Rationale: algorithmic first (may eliminate work), then dtype, then fusion
(on cleaned ops), then low-level tuning on final fused kernel.

Each stage uses CoVeR. The pipeline builds the issue list and passes it
to each stage; the LLM uses its own knowledge to optimize.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import dspy
import httpx
import litellm

from xe_forge.agents import AnalyzerAgent, Optimizer, OptimizerAgent, OptimizerReActAgent
from xe_forge.config import Config, get_config
from xe_forge.core import get_xpu_config_for_pipeline
from xe_forge.models import (
    OptimizationResult,
    OptimizationStage,
)

logger = logging.getLogger(__name__)

DEFAULT_STAGE_ORDER: list[OptimizationStage] = [
    OptimizationStage.ANALYSIS,
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.DTYPE_FIX,
    OptimizationStage.FUSION,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.BLOCK_POINTERS,
    OptimizationStage.PERSISTENT_KERNEL,
    OptimizationStage.XPU_SPECIFIC,
    OptimizationStage.AUTOTUNING,
]


class XeForgePipeline:
    """Multi-stage optimization pipeline for Triton kernels targeting Intel XPU.
    Uses LLM knowledge (no local knowledge base required)."""

    config: Config
    analyzer: AnalyzerAgent
    optimizer: Optimizer

    def __init__(self, config=None, executor=None, validator=None):
        self.config = config or get_config()
        self._setup_logging()
        self._setup_llm()

        if executor is None:
            from xe_forge.core import KernelBenchExecutor

            executor = KernelBenchExecutor(
                device=self.config.xpu.device,
                require_correctness=self.config.optimization.require_correctness,
                rtol=self.config.optimization.correctness_rtol,
                atol=self.config.optimization.correctness_atol,
            )

        # No knowledge base needed
        self.analyzer = AnalyzerAgent()

        match self.config.agent.strategy:
            case "cover":
                Agent = OptimizerAgent
            case "react":
                Agent = OptimizerReActAgent
            case _:
                Agent = OptimizerAgent

        self.optimizer = Agent(
            executor=executor,
            validator=validator,
            max_iterations=self.config.agent.max_iterations,
        )
        self.executor = executor
        self.validator = validator

        logger.info("XeForgePipeline initialized (LLM-knowledge mode)")
        logger.info(f"  LLM: {self.config.llm.model}")
        logger.info(
            f"  Agent: {self.config.agent.strategy} (max_iters={self.config.agent.max_iterations})"
        )

    def _setup_logging(self):
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        Path(self.config.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging.kernel_dir).mkdir(parents=True, exist_ok=True)

    def _setup_llm(self):
        if self.config.llm.api_base:
            os.environ["OPENAI_API_BASE"] = self.config.llm.api_base
        if self.config.llm.api_key:
            os.environ["OPENAI_API_KEY"] = self.config.llm.api_key
        try:
            litellm.client_session = httpx.Client(verify=False)
            lm = dspy.LM(
                model=self.config.llm.model,
                api_base=self.config.llm.api_base,
                model_type="responses",
                api_key=self.config.llm.api_key or "",
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                cache=False,
            )
            dspy.configure(lm=lm)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}") from e

    def _resolve_tolerances(self, spec=None, variant_type="bench-gpu", rtol=None, atol=None):
        ertol = self.config.optimization.correctness_rtol
        eatol = self.config.optimization.correctness_atol
        if spec:
            sr, sa = spec.get_rtol(variant_type), spec.get_atol(variant_type)
            if sr is not None:
                ertol = sr
            if sa is not None:
                eatol = sa
        if rtol is not None:
            ertol = rtol
        if atol is not None:
            eatol = atol
        return ertol, eatol

    def optimize(
        self,
        triton_code,
        pytorch_code,
        kernel_name=None,
        input_shapes=None,
        reference_fn=None,
        stages=None,
        spec_path=None,
        variant_type="bench-gpu",
        target_dtype=None,
        rtol=None,
        atol=None,
    ):
        """Optimize Triton kernel through multi-stage pipeline.

        The pipeline: analyze -> plan -> execute stages in order:
        ALGORITHMIC -> DTYPE -> FUSION -> MEMORY -> BLOCK_PTRS -> PERSISTENT -> XPU.
        Each stage receives its relevant issue list. After each stage, re-analyze.
        """
        import torch

        spec, flop, dtype, init_args = None, None, None, None
        if spec_path:
            from xe_forge.core.spec_loader import load_spec

            spec = load_spec(spec_path)
            input_shapes = spec.get_input_shapes(variant_type)
            flop = spec.get_flop(variant_type)
            dtype = spec.get_dtype(variant_type)
            init_args = spec.get_init_args(variant_type)
            logger.info(f"Loaded spec: shapes={input_shapes}, flop={flop}, dtype={dtype}")
            if init_args:
                logger.info(f"  Model init args: {init_args}")

        ertol, eatol = self._resolve_tolerances(spec, variant_type, rtol, atol)
        if hasattr(self.executor, "rtol"):
            self.executor.rtol = ertol
        if hasattr(self.executor, "atol"):
            self.executor.atol = eatol

        if target_dtype:
            dm = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dm.get(target_dtype, dtype)

        display_name = kernel_name or "Model"
        logger.info(f"Starting optimization for kernel: {display_name}")

        # Measure original performance
        val_orig_tflops, val_orig_ms = None, None
        if self.executor and input_shapes:
            try:
                from xe_forge.core.executor import KernelBenchExecutor

                ex = (
                    self.executor
                    if isinstance(self.executor, KernelBenchExecutor)
                    else KernelBenchExecutor(device=self.config.xpu.device)
                )
                orig_r = ex.execute(
                    triton_code,
                    kernel_name,
                    input_shapes,
                    flop=flop,
                    dtype=dtype,
                    init_args=init_args,
                )
                if orig_r.success:
                    val_orig_tflops, val_orig_ms = orig_r.tflops, orig_r.execution_time_ms
                    logger.info(
                        f"Original: {orig_r.tflops:.2f} TFLOPS, {orig_r.execution_time_ms:.2f} ms"
                    )
            except Exception as e:
                logger.warning(f"Failed to measure original: {e}")

        candidates = []
        best_k = max(1, self.config.optimization.best_k)

        for attempt in range(best_k):
            if best_k > 1:
                logger.info(f"Attempt {attempt + 1}/{best_k}")

            result = OptimizationResult(
                kernel_name=display_name, original_code=triton_code, timestamp=datetime.now()
            )
            result.original_tflops, result.original_ms = val_orig_tflops, val_orig_ms

            etd = target_dtype or self.config.optimization.target_dtype
            if etd is None and dtype is not None:
                etd = {
                    torch.float16: "float16",
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }.get(dtype)

            xpu_config = get_xpu_config_for_pipeline(
                input_shapes=input_shapes, config=self.config, dtype=etd or "float16"
            )

            # === ANALYSIS ===
            logger.info("=" * 60 + "\nSTAGE: ANALYSIS\n" + "=" * 60)
            analysis = self.analyzer.analyze(
                triton_code, pytorch_code, display_name, input_shapes, flop, target_dtype=etd
            )
            result.analysis = analysis

            logger.info(f"Detected {len(analysis.detected_issues)} issues:")
            for iss in analysis.detected_issues:
                logger.info(f"  [{iss.severity}] {iss.issue_type.value}: {iss.description}")

            if not analysis.detected_issues:
                result.success, result.optimized_code = True, triton_code
                candidates.append(result)
                continue

            # === PLANNING ===
            logger.info("=" * 60 + "\nSTAGE: PLANNING\n" + "=" * 60)
            from xe_forge.knowledge.patterns import get_stage_for_issue

            stages_needed: dict[OptimizationStage, list[str]] = {}
            for iss in analysis.detected_issues:
                st = get_stage_for_issue(iss.issue_type)
                stages_needed.setdefault(st, []).append(iss.issue_type.value)

            all_stages = stages or DEFAULT_STAGE_ORDER
            stages_to_apply = [
                s for s in all_stages if s in stages_needed and s != OptimizationStage.ANALYSIS
            ]

            logger.info("Optimization plan:")
            for s in all_stages:
                if s == OptimizationStage.ANALYSIS:
                    continue
                if s in stages_needed:
                    logger.info(f"  + {s.value}: {', '.join(stages_needed[s])}")
                else:
                    logger.info(f"  - {s.value}: skipped")

            if not stages_to_apply:
                result.success, result.optimized_code = True, triton_code
                candidates.append(result)
                continue

            # === EXECUTE STAGES ===
            current_code = triton_code

            for stage in stages_to_apply:
                logger.info("=" * 60 + f"\nSTAGE: {stage.value.upper()}\n" + "=" * 60)
                logger.info(f"Issues: {', '.join(stages_needed.get(stage, []))}")

                stage_result = self.optimizer.optimize_stage(
                    code=current_code,
                    stage=stage,
                    analysis=analysis,
                    xpu_config=xpu_config,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    flop=flop,
                    dtype=dtype,
                    pytorch_code=pytorch_code,
                    init_args=init_args,
                )
                result.stages_applied.append(stage_result)

                if (
                    stage_result.success
                    and stage_result.output_code
                    and stage_result.output_code != current_code
                ):
                    current_code = stage_result.output_code
                    logger.info(
                        f"Stage {stage.value} OK"
                        + (f" ({stage_result.speedup:.2f}x)" if stage_result.speedup else "")
                    )
                elif not stage_result.success:
                    logger.warning(f"Stage {stage.value} failed: {stage_result.error_message}")

                # Re-analyze after each stage
                analysis = self.analyzer.analyze(
                    current_code, pytorch_code, display_name, input_shapes, flop, target_dtype=etd
                )

            # Measure optimized performance
            if self.executor and input_shapes and current_code != triton_code:
                try:
                    from xe_forge.core.executor import KernelBenchExecutor

                    ex = (
                        self.executor
                        if isinstance(self.executor, KernelBenchExecutor)
                        else KernelBenchExecutor(device=self.config.xpu.device)
                    )
                    opt_r = ex.execute(
                        current_code,
                        kernel_name,
                        input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                    if opt_r.success:
                        result.optimized_tflops, result.optimized_ms = (
                            opt_r.tflops,
                            opt_r.execution_time_ms,
                        )
                        if result.original_ms and result.optimized_ms:
                            result.total_speedup = result.original_ms / result.optimized_ms
                            logger.info(f"Total speedup: {result.total_speedup:.2f}x")
                except Exception as e:
                    logger.warning(f"Failed to measure optimized: {e}")

            result.optimized_code, result.success = current_code, True
            candidates.append(result)

        if not candidates:
            return OptimizationResult(
                kernel_name=display_name, original_code=triton_code, timestamp=datetime.now()
            )

        result = max(
            candidates, key=lambda r: r.total_speedup if r.total_speedup is not None else -1.0
        )
        self._save_results(result)

        logger.info("=" * 60 + "\nOPTIMIZATION COMPLETE\n" + "=" * 60)
        ok = [s for s in result.stages_applied if s.success]
        fail = [s for s in result.stages_applied if not s.success]
        logger.info(f"Stages: {len(ok)}/{len(result.stages_applied)} succeeded")
        if fail:
            logger.info(f"Failed: {[s.stage.value for s in fail]}")
        if result.total_speedup:
            logger.info(f"Speedup: {result.total_speedup:.2f}x")
        return result

    def optimize_file(
        self,
        input_path,
        output_path=None,
        kernel_name=None,
        spec_path=None,
        variant_type="bench-gpu",
        target_dtype=None,
    ):
        with open(input_path) as f:
            triton_code = f.read()
        result = self.optimize(
            triton_code,
            "",
            kernel_name,
            spec_path=spec_path,
            variant_type=variant_type,
            target_dtype=target_dtype,
        )
        if output_path and result.optimized_code:
            with open(output_path, "w") as f:
                f.write(result.optimized_code)
        return result

    def _save_results(self, result):
        if not self.config.logging.save_intermediate:
            return
        ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
        kd = Path(self.config.logging.kernel_dir)
        with open(kd / f"{result.kernel_name}_{ts}_original.py", "w") as f:
            f.write(f"# Original: {result.kernel_name}\n\n{result.original_code}")
        if result.optimized_code and result.optimized_code != result.original_code:
            with open(kd / f"{result.kernel_name}_{ts}_optimized.py", "w") as f:
                f.write(f"# Optimized: {result.kernel_name}\n")
                if result.total_speedup:
                    f.write(f"# Speedup: {result.total_speedup:.2f}x\n")
                f.write(
                    f"# Stages: {[s.stage.value for s in result.stages_applied if s.success]}\n\n"
                )
                f.write(result.optimized_code)
