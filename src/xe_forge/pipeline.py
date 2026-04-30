import logging
import os
from datetime import datetime
from pathlib import Path

import dspy
import httpx
import litellm

from xe_forge.agents import AnalyzerAgent, Optimizer, OptimizerAgent, OptimizerReActAgent
from xe_forge.config import Config, get_config
from xe_forge.core.device_query import get_device_config_for_pipeline
from xe_forge.knowledge.loader import KnowledgeBase, load_knowledge_base
from xe_forge.models import (
    DSL,
    IssueType,
    OptimizationResult,
    OptimizationStage,
)
from xe_forge.planner import DEFAULT_STAGE_ORDER as PLANNER_DEFAULT_STAGE_ORDER
from xe_forge.planner import PlannerAgent

logger = logging.getLogger(__name__)


def _extract_gemm_dims(
    input_shapes: list[tuple[int, ...]] | None,
) -> tuple[int, int, int]:
    """Extract M, N, K from GEMM input shapes [(M, K), (K, N)]."""
    if input_shapes and len(input_shapes) >= 2:
        a, b = input_shapes[0], input_shapes[1]
        if len(a) >= 2 and len(b) >= 2:
            return a[-2], b[-1], a[-1]
    return 1024, 1024, 1024


DEFAULT_STAGE_ORDER: list[OptimizationStage] = [
    OptimizationStage.ANALYSIS,
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.DISCOVERY,
    OptimizationStage.DTYPE_FIX,
    OptimizationStage.FUSION,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.BLOCK_POINTERS,
    OptimizationStage.PERSISTENT_KERNEL,
    OptimizationStage.DEVICE_SPECIFIC,
    OptimizationStage.AUTOTUNING,
]


class XeForgePipeline:
    config: Config
    analyzer: AnalyzerAgent
    optimizer: Optimizer

    def __init__(self, config=None, executor=None, validator=None):
        self.config = config or get_config()
        self._setup_logging()
        self._setup_llm()

        if executor is None:
            if self.config.device_config.dsl == DSL.SYCL:
                from xe_forge.core import SyclExecutor

                executor = SyclExecutor(
                    verify=self.config.optimization.require_correctness,
                )
            else:
                from xe_forge.core import KernelBenchExecutor

                executor = KernelBenchExecutor(
                    device=self.config.device_config.device,
                    require_correctness=self.config.optimization.require_correctness,
                    rtol=self.config.optimization.correctness_rtol,
                    atol=self.config.optimization.correctness_atol,
                )

        self.knowledge_base: KnowledgeBase | None = None
        if self.config.knowledge.enabled:
            self.knowledge_base = load_knowledge_base(
                self.config.knowledge.knowledge_dir,
                dsl=self.config.device_config.dsl,
                device_type=self.config.device_config.device,
            )
            logger.info("  Knowledge base: %s", self.knowledge_base.summary())
        else:
            logger.info("  Knowledge base: disabled (set KNOWLEDGE_BASE_ENABLED=true to enable)")

        self.analyzer = AnalyzerAgent(
            knowledge_base=self.knowledge_base,
            dsl=self.config.device_config.dsl,
        )
        self.planner = PlannerAgent()

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
            knowledge_base=self.knowledge_base,
            dsl=self.config.device_config.dsl,
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
            dspy.configure(lm=lm, warn_on_type_mismatch=False)
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
        kernel_code=None,
        reference_code=None,
        kernel_name=None,
        input_shapes=None,
        reference_fn=None,
        stages=None,
        spec_path=None,
        variant_type="bench-gpu",
        target_dtype=None,
        rtol=None,
        atol=None,
        *,
        triton_code=None,
        pytorch_code=None,
    ):
        # Backward compat aliases
        if kernel_code is None:
            kernel_code = triton_code
        if reference_code is None:
            reference_code = pytorch_code or ""
        import torch

        spec, flop, dtype, init_args, spec_dims = None, None, None, None, None
        if spec_path:
            from xe_forge.core.spec_loader import load_spec

            spec = load_spec(spec_path)
            variant_type = spec.resolve_variant(variant_type)
            input_shapes = spec.get_input_shapes(variant_type)
            spec_dims = spec.get_dims(variant_type)
            flop = spec.get_flop(variant_type)
            dtype = spec.get_dtype(variant_type)
            init_args = spec.get_init_args(variant_type)
            logger.info(
                f"Loaded spec: variant={variant_type}, shapes={input_shapes}, "
                f"dims={spec_dims}, flop={flop}, dtype={dtype}"
            )
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

        val_orig_tflops, val_orig_ms = None, None
        from xe_forge.core.executor import KernelBenchExecutor
        from xe_forge.core.sycl_executor import SyclExecutor

        _is_sycl = isinstance(self.executor, SyclExecutor)
        _bench_ex = (
            self.executor
            if isinstance(self.executor, (KernelBenchExecutor, SyclExecutor))
            else KernelBenchExecutor(device=self.config.device_config.device)
        )
        if self.executor and (_is_sycl or input_shapes):
            try:
                if _is_sycl:
                    _sycl_dims = spec_dims or dict(
                        zip(("M", "N", "K"), _extract_gemm_dims(input_shapes))
                    )
                    orig_r = _bench_ex.execute(
                        kernel_code=kernel_code,
                        dims=_sycl_dims,
                    )
                else:
                    orig_r = _bench_ex.execute(
                        kernel_code,
                        None,
                        input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                if orig_r.success:
                    val_orig_tflops, val_orig_ms = orig_r.tflops, orig_r.execution_time_ms
                    logger.info(f"Original: {val_orig_tflops:.2f} TFLOPS, {val_orig_ms:.2f} ms")
                else:
                    logger.error(f"Baseline FAILED: {orig_r.error_message}")
                    if hasattr(orig_r, "error_traceback"):
                        logger.debug(orig_r.error_traceback)
            except Exception as e:
                logger.warning(f"Failed to measure original: {e}")

        candidates = []
        best_k = max(1, self.config.optimization.best_k)

        for attempt in range(best_k):
            if best_k > 1:
                logger.info(f"Attempt {attempt + 1}/{best_k}")

            result = OptimizationResult(
                kernel_name=display_name, original_code=kernel_code, timestamp=datetime.now()
            )
            result.original_tflops, result.original_ms = val_orig_tflops, val_orig_ms

            etd = target_dtype or self.config.optimization.target_dtype
            if etd is None and dtype is not None:
                etd = {
                    torch.float16: "float16",
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }.get(dtype)

            device_type = self.config.device_config.device
            xpu_config = get_device_config_for_pipeline(
                device_type=device_type,
                input_shapes=input_shapes,
                config=self.config,
                dtype=etd or "float16",
            )

            logger.info("=" * 60 + "\nSTAGE: ANALYSIS\n" + "=" * 60)
            analysis = self.analyzer.analyze(
                kernel_code,
                reference_code,
                display_name,
                input_shapes,
                flop,
                target_dtype=etd,
            )
            result.analysis = analysis

            logger.info(f"Detected {len(analysis.detected_issues)} issues:")
            for iss in analysis.detected_issues:
                logger.info(f"  [{iss.severity}] {iss.issue_type.value}: {iss.description}")

            if not analysis.detected_issues:
                result.success, result.optimized_code = True, kernel_code
                candidates.append(result)
                continue

            logger.info("=" * 60 + "\nSTAGE: PLANNING\n" + "=" * 60)
            from xe_forge.knowledge.patterns import get_stage_for_issue

            stages_needed: dict[OptimizationStage, list[str]] = {}
            for iss in analysis.detected_issues:
                st = get_stage_for_issue(iss.issue_type)
                stages_needed.setdefault(st, []).append(iss.issue_type.value)

            from xe_forge.dsl_registry import get_stages_for_dsl

            _supported = set(get_stages_for_dsl(self.config.device_config.dsl))
            stages_needed = {s: v for s, v in stages_needed.items() if s in _supported}

            if stages:
                stages_to_apply = [
                    s for s in stages if s in stages_needed and s != OptimizationStage.ANALYSIS
                ]
                logger.info("Stage order: manual override")
            else:
                stages_to_apply = self.planner.plan(
                    stages_needed=stages_needed,
                    analysis=analysis,
                    input_shapes=input_shapes,
                    flop=flop,
                )

            logger.info("Optimization plan:")
            for s in PLANNER_DEFAULT_STAGE_ORDER:
                if s == OptimizationStage.ANALYSIS:
                    continue
                if s in stages_needed:
                    if s in stages_to_apply:
                        pos = stages_to_apply.index(s) + 1
                        issues_str = ", ".join(stages_needed[s])
                        logger.info(f"  + {s.value} [#{pos}]: {issues_str}")
                    else:
                        issues_str = ", ".join(stages_needed[s])
                        logger.info(f"  ~ {s.value} (deferred): {issues_str}")
                else:
                    logger.info(f"  - {s.value}: skipped")

            if not stages_to_apply:
                result.success, result.optimized_code = True, kernel_code
                candidates.append(result)
                continue

            current_code = kernel_code
            current_ms: float | None = val_orig_ms

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
                    spec_dims=spec_dims,
                    flop=flop,
                    dtype=dtype,
                    pytorch_code=reference_code,
                    init_args=init_args,
                    perf_context={
                        "original_ms": val_orig_ms,
                        "original_tflops": val_orig_tflops,
                        "current_ms": current_ms,
                        "speedup_so_far": (
                            round(val_orig_ms / current_ms, 3)
                            if val_orig_ms and current_ms and current_ms > 0
                            else None
                        ),
                    },
                )
                result.stages_applied.append(stage_result)

                if (
                    stage_result.success
                    and stage_result.output_code
                    and stage_result.output_code != current_code
                ):
                    current_code = stage_result.output_code
                    if stage_result.speedup and val_orig_ms:
                        current_ms = val_orig_ms / stage_result.speedup
                    elif (
                        stage_result.metrics_after
                        and "execution_time_ms" in stage_result.metrics_after
                    ):
                        current_ms = stage_result.metrics_after["execution_time_ms"]
                    logger.info(
                        f"Stage {stage.value} OK"
                        + (f" ({stage_result.speedup:.2f}x)" if stage_result.speedup else "")
                    )
                elif not stage_result.success:
                    logger.warning(f"Stage {stage.value} failed: {stage_result.error_message}")

                if stage == OptimizationStage.DISCOVERY and stage_result.success:
                    open_ended_issues = [
                        i
                        for i in analysis.detected_issues
                        if i.issue_type == IssueType.OPEN_ENDED and i.open_ended_proposal
                    ]
                    for oi in open_ended_issues:
                        logger.info(
                            "DISCOVERY succeeded — promote to named IssueType:\n%s",
                            oi.open_ended_proposal,
                        )

                analysis = self.analyzer.analyze(
                    current_code,
                    reference_code,
                    display_name,
                    input_shapes,
                    flop,
                    target_dtype=etd,
                )

            if self.executor and (_is_sycl or input_shapes) and current_code != kernel_code:
                try:
                    if _is_sycl:
                        opt_r = _bench_ex.execute(
                            kernel_code=current_code,
                            dims=_sycl_dims,
                        )
                    else:
                        opt_r = _bench_ex.execute(
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
                    if current_ms and current_ms != val_orig_ms:
                        result.optimized_ms = current_ms
                        if result.original_ms and result.optimized_ms:
                            result.total_speedup = result.original_ms / result.optimized_ms
                            logger.info(
                                f"Total speedup (from stage measurements): {result.total_speedup:.2f}x"
                            )

            result.optimized_code, result.success = current_code, True
            candidates.append(result)

        if not candidates:
            return OptimizationResult(
                kernel_name=display_name, original_code=kernel_code, timestamp=datetime.now()
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
            kernel_code = f.read()
        result = self.optimize(
            kernel_code,
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
        ext = ".cpp" if DSL(self.config.device_config.dsl).code_language == "cpp" else ".py"
        comment = "//" if ext == ".cpp" else "#"
        ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
        kd = Path(self.config.logging.kernel_dir)
        with open(kd / f"{result.kernel_name}_{ts}_original{ext}", "w") as f:
            f.write(f"{comment} Original: {result.kernel_name}\n\n{result.original_code}")
        if result.optimized_code and result.optimized_code != result.original_code:
            with open(kd / f"{result.kernel_name}_{ts}_optimized{ext}", "w") as f:
                f.write(f"{comment} Optimized: {result.kernel_name}\n")
                if result.total_speedup:
                    f.write(f"{comment} Speedup: {result.total_speedup:.2f}x\n")
                f.write(
                    f"{comment} Stages: {[s.stage.value for s in result.stages_applied if s.success]}\n\n"
                )
                f.write(result.optimized_code)
