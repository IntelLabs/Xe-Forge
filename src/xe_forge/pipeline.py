import logging
import os
from datetime import datetime
from pathlib import Path

import dspy
import httpx
import litellm

from xe_forge.agents import (
    AnalyzerAgent,
    GeneratorAgent,
    Optimizer,
    OptimizerAgent,
    OptimizerReActAgent,
)
from xe_forge.config import Config, get_config
from xe_forge.core import get_xpu_config_for_pipeline
from xe_forge.knowledge.loader import KnowledgeBase, load_knowledge_base
from xe_forge.models import (
    OptimizationResult,
    OptimizationStage,
)
from xe_forge.planner import DEFAULT_STAGE_ORDER as PLANNER_DEFAULT_STAGE_ORDER  # noqa: F401
from xe_forge.planner import PlannerAgent
from xe_forge.trials import TrialContext, TrialRunner
from xe_forge.trials.search import get_search
from xe_forge.trials.writers import get_writer_cls

logger = logging.getLogger(__name__)

DEFAULT_STAGE_ORDER: list[OptimizationStage] = [
    OptimizationStage.ANALYSIS,
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.DISCOVERY,
    OptimizationStage.DTYPE_FIX,
    OptimizationStage.FUSION,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.BLOCK_POINTERS,
    OptimizationStage.PERSISTENT_KERNEL,
    OptimizationStage.XPU_SPECIFIC,
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
            from xe_forge.core import KernelBenchExecutor

            executor = KernelBenchExecutor(
                device=self.config.xpu.device,
                require_correctness=self.config.optimization.require_correctness,
                rtol=self.config.optimization.correctness_rtol,
                atol=self.config.optimization.correctness_atol,
            )

        self.knowledge_base: KnowledgeBase | None = None
        if self.config.knowledge.enabled:
            self.knowledge_base = load_knowledge_base(self.config.knowledge.knowledge_dir)
            logger.info("  Knowledge base: %s", self.knowledge_base.summary())
        else:
            logger.info("  Knowledge base: disabled (set KNOWLEDGE_BASE_ENABLED=true to enable)")

        self.analyzer = AnalyzerAgent(knowledge_base=self.knowledge_base)
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
        pytorch_path=None,
        output_path=None,
    ):
        """Drive the trial-tree engine and return an ``OptimizationResult``.

        Back-compat shape: the signature of the old flat-``best_k`` method is
        preserved so existing callers (CLI + tests) keep working. Internally
        this now delegates to ``TrialRunner``, which does tree-structured
        search (default: Triton8 semantics) using configurable writers.

        When ``triton_code`` is empty and ``pytorch_code`` is provided the
        runner invokes ``GeneratorAgent`` to synthesize t0.
        """
        import torch

        spec, flop, dtype, init_args = None, None, None, None
        if spec_path:
            from xe_forge.core.spec_loader import load_spec

            spec = load_spec(spec_path)
            variant_type = spec.resolve_variant(variant_type)
            input_shapes = spec.get_input_shapes(variant_type)
            flop = spec.get_flop(variant_type)
            dtype = spec.get_dtype(variant_type)
            init_args = spec.get_init_args(variant_type)
            logger.info(
                f"Loaded spec: variant={variant_type}, shapes={input_shapes}, "
                f"flop={flop}, dtype={dtype}"
            )
            if init_args:
                logger.info(f"  Model init args: {init_args}")

        ertol, eatol = self._resolve_tolerances(spec, variant_type, rtol, atol)
        if hasattr(self.executor, "rtol"):
            self.executor.rtol = ertol
        if hasattr(self.executor, "atol"):
            self.executor.atol = eatol

        if target_dtype:
            dm = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dm.get(target_dtype, dtype)

        display_name = kernel_name or "Model"
        logger.info(f"Starting trial-tree optimization for kernel: {display_name}")

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

        ctx = TrialContext(
            kernel_name=display_name,
            config=self.config,
            spec=spec,
            variant_type=variant_type,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            init_args=init_args,
            xpu_config=xpu_config,
            knowledge_base=self.knowledge_base,
            pytorch_code=pytorch_code or "",
            pytorch_path=pytorch_path,
        )

        search = get_search(self.config.trial.search)
        writer = self._build_writer(self.config.trial.writer)
        generator = self._build_generator()

        runner = TrialRunner(
            ctx=ctx,
            search=search,
            writer=writer,
            executor=self.executor,
            generator=generator,
        )

        if triton_code:
            trial_result = runner.run(triton_code=triton_code, output_path=output_path)
        else:
            trial_result = runner.run(
                pytorch_code=pytorch_code or "",
                pytorch_path=pytorch_path,
                output_path=output_path,
            )

        # Translate TrialRunResult -> OptimizationResult so callers unchanged.
        result = OptimizationResult(
            kernel_name=display_name,
            original_code=triton_code or pytorch_code or "",
            timestamp=datetime.now(),
        )
        result.success = trial_result.success
        result.optimized_code = trial_result.best_code or triton_code or ""
        if trial_result.baseline_us:
            result.original_ms = trial_result.baseline_us / 1000.0
        if trial_result.best_speedup and trial_result.best_speedup > 0:
            result.total_speedup = trial_result.best_speedup
            if result.original_ms:
                result.optimized_ms = result.original_ms / trial_result.best_speedup

        logger.info("=" * 60 + "\nTRIAL RUN COMPLETE\n" + "=" * 60)
        logger.info(
            "trials=%d best=%s speedup=%s output=%s",
            trial_result.trials_count,
            trial_result.best_trial_id,
            f"{trial_result.best_speedup:.2f}x" if trial_result.best_speedup else "n/a",
            trial_result.output_path,
        )
        self._save_results(result)
        return result

    def _build_writer(self, name: str):
        """Construct a writer with the deps it needs.

        Only ``stage_sequence`` is built-in; it wires the existing per-stage
        pipeline (analyzer + planner + optimizer) into the trial tree. If a
        user registers additional writer types, this method is where their
        dependency wiring lives.
        """
        cls = get_writer_cls(name)
        if name == "stage_sequence":
            return cls(
                analyzer=self.analyzer,
                planner=self.planner,
                optimizer=self.optimizer,
                display_name="Model",
            )
        # Unknown writer — assume zero-dep constructor.
        return cls()

    def _build_generator(self) -> GeneratorAgent:
        return GeneratorAgent(
            knowledge_base=self.knowledge_base,
            max_iterations=self.config.agent.max_iterations,
        )

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
