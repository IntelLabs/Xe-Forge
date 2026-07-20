"""
CoordinatorAgent — dspy.ReActV2-driven agentic orchestrator for kernel optimization.

Replaces the fixed analyze→plan→stage loop with an LLM that decides at runtime:
what to analyze, which stages to apply, whether to profile before/after a stage,
and when to stop. Code never flows through tool arguments — CoordinatorState holds
all code state so the LLM orchestrates at semantic level.
"""

from __future__ import annotations

import logging

import dspy

from xe_forge.agents.coordinator_tools import (
    CoordinatorState,
    make_analyze_tool,
    make_apply_stage_tool,
    make_benchmark_tool,
    make_profile_tool,
    make_retrieve_patterns_tool,
    make_status_tool,
)
from xe_forge.models import DSL, StageResult

logger = logging.getLogger(__name__)


class CoordinatorSignature(dspy.Signature):
    """Orchestrate GPU kernel optimization to maximize speedup using specialist tools."""

    kernel_specs: str = dspy.InputField(
        desc="Kernel specs: name, input shapes, dtype, FLOP count, device, DSL"
    )
    achieved_speedup: float = dspy.OutputField(
        desc="Final speedup achieved vs original baseline (1.0 = no improvement)"
    )
    optimization_summary: str = dspy.OutputField(
        desc="Concise summary: which stages were applied, what changed, final speedup"
    )


class CoordinatorAgent(dspy.Module):
    """Agentic coordinator that uses dspy.ReActV2 to drive multi-stage kernel optimization.

    DSPy >=3.3.0b1 required — dspy.ReActV2 is used unconditionally.
    """

    def __init__(
        self,
        analyzer,
        executor,
        knowledge_base=None,
        profiler=None,
        max_iters: int = 20,
        extra_instructions: str = "",
        dsl: DSL | str = DSL.TRITON,
    ):
        super().__init__()
        self.analyzer = analyzer
        self.executor = executor
        self.knowledge_base = knowledge_base
        self.profiler = profiler
        self.max_iters = max_iters
        self.extra_instructions = extra_instructions
        self.dsl = DSL(dsl) if isinstance(dsl, str) else dsl

    def run(
        self,
        kernel_code: str,
        kernel_specs: str,
        pytorch_code: str = "",
        kernel_name: str | None = None,
        input_shapes=None,
        flop=None,
        dtype=None,
        spec_dims=None,
        init_args=None,
        input_dtypes=None,
        xpu_config: dict | None = None,
        spec_path: str | None = None,
        variant_type: str = "bench-gpu",
    ) -> tuple[str, float, str, list[StageResult]]:
        """Run the coordinator agent.

        Returns (best_code, best_speedup, summary, stage_results).
        """
        state = CoordinatorState(
            original_code=kernel_code,
            current_code=kernel_code,
            best_code=kernel_code,
        )

        # Shared mutable ref so profile_tool can update the vtune_report used by apply_stage
        vtune_report_ref: list[str] = [""]

        tools = self._build_tools(
            state=state,
            pytorch_code=pytorch_code,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            spec_dims=spec_dims,
            init_args=init_args,
            input_dtypes=input_dtypes,
            xpu_config=xpu_config or {},
            spec_path=spec_path,
            variant_type=variant_type,
            vtune_report_ref=vtune_report_ref,
        )

        sig = CoordinatorSignature
        # Inject coordinator guidance from template
        try:
            from xe_forge.config import get_config
            from xe_forge.prompts import render_signature_instructions
            from xe_forge.prompts.device_prompts import (
                _DEVICE_DESCRIPTIONS,
                _DEVICE_TUNING_DEFAULTS,
                _DSL_NAMES,
            )
            cfg = get_config()
            template_text = render_signature_instructions(
                "coordinator_signature",
                dsl=str(self.dsl.value if hasattr(self.dsl, "value") else self.dsl),
                dsl_name=_DSL_NAMES.get(
                    str(self.dsl.value if hasattr(self.dsl, "value") else self.dsl), "Triton"
                ),
                device_type=cfg.device_config.device,
                device_description=_DEVICE_DESCRIPTIONS.get(cfg.device_config.device, "Intel XPU"),
                defaults=_DEVICE_TUNING_DEFAULTS.get(cfg.device_config.device, {}),
            )
            sig = sig.append_instructions(template_text)
        except Exception as e:
            logger.debug("Coordinator template render failed: %s", e)

        if self.extra_instructions:
            sig = sig.append_instructions(self.extra_instructions)

        # DSPy >=3.3.0b1 required
        agent = dspy.ReActV2(
            signature=sig,
            tools=tools,
            max_iters=self.max_iters,
        )

        logger.info(
            "CoordinatorAgent: starting ReActV2 (max_iters=%d)", self.max_iters
        )

        try:
            result = agent(kernel_specs=kernel_specs)
        except Exception as e:
            logger.error("CoordinatorAgent: ReActV2 failed: %s", e)
            return state.best_code, state.best_speedup, f"Coordinator failed: {e}", state.stage_results

        termination = getattr(result, "termination_reason", None)
        if termination:
            logger.info("CoordinatorAgent termination_reason: %s", termination)
            if termination in ("max_iters", "context_window_exceeded"):
                logger.warning("Coordinator stopped early due to %s", termination)

        llm_speedup = getattr(result, "achieved_speedup", None)
        summary = getattr(result, "optimization_summary", "") or ""

        # Use the better of: LLM-reported speedup vs what Python state tracked
        final_speedup = max(
            float(llm_speedup) if llm_speedup is not None else 0.0,
            state.best_speedup,
        )

        logger.info(
            "CoordinatorAgent done: speedup=%.3fx, stages_succeeded=%s",
            final_speedup,
            state.stages_succeeded,
        )

        return state.best_code, final_speedup, summary, state.stage_results

    def _build_tools(
        self,
        state: CoordinatorState,
        pytorch_code: str,
        kernel_name,
        input_shapes,
        flop,
        dtype,
        spec_dims,
        init_args,
        input_dtypes,
        xpu_config: dict,
        spec_path,
        variant_type: str,
        vtune_report_ref: list[str],
    ) -> list:
        tools = [
            dspy.Tool(
                make_analyze_tool(
                    state=state,
                    analyzer=self.analyzer,
                    pytorch_code=pytorch_code,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    flop=flop,
                    dtype=dtype,
                )
            ),
            dspy.Tool(
                make_retrieve_patterns_tool(
                    state=state,
                    knowledge_base=self.knowledge_base,
                )
            ),
            dspy.Tool(
                make_apply_stage_tool(
                    state=state,
                    optimizer=self._build_stage_optimizer(xpu_config),
                    xpu_config=xpu_config,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    spec_dims=spec_dims,
                    flop=flop,
                    dtype=dtype,
                    pytorch_code=pytorch_code,
                    init_args=init_args,
                    input_dtypes=input_dtypes,
                    vtune_report_ref=vtune_report_ref,
                )
            ),
            dspy.Tool(
                make_benchmark_tool(
                    state=state,
                    executor=self.executor,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    flop=flop,
                    dtype=dtype,
                    spec_dims=spec_dims,
                    init_args=init_args,
                    input_dtypes=input_dtypes,
                )
            ),
            dspy.Tool(make_status_tool(state=state)),
        ]

        if self.profiler is not None and hasattr(self.profiler, "available") and self.profiler.available():
            tools.append(
                dspy.Tool(
                    make_profile_tool(
                        state=state,
                        profiler=self.profiler,
                        kernel_name=kernel_name,
                        spec_path=spec_path,
                        variant_type=variant_type,
                        vtune_report_ref=vtune_report_ref,
                    )
                )
            )

        return tools

    def _build_stage_optimizer(self, xpu_config: dict):
        """Build an OptimizerAgent for the apply_stage tool to delegate to."""
        from xe_forge.agents.optimizer_agent import OptimizerAgent

        return OptimizerAgent(
            executor=self.executor,
            knowledge_base=self.knowledge_base,
            dsl=self.dsl,
            extra_instructions=self.extra_instructions,
        )
