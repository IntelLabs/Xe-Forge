"""DSPy engine: wraps XeForgePipeline with trial management and profiling."""

from __future__ import annotations

import logging
from pathlib import Path

from xe_forge.config import Config
from xe_forge.core.profiler import XPUProfiler
from xe_forge.core.trial_manager import TrialManager
from xe_forge.engines.base import BaseEngine
from xe_forge.models import OptimizationResult, OptimizationStage

logger = logging.getLogger(__name__)


class DSPyEngine(BaseEngine):
    """Automated optimization via DSPy pipeline + trial tracking + VTune."""

    def __init__(self, config: Config, executor=None):
        super().__init__(config)
        self.executor = executor

        self.trial_manager: TrialManager | None = None
        if config.trial.enabled:
            self.trial_manager = TrialManager(config.trial.trials_dir)

        self.profiler: XPUProfiler | None = None
        if config.profiler.vtune_enabled:
            self.profiler = XPUProfiler(config.profiler.vtune_bin)
            if not self.profiler.available():
                logger.warning(
                    "VTune not found at '%s', profiling disabled.", config.profiler.vtune_bin
                )
                self.profiler = None

    def optimize(
        self,
        kernel_code: str,
        reference_code: str = "",
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        spec_path: str | None = None,
        variant_type: str = "bench-gpu",
        target_dtype: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        stages: list[OptimizationStage] | None = None,
    ) -> OptimizationResult:
        from xe_forge.pipeline import XeForgePipeline

        pipeline = XeForgePipeline(
            config=self.config,
            executor=self.executor,
            trial_manager=self.trial_manager,
            profiler=self.profiler,
        )

        result = pipeline.optimize(
            kernel_code=kernel_code,
            reference_code=reference_code,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            stages=stages,
            spec_path=spec_path,
            variant_type=variant_type,
            target_dtype=target_dtype,
            rtol=rtol,
            atol=atol,
        )

        if self.trial_manager and kernel_name and result.optimized_code:
            try:
                self.trial_manager.finalize(
                    kernel_name,
                    str(Path(self.config.trial.trials_dir) / kernel_name / "best_output.py"),
                )
            except Exception:
                logger.warning("Could not finalize trial for '%s'", kernel_name, exc_info=True)

        return result
