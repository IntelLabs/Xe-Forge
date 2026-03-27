from abc import ABC, abstractmethod

from xpu_forge import OptimizationStage
from xpu_forge.models import KernelAnalysis, StageResult


class Optimizer(ABC):
    @abstractmethod
    def optimize_stage(
        self,
        code: str,
        stage: OptimizationStage,
        analysis: KernelAnalysis,
        xpu_config: dict,
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        flop: float | None = None,
        dtype=None,
        pytorch_code: str = "",
        init_args: list | None = None,
    ) -> StageResult:
        """Apply optimization stage to kernel code."""
        ...
