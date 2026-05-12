"""Abstract base class for optimization engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from xe_forge.config import Config
from xe_forge.models import OptimizationResult, OptimizationStage


class BaseEngine(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
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
        """Run the optimization loop and return results."""
        ...
