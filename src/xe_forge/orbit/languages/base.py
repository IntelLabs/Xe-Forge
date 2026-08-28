"""The `LanguageBackend` protocol: one backend per kernel language, each carrying a
`cost_profile` used for budget accounting and ranking. Design rationale: docs/DESIGN.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xe_forge.orbit.models import (
    BuildRecipe,
    ExtractionCheck,
    KernelBundle,
    KernelLanguage,
    SourceLocation,
)


@dataclass
class CostProfile:
    """What one iteration in this language actually costs."""

    build_seconds: float
    iteration_seconds: float
    notes: str = ""

    @property
    def relative_cost(self) -> float:
        """Cost of one candidate relative to a Triton JIT iteration."""
        return max(1.0, (self.build_seconds + self.iteration_seconds) / 5.0)


@dataclass
class CompilerAxis:
    """A compiler or runtime option that can be swept without touching code."""

    name: str
    values: list[Any] = field(default_factory=list)
    flag_template: str = ""
    description: str = ""
    changes_numerics: bool = False


@dataclass
class BuildResult:
    ok: bool
    output: str = ""
    artifact: Path | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@runtime_checkable
class LanguageBackend(Protocol):
    """One backend per kernel language."""

    name: str
    cost_profile: CostProfile

    def identify(self, kernel_name: str) -> float: ...
    def resolve_source(self, kernel_name: str, launch: Any) -> SourceLocation: ...
    def closure(self, source: SourceLocation) -> list[Path]: ...
    def build(self, bundle: KernelBundle) -> BuildResult: ...
    def harness(self, bundle: KernelBundle) -> Path: ...
    def verify(self, bundle: KernelBundle) -> ExtractionCheck: ...
    def option_axes(self) -> list[CompilerAxis]: ...


class BaseLanguageBackend:
    """Shared defaults. Subclasses override what their language actually supports."""

    name: str = "base"
    language: KernelLanguage = KernelLanguage.OPAQUE
    cost_profile: CostProfile = CostProfile(build_seconds=0.0, iteration_seconds=1.0)

    def identify(self, kernel_name: str) -> float:
        return 0.0

    def resolve_source(self, kernel_name: str, launch: Any = None) -> SourceLocation:
        return SourceLocation(symbol=kernel_name, confidence=0.0)

    def closure(self, source: SourceLocation) -> list[Path]:
        return []

    def build(self, bundle: KernelBundle) -> BuildResult:
        return BuildResult(ok=False, reason=f"{self.name} backend cannot build bundles")

    def harness(self, bundle: KernelBundle) -> Path:
        raise NotImplementedError(f"{self.name} backend has no harness generator")

    def verify(self, bundle: KernelBundle) -> ExtractionCheck:
        return ExtractionCheck(verified=False, failures=[f"{self.name}: no verifier"])

    def option_axes(self) -> list[CompilerAxis]:
        return []

    def build_recipe(self, source: SourceLocation) -> BuildRecipe | None:
        return None
