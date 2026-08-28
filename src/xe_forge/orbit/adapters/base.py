"""
The framework adapter protocol (plan §10).

vLLM is the first target, not a special one. The way that fails is subtle: vLLM gets
built first, its assumptions leak into the analyzer, the measurement layer and the
patch logic, and the second framework needs a core rewrite. Two rules prevent it:

* **The core imports no framework.** Nothing in `orbit/models`, `orbit/analysis`,
  `orbit/extract` or `orbit/patch` may import vllm, sglang or any serving package.
  Enforced by a test that scans imports, not by convention (`test_core_purity`).
* **Every framework is reached through this protocol**, one adapter each, and every
  adapter passes the same conformance suite. Adding a framework is then a bounded,
  testable unit of work rather than a negotiation with the core.

Capabilities are *declared*, never assumed (§10.4). If an adapter cannot report TTFT,
the analysis falls back to throughput or wall-clock and says so in the report. It
never substitutes one metric for another silently, and where a capability is missing
the affected actions are removed from the space rather than attempted and failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xe_forge.orbit.executor import Executor
from xe_forge.orbit.models import (
    ConfigAxis,
    DeterminismProfile,
    FrameworkCapabilities,
    KernelRecord,
    PatchPoint,
    QualityResult,
    WorkloadMeasurement,
    WorkloadSpec,
)


@dataclass
class PreparedWorkload:
    """A workload with adapter-specific setup applied."""

    spec: WorkloadSpec
    notes: list[str] = field(default_factory=list)


@dataclass
class Handle:
    """A live (or prepared) workload the adapter can measure and tear down."""

    spec: WorkloadSpec
    adapter: str
    state: dict[str, Any] = field(default_factory=dict)
    warmed_up: bool = False


@dataclass
class LoadSpec:
    """What load to apply when benchmarking."""

    repetitions: int = 5
    warmup: int = 1
    profile_id: str | None = None
    metric: str = "wall_time"


@dataclass
class MetricSpec:
    name: str
    unit: str = ""
    lower_is_better: bool = True
    description: str = ""


@runtime_checkable
class FrameworkAdapter(Protocol):
    """One adapter per framework. Same protocol, same conformance obligations."""

    name: str
    tier: int
    capabilities: FrameworkCapabilities

    # identity and lifecycle
    def detect(self, spec: WorkloadSpec) -> bool: ...
    def versions(self) -> dict[str, str]: ...
    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload: ...
    def launch(self, spec: WorkloadSpec, executor: Executor) -> Handle: ...
    def warmup(self, handle: Handle) -> None: ...
    def teardown(self, handle: Handle) -> None: ...

    # measurement
    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement: ...
    def metrics_schema(self) -> list[MetricSpec]: ...

    # reproducibility
    def determinism_profile(self) -> DeterminismProfile: ...
    def reset_state(self, handle: Handle) -> None: ...

    # discovery and provenance
    def dispatch_roots(self) -> list[str]: ...
    def provenance_hints(self) -> list[str]: ...

    # extraction
    def build_in_situ_harness(self, kernel: KernelRecord, inputs: Any) -> Path: ...
    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]: ...

    # action space
    def config_axes(self) -> list[ConfigAxis]: ...
    def apply_config(self, spec: WorkloadSpec, config: dict[str, Any]) -> WorkloadSpec: ...

    # correctness
    def quality_gate(self, handle: Handle, prompts: list[str]) -> QualityResult: ...


class AdapterError(RuntimeError):
    """Raised when an adapter is asked for something it never declared."""


class BaseAdapter:
    """Shared behaviour so a new adapter is genuinely one class plus a knowledge file.

    Subclasses override what they actually support. Every default here either works
    generically or raises a clear `AdapterError` naming the missing capability — never
    a silent no-op, because a silently-missing capability is how a measurement chain
    produces a confident wrong number.
    """

    name: str = "base"
    tier: int = 0
    capabilities: FrameworkCapabilities = FrameworkCapabilities()

    def detect(self, spec: WorkloadSpec) -> bool:
        return False

    def versions(self) -> dict[str, str]:
        return {}

    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload:
        return PreparedWorkload(spec=spec)

    def launch(self, spec: WorkloadSpec, executor: Executor) -> Handle:
        return Handle(spec=spec, adapter=self.name, state={"executor": executor})

    def warmup(self, handle: Handle) -> None:
        handle.warmed_up = True

    def teardown(self, handle: Handle) -> None:
        handle.state.clear()

    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement:
        raise NotImplementedError

    def metrics_schema(self) -> list[MetricSpec]:
        return [MetricSpec(name="wall_time", unit="s", lower_is_better=True)]

    def determinism_profile(self) -> DeterminismProfile:
        return DeterminismProfile()

    def reset_state(self, handle: Handle) -> None:
        if not self.capabilities.can_reset_state:
            raise AdapterError(
                f"{self.name} does not declare can_reset_state; refusing to pretend it did"
            )

    def dispatch_roots(self) -> list[str]:
        return []

    def provenance_hints(self) -> list[str]:
        return []

    def build_in_situ_harness(self, kernel: KernelRecord, inputs: Any) -> Path:
        if not self.capabilities.can_construct_single_layer:
            raise AdapterError(
                f"{self.name} cannot construct a single layer, so E3 extraction is "
                f"unavailable for {kernel.id}; the ranking should have excluded it"
            )
        raise NotImplementedError

    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]:
        return []

    def config_axes(self) -> list[ConfigAxis]:
        return []

    def apply_config(self, spec: WorkloadSpec, config: dict[str, Any]) -> WorkloadSpec:
        env = dict(spec.env)
        for axis in self.config_axes():
            if axis.name in config and axis.env_var:
                env[axis.env_var] = str(config[axis.name])
        return spec.model_copy(update={"env": env})

    def quality_gate(self, handle: Handle, prompts: list[str]) -> QualityResult:
        return QualityResult(
            passed=False,
            detail=f"{self.name} does not implement a quality gate",
        )

    def supports_metric(self, metric: str) -> bool:
        return metric in self.capabilities.metrics
