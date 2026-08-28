"""
Tier 0 framework-agnostic adapter: kernel discovery, provenance, input capture and a
wall-clock number for any torch workload, so an unfamiliar framework degrades to
Tier 0 rather than failing. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

from typing import Any

from xe_forge.orbit.adapters.base import (
    BaseAdapter,
    Handle,
    LoadSpec,
    MetricSpec,
    PreparedWorkload,
)
from xe_forge.orbit.bench.core import BenchRunner
from xe_forge.orbit.executor import Executor, LocalExecutor
from xe_forge.orbit.models import (
    ConfigAxis,
    DeterminismProfile,
    FrameworkCapabilities,
    KernelRecord,
    PatchPoint,
    WorkloadMeasurement,
    WorkloadSpec,
)

# Environment variables that steer any torch workload's backend selection, and so are
# a legitimate config action space even with no framework knowledge at all.
_GENERIC_AXES = (
    ConfigAxis(
        name="inductor_cache_dir",
        env_var="TORCHINDUCTOR_CACHE_DIR",
        description="Pin Inductor codegen cache so extraction is reproducible",
    ),
    ConfigAxis(
        name="fx_graph_cache",
        env_var="TORCHINDUCTOR_FX_GRAPH_CACHE",
        values=["0", "1"],
        description="Disable during extraction so codegen actually re-runs",
    ),
    ConfigAxis(
        name="triton_cache_dir",
        env_var="TRITON_CACHE_DIR",
        description="Pin Triton JIT cache",
    ),
    ConfigAxis(
        name="onednn_verbose",
        env_var="DNNL_VERBOSE",
        values=["0", "1"],
        description="Emit oneDNN primitive descriptors — the E4 reproducer source",
    ),
    ConfigAxis(
        name="device_selector",
        env_var="ONEAPI_DEVICE_SELECTOR",
        description="Restrict which Level Zero devices are visible",
    ),
)


class GenericTorchAdapter(BaseAdapter):
    """Works on any torch workload. Declares only what it can actually deliver."""

    name = "generic_torch"
    tier = 0
    capabilities = FrameworkCapabilities(
        metrics={"wall_time"},
        can_reset_state=False,
        can_pin_batching=False,
        can_disable_prefix_cache=False,
        can_construct_single_layer=False,
        patchable_layers=set(),
    )

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor or LocalExecutor()

    def detect(self, spec: WorkloadSpec) -> bool:
        """The universal fallback: claims any workload, at Tier 0 precision.

        Registry resolution tries higher tiers first, so returning True here does not
        shadow a real adapter — it guarantees there is always *some* adapter.
        """
        return True

    def versions(self) -> dict[str, str]:
        from xe_forge.orbit.runtime import environment

        return environment.package_versions()

    def prepare(self, spec: WorkloadSpec) -> PreparedWorkload:
        notes = [
            "Tier 0: wall-clock only. Serving metrics (TTFT/TPOT/throughput), "
            "determinism control and in-situ harness construction are unavailable — "
            "write a Tier 1 adapter for this framework to get them."
        ]
        return PreparedWorkload(spec=spec, notes=notes)

    def launch(self, spec: WorkloadSpec, executor: Executor) -> Handle:
        return Handle(spec=spec, adapter=self.name, state={"executor": executor})

    def benchmark(self, handle: Handle, load: LoadSpec) -> WorkloadMeasurement:
        executor = handle.state.get("executor") or self.executor
        runner = BenchRunner(executor=executor)
        measurement = runner.measure(
            handle.spec,
            repetitions=load.repetitions,
            profile_id=load.profile_id,
        )
        # Never report a metric we did not declare (conformance test 3).
        measurement.metrics_available = ["wall_time"]
        measurement.throughput = None
        measurement.ttft_ms = None
        measurement.tpot_ms = None
        return measurement

    def metrics_schema(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                name="wall_time",
                unit="s",
                lower_is_better=True,
                description="End-to-end process wall time",
            )
        ]

    def determinism_profile(self) -> DeterminismProfile:
        """Tier 0 can pin nothing it cannot see; it names the non-pinnable sources so
        the measurement layer can refuse ACCEPT when variance exceeds the MDE."""
        return DeterminismProfile(
            pinnable=set(),
            non_pinnable={
                "prefix_cache_reuse",
                "continuous_batching_order",
                "chunked_prefill_boundaries",
                "speculative_decoding",
                "graph_capture_warmup",
                "request_arrival_jitter",
            },
            notes=(
                "Tier 0 has no framework hooks, so no nondeterminism source can be "
                "pinned. Interleaved measurement is the only mitigation."
            ),
        )

    def dispatch_roots(self) -> list[str]:
        return ["aten", "torch.ops"]

    def provenance_hints(self) -> list[str]:
        return ["inductor", "triton", "onednn", "sycl"]

    def patch_points(self, kernel: KernelRecord) -> list[PatchPoint]:
        """P1 operator override is available whenever the kernel sits behind an op;
        it needs no framework knowledge, which is why Tier 0 can offer it."""
        if not kernel.framework_op:
            return []
        return [
            PatchPoint(
                rung="P1",
                target=kernel.framework_op,
                mechanism="torch.library operator override on the device key",
                description=(
                    "Register an implementation for the existing op, shadowing the "
                    "default. Touches nothing in the framework; revert is not "
                    "importing the module."
                ),
            )
        ]

    def config_axes(self) -> list[ConfigAxis]:
        return list(_GENERIC_AXES)

    def apply_config(self, spec: WorkloadSpec, config: dict[str, Any]) -> WorkloadSpec:
        return super().apply_config(spec, config)
