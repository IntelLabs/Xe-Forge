"""
Typed data models for Xe-Orbit. Every persisted artifact is one of these models and
carries a ``schema_version``; dictionary-passing between subsystems is not allowed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Bumped when any artifact model changes shape incompatibly. Stored in every
# artifact; `xe_forge.orbit.artifacts` refuses to load a mismatched major version.
SCHEMA_VERSION = "1.1"


class Artifact(BaseModel):
    """Base for anything Orbit persists. Carries the schema version and a stamp."""

    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(StrEnum):
    """What can be done about a kernel or region. An opaque library GEMM has no
    editable source but still supports fusion, backend and layout actions."""

    KERNEL_REWRITE = "kernel_rewrite"
    KERNEL_AUTOTUNE = "kernel_autotune"
    KERNEL_TILE_SEARCH = "kernel_tile_search"
    REGION_FUSION = "region_fusion"
    LAYOUT_CHANGE = "layout_change"
    BACKEND_CHANGE = "backend_change"
    LIBRARY_CONFIG = "library_config"
    CONFIG_CHANGE = "config_change"
    COMPILER_OPTION = "compiler_option"
    HOST_OPTIMIZATION = "host_optimization"
    GRAPH_CAPTURE = "graph_capture"
    PROFILE_MORE = "profile_more"
    NO_ACTION = "no_action"


class ExtractionLevel(StrEnum):
    """The extraction ladder. E0 is in-place, E4 is opaque."""

    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"


class Provider(StrEnum):
    """Who produced the kernel that ran."""

    INDUCTOR = "inductor"
    TRITON = "triton"
    ONEDNN = "onednn"
    ONEMKL = "onemkl"
    SYCL = "sycl"
    IPEX = "ipex"
    CUSTOM = "custom"
    # A runtime memory operation (copy, fill), not a kernel — distinct from UNKNOWN's
    # "we could not attribute this".
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class KernelLanguage(StrEnum):
    """Source language of a kernel."""

    TRITON = "triton"
    SYCL = "sycl"
    SYCL_TLA = "sycl_tla"
    CPP = "cpp"
    OPAQUE = "opaque"


class Decision(StrEnum):
    """Accept/reject arithmetic outcomes. INCONCLUSIVE is not REJECT."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class WorkloadSpec(Artifact):
    command: list[str]
    cwd: Path = Field(default_factory=Path.cwd)
    env: dict[str, str] = Field(default_factory=dict)
    framework: str | None = None
    warmup_iterations: int = 1
    repetitions: int = 5
    timeout_s: float = 1800.0

    def model_post_init(self, _context: Any) -> None:
        if not self.command:
            raise ValueError("WorkloadSpec.command must not be empty")


class EnvironmentInfo(Artifact):
    """Everything needed to decide whether two runs are comparable."""

    python_version: str = ""
    platform: str = ""
    packages: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    device_name: str | None = None
    device_count: int = 0
    driver_version: str | None = None
    env_pins: dict[str, str] = Field(default_factory=dict)
    frequency_locked: bool = False
    clock_samples: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class MetricEstimate(BaseModel):
    """A measured quantity with an interval. Never a bare float."""

    mean: float
    stdev: float = 0.0
    n: int = 1
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    samples: list[float] = Field(default_factory=list)
    unit: str = ""


class ShapeObservation(BaseModel):
    """One observed invocation shape and how often it occurred."""

    dims: dict[str, int] = Field(default_factory=dict)
    dtypes: dict[str, str] = Field(default_factory=dict)
    count: int = 1
    profile_id: str | None = None

    @property
    def key(self) -> str:
        dims = ",".join(f"{k}={v}" for k, v in sorted(self.dims.items()))
        dtypes = ",".join(f"{k}:{v}" for k, v in sorted(self.dtypes.items()))
        return f"{dims}|{dtypes}"


# ---------------------------------------------------------------------------
# Bundle contract
# ---------------------------------------------------------------------------


class LaunchRecord(BaseModel):
    """One intercepted launch. The ground truth extraction is built from."""

    fq_name: str
    source_file: str | None = None
    source_line: int | None = None
    grid: list[int] = Field(default_factory=list)
    num_warps: int | None = None
    num_stages: int | None = None
    constexprs: dict[str, Any] = Field(default_factory=dict)
    specialization: dict[str, Any] = Field(default_factory=dict)
    arg_order: list[str] = Field(default_factory=list)
    selected_autotune_config: dict[str, Any] | None = None
    compiled_metadata: dict[str, Any] = Field(default_factory=dict)
    call_index: int = 0


class BuildRecipe(BaseModel):
    """How to rebuild a compiled-language kernel standalone."""

    compiler: str
    flags: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    defines: dict[str, str] = Field(default_factory=dict)
    link: list[str] = Field(default_factory=list)
    entry_symbol: str = ""
    # The concrete template arguments the workload actually used. Two instantiations
    # share an entry symbol but are different code; recording only the symbol can
    # rebuild the wrong specialization.
    instantiation: str = ""
    aot_target: str | None = None


class ExtractionCheck(BaseModel):
    """Result of proving a bundle is the kernel that actually ran."""

    verified: bool = False
    identity_match: bool | None = None
    launch_match: bool | None = None
    output_match: bool | None = None
    isolated_import: bool | None = None
    mutation_detected: bool | None = None
    failures: list[str] = Field(default_factory=list)


class CapturedInvocation(Artifact):
    """Real tensors captured from the running workload."""

    kernel_id: str
    call_index: int = 0
    tensors: list[str] = Field(default_factory=list)
    scalars: dict[str, Any] = Field(default_factory=dict)
    dtype_map: dict[str, str] = Field(default_factory=dict)
    shape_map: dict[str, list[int]] = Field(default_factory=dict)
    stride_map: dict[str, list[int]] = Field(default_factory=dict)
    contiguous_map: dict[str, bool] = Field(default_factory=dict)
    output_reference: str | None = None
    data_deps: list[str] = Field(default_factory=list)


class KernelBundle(Artifact):
    """Everything extraction produces, regardless of source or language."""

    kernel_id: str
    extraction_level: ExtractionLevel = ExtractionLevel.E0
    language: KernelLanguage = KernelLanguage.OPAQUE
    entrypoint: str = ""
    primary_source: str | None = None
    closure: list[str] = Field(default_factory=list)
    data_deps: list[str] = Field(default_factory=list)
    launch: LaunchRecord | None = None
    build: BuildRecipe | None = None
    inputs: CapturedInvocation | None = None
    dispatch_chain: list[str] = Field(default_factory=list)
    env_pins: dict[str, str] = Field(default_factory=dict)
    verification: ExtractionCheck = Field(default_factory=ExtractionCheck)
    downgrade_reason: str | None = None


# ---------------------------------------------------------------------------
# Kernels and regions
# ---------------------------------------------------------------------------


class ResolutionMethod(StrEnum):
    """How a kernel's source location was decided, best evidence first.

    Recorded so a resolution is auditable: a path alone cannot be reviewed.
    """

    # The build system named the translation unit. Authoritative.
    BUILD_GRAPH = "build_graph"
    # An exact match in a symbol index over the checked-out tree.
    SYMBOL_INDEX = "symbol_index"
    # Matched by filename or identifier pattern rather than by an exact symbol.
    NAME_MATCH = "name_match"
    # A RepoAgent read the tree and named the file. Runs last — later, not more
    # trustworthy.
    AGENT = "agent"
    UNRESOLVED = "unresolved"


# Tiers that are exact: they either resolve correctly or produce nothing at all.
DETERMINISTIC_METHODS = frozenset({ResolutionMethod.BUILD_GRAPH, ResolutionMethod.SYMBOL_INDEX})


class SourceLocation(BaseModel):
    file: str | None = None
    line: int | None = None
    symbol: str | None = None
    # `None` for the deterministic tiers, which are either right or silent; a float
    # always means "someone estimated this".
    confidence: float | None = None
    method: ResolutionMethod = ResolutionMethod.UNRESOLVED
    candidates: list[str] = Field(default_factory=list)
    # What an agent overrode, kept so the override is reviewable and reversible.
    previous_file: str | None = None
    previous_method: ResolutionMethod | None = None

    @property
    def deterministic(self) -> bool:
        return self.method in DETERMINISTIC_METHODS

    @property
    def resolved(self) -> bool:
        return bool(self.file) and self.method is not ResolutionMethod.UNRESOLVED

    def describe_confidence(self) -> str:
        """Render confidence without implying an estimate where none was made."""
        if self.deterministic:
            return "exact"
        if self.confidence is None:
            return "—"
        return f"{self.confidence:.2f}"


class KernelRecord(BaseModel):
    id: str
    runtime_name: str
    demangled_name: str | None = None
    framework_op: str | None = None
    graph_node: str | None = None
    provider: Provider = Provider.UNKNOWN
    language: KernelLanguage | None = None
    build_system: str | None = None
    aot: bool | None = None
    source_file: str | None = None
    source_symbol: str | None = None
    dispatch_chain: list[str] = Field(default_factory=list)
    calls: int = 0
    total_time_us: float = 0.0
    avg_time_us: float = 0.0
    shapes: list[ShapeObservation] = Field(default_factory=list)
    actions_available: list[ActionType] = Field(default_factory=list)
    # Mirrors `SourceLocation`: `None` means no one estimated anything (deterministic
    # hit, or nothing resolved — `resolution_method` says which); a float always means
    # "someone estimated this".
    provenance_confidence: float | None = None
    # How the source location was decided, persisted so the catalog can be reviewed.
    resolution_method: ResolutionMethod = ResolutionMethod.UNRESOLVED
    extraction_level: ExtractionLevel | None = None
    bundle: str | None = None
    captured_inputs: str | None = None

    # Derived by the catalog stage; present so the ranking is auditable.
    gpu_time_share: float = 0.0
    max_e2e_gain: float = 0.0
    roofline_headroom: float = 1.0
    extraction_tractability: float = 0.0
    priority: float = 0.0
    skip_reason: str | None = None

    @property
    def confidence_factor(self) -> float:
        """Confidence as a ranking multiplier.

        A deterministic tier contributes 1.0; an estimated tier its estimate; no
        estimate contributes the conservative floor, so an unattributed kernel
        cannot outrank an attributed one by omitting a number.
        """
        if self.resolution_method in DETERMINISTIC_METHODS:
            return 1.0
        if self.provenance_confidence is not None:
            return self.provenance_confidence
        return 0.2

    def describe_confidence(self) -> str:
        """Render confidence without implying an estimate where none was made."""
        if self.resolution_method in DETERMINISTIC_METHODS:
            return "exact"
        if self.provenance_confidence is None:
            return "—"
        return f"{self.provenance_confidence:.2f}"


class TensorInfo(BaseModel):
    name: str
    shape: list[int] = Field(default_factory=list)
    dtype: str = ""
    bytes: int = 0


class RegionRecord(BaseModel):
    """A multi-kernel optimization unit. Xe-Fuse's input."""

    id: str
    kernel_ids: list[str] = Field(default_factory=list)
    aten_ops: list[str] = Field(default_factory=list)
    producer_consumer_edges: list[tuple[str, str]] = Field(default_factory=list)
    intermediate_tensors: list[TensorInfo] = Field(default_factory=list)
    combined_time_us: float = 0.0
    fusion_pattern: str | None = None
    actions_available: list[ActionType] = Field(default_factory=list)
    gpu_time_share: float = 0.0


# ---------------------------------------------------------------------------
# Workload measurement
# ---------------------------------------------------------------------------


class WorkloadMeasurement(Artifact):
    wall_time: MetricEstimate
    throughput: MetricEstimate | None = None
    ttft_ms: MetricEstimate | None = None
    tpot_ms: MetricEstimate | None = None
    gpu_busy_percent: float = 0.0
    launch_gap_total_us: float = 0.0
    host_bound_fraction: float = 0.0
    minimum_detectable_effect: float = 0.0
    frequency_locked: bool = False
    clock_samples: list[float] = Field(default_factory=list)
    profile_id: str | None = None
    metrics_available: list[str] = Field(default_factory=list)


class KernelCatalog(Artifact):
    """Output of the `kernels` stage: ranked kernels plus the gating verdict."""

    run_id: str
    kernels: list[KernelRecord] = Field(default_factory=list)
    regions: list[RegionRecord] = Field(default_factory=list)
    gpu_busy_percent: float = 0.0
    launch_gap_total_us: float = 0.0
    host_bound_fraction: float = 0.0
    minimum_detectable_effect: float = 0.0
    total_gpu_time_us: float = 0.0
    total_wall_time_us: float = 0.0
    gating_action: ActionType = ActionType.NO_ACTION
    gating_reason: str = ""
    considered_but_not_attempted: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Framework adapters
# ---------------------------------------------------------------------------


class FrameworkCapabilities(BaseModel):
    """Declared, never assumed."""

    metrics: set[str] = Field(default_factory=lambda: {"wall_time"})
    can_reset_state: bool = False
    can_pin_batching: bool = False
    can_disable_prefix_cache: bool = False
    can_construct_single_layer: bool = False
    patchable_layers: set[str] = Field(default_factory=set)

    # Whether the framework runs its device work in the calling process. False means
    # an in-process `torch.profiler` sees nothing — the adapter must expose its own
    # profiling hook (`profile_hook` below).
    profiles_in_process: bool = True

    # Framework-specific mechanism that captures device activity when
    # `profiles_in_process` is False (for vLLM, `VLLM_TORCH_PROFILER_DIR` plus
    # `start_profile()`/`stop_profile()`). Empty when none exists — device
    # attribution is then unavailable for that framework.
    profile_hook: str = ""


class DeterminismProfile(BaseModel):
    """Which nondeterminism sources an adapter can pin, and which it cannot."""

    pinnable: set[str] = Field(default_factory=set)
    non_pinnable: set[str] = Field(default_factory=set)
    active_non_pinnable: set[str] = Field(default_factory=set)
    notes: str = ""


class ConfigAxis(BaseModel):
    name: str
    values: list[Any] = Field(default_factory=list)
    env_var: str | None = None
    description: str = ""


class PatchPoint(BaseModel):
    """Where an optimized kernel can be reinserted."""

    rung: str
    target: str
    mechanism: str = ""
    description: str = ""


class QualityResult(BaseModel):
    passed: bool = False
    token_exact: bool | None = None
    max_logit_deviation: float | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Serving profiles
# ---------------------------------------------------------------------------


class ServingProfile(BaseModel):
    id: str
    model: str = ""
    dtype: str = "bf16"
    quantization: str | None = None
    tp: int = 1
    ep: int | None = None
    isl: int = 1024
    osl: int = 256
    concurrency: int = 1
    prefill_decode_ratio: float | None = None
    attention_backend: str | None = None
    weight: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkloadMatrix(Artifact):
    profiles: list[ServingProfile] = Field(default_factory=list)

    def normalized_weights(self) -> dict[str, float]:
        total = sum(p.weight for p in self.profiles)
        if total <= 0:
            raise ValueError("WorkloadMatrix weights must sum to something positive")
        return {p.id: p.weight / total for p in self.profiles}


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class ComparisonOutcome(Artifact):
    """The result of comparing a candidate against a baseline."""

    decision: Decision = Decision.INCONCLUSIVE
    metric: str = "wall_time"
    baseline: MetricEstimate | None = None
    candidate: MetricEstimate | None = None
    delta_percent: float = 0.0
    delta_ci95_low: float = 0.0
    delta_ci95_high: float = 0.0
    minimum_detectable_effect: float = 0.0
    reason: str = ""
    per_profile: dict[str, str] = Field(default_factory=dict)


class RunManifest(Artifact):
    """Identity of one Orbit run."""

    run_id: str
    workload: WorkloadSpec | None = None
    environment: EnvironmentInfo | None = None
    framework: str | None = None
    adapter: str | None = None
    adapter_tier: int = 0
    stages_completed: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
