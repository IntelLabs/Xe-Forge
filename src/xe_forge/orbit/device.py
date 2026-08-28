"""
Queried device facts for the accelerator the workload runs on. Facts that cannot be
read are reported as unknown, never defaulted.
Design rationale: docs/DESIGN.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Below this EU count, occupancy advice written for discrete GPUs is misleading.
SMALL_DEVICE_EU_THRESHOLD = 64


@dataclass
class DeviceFacts:
    """Measured properties of the accelerator, as the optimizer needs to see them."""

    name: str = ""
    platform: str = ""
    driver_version: str = ""
    integrated: bool | None = None
    eu_count: int = 0
    compute_units: int = 0
    subslices: int = 0
    max_work_group_size: int = 0
    sub_group_sizes: list[int] = field(default_factory=list)
    local_mem_bytes: int = 0
    last_level_cache_bytes: int = 0
    total_memory_bytes: int = 0
    memory_bus_width: int = 0
    has_fp16: bool | None = None
    has_bf16: bool | None = None
    has_fp64: bool | None = None
    has_matrix_engine: bool | None = None
    available: bool = False

    @property
    def small(self) -> bool:
        """Whether occupancy advice written for discrete GPUs will mislead here."""
        return bool(self.eu_count) and self.eu_count < SMALL_DEVICE_EU_THRESHOLD

    def describe(self) -> str:
        """Render for an agent prompt, leading with the constraints that bind."""
        if not self.available:
            return (
                "DEVICE: unknown — no accelerator could be queried. Any advice below is "
                "untethered from the hardware and should be treated as such."
            )

        lines = [f"DEVICE: {self.name or 'unnamed'}"]
        if self.platform:
            lines.append(f"  runtime: {self.platform} (driver {self.driver_version or '?'})")

        kind = (
            "integrated" if self.integrated else "discrete" if self.integrated is not None else "?"
        )
        lines.append(
            f"  {kind}, {self.eu_count or '?'} EUs / {self.compute_units or '?'} compute units"
            + (f", {self.subslices} subslices" if self.subslices else "")
        )

        # Hard limits first: they invalidate proposals outright.
        if self.max_work_group_size:
            lines.append(
                f"  HARD LIMIT: max work-group size {self.max_work_group_size} — a block "
                f"larger than this cannot be launched as one work-group"
            )
        if self.sub_group_sizes:
            sizes = ", ".join(str(s) for s in self.sub_group_sizes)
            lines.append(f"  sub-group sizes supported: {sizes} (not a single fixed width)")
        if self.local_mem_bytes:
            lines.append(f"  SLM: {self.local_mem_bytes // 1024} KB per work-group")
        if self.last_level_cache_bytes:
            lines.append(f"  last-level cache: {self.last_level_cache_bytes // 1024} KB")
        if self.total_memory_bytes:
            shared = " (shared with the host)" if self.integrated else ""
            lines.append(f"  memory: {self.total_memory_bytes / 2**30:.1f} GiB{shared}")
        if self.memory_bus_width:
            lines.append(f"  memory bus: {self.memory_bus_width}-bit")

        caps = [
            name
            for name, present in (
                ("fp16", self.has_fp16),
                ("bf16", self.has_bf16),
                ("fp64", self.has_fp64),
                ("matrix engine (DPAS)", self.has_matrix_engine),
            )
            if present
        ]
        if caps:
            lines.append(f"  capabilities: {', '.join(caps)}")

        if self.small:
            lines.append("")
            lines.append(
                f"  NOTE: {self.eu_count} EUs is a small integrated GPU. Occupancy and "
                f"block-size advice written for discrete GPUs with hundreds of EUs does "
                f"not transfer — large work-groups oversubscribe rather than saturate, "
                f"and a narrow bus makes this device memory-bound much earlier than a "
                f"discrete part."
            )
        return "\n".join(lines)


def probe_device(index: int = 0) -> DeviceFacts:
    """Read the accelerator's real properties; unreadable ones stay at their defaults."""
    try:
        import torch
    except ImportError:
        return DeviceFacts()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return DeviceFacts()

    try:
        properties = torch.xpu.get_device_properties(index)
    except Exception:
        return DeviceFacts()

    def read(attr: str, default=0):
        value = getattr(properties, attr, default)
        return default if value is None else value

    sub_groups = read("sub_group_sizes", []) or []
    return DeviceFacts(
        name=str(read("name", "")),
        platform=str(read("platform_name", "")),
        driver_version=str(read("driver_version", "")),
        integrated=bool(read("is_integrated_gpu", False)),
        eu_count=int(read("gpu_eu_count")),
        compute_units=int(read("max_compute_units")),
        subslices=int(read("gpu_subslice_count")),
        max_work_group_size=int(read("max_work_group_size")),
        sub_group_sizes=[int(s) for s in sub_groups],
        local_mem_bytes=int(read("local_mem_size")),
        last_level_cache_bytes=int(read("last_level_cache_size")),
        total_memory_bytes=int(read("total_memory")),
        memory_bus_width=int(read("memory_bus_width")),
        has_fp16=bool(read("has_fp16", False)),
        has_bf16=bool(read("has_bfloat16_conversions", False)),
        has_fp64=bool(read("has_fp64", False)),
        has_matrix_engine=bool(read("has_subgroup_matrix_multiply_accumulate", False)),
        available=True,
    )


def launch_constraints(facts: DeviceFacts) -> list[str]:
    """Rules a proposal must satisfy on this device, stated as checkable bounds."""
    rules: list[str] = []
    if not facts.available:
        return rules
    if facts.max_work_group_size:
        rules.append(
            f"A Triton BLOCK_SIZE maps onto a work-group bounded by "
            f"{facts.max_work_group_size} work-items; exceeding it forces each work-item "
            f"to process multiple elements, raising register pressure rather than "
            f"parallelism."
        )
    if facts.sub_group_sizes and facts.eu_count:
        widest = max(facts.sub_group_sizes)
        rules.append(
            f"num_warps x sub-group width is the work-group size. With {facts.eu_count} "
            f"EUs and sub-groups of {facts.sub_group_sizes}, num_warps=16 at width "
            f"{widest} requests {16 * widest} work-items — far past what this device can "
            f"co-resident schedule."
        )
    if facts.memory_bus_width and facts.memory_bus_width <= 128:
        rules.append(
            f"The {facts.memory_bus_width}-bit bus makes bandwidth the binding constraint "
            f"for most elementwise and reduction kernels; changes that add arithmetic to "
            f"save memory traffic are favoured, and the reverse is not."
        )
    return rules
