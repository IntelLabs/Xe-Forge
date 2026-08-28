"""
Kernel catalog, gating and ranking: gate on whether the workload is GPU-bound first,
then rank deterministically (Amdahl ceiling x roofline headroom x action availability
x confidence x capped tractability). Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Mapping

from xe_forge.orbit import stats
from xe_forge.orbit.analysis.regions import detect_regions
from xe_forge.orbit.analysis.roofline import (
    NEUTRAL_HEADROOM,
    KernelCost,
    headroom_estimate_for,
)
from xe_forge.orbit.models import (
    ActionType,
    KernelCatalog,
    KernelRecord,
    ShapeObservation,
)
from xe_forge.orbit.profiling.trace import TraceEvents
from xe_forge.orbit.provenance import resolvers

# Bounds the tractability term so a cheap-to-extract kernel breaks ties but never
# outranks a kernel with materially more headroom: the CAP/FLOOR band spans at most
# 1.43x, so widening it would let extraction ease overturn real headroom.
TRACTABILITY_CAP = 1.0
TRACTABILITY_FLOOR = 0.7

# The speedup we assume a candidate could plausibly reach when computing the Amdahl
# ceiling. Deliberately modest: an optimistic default makes everything look worth
# doing, which defeats the purpose of gating.
DEFAULT_ESTIMATED_SPEEDUP = 2.0

# Actions that represent real optimization work, as opposed to "go look harder".
_PRODUCTIVE_ACTIONS = {
    ActionType.KERNEL_REWRITE,
    ActionType.KERNEL_AUTOTUNE,
    ActionType.KERNEL_TILE_SEARCH,
    ActionType.REGION_FUSION,
    ActionType.COMPILER_OPTION,
    ActionType.BACKEND_CHANGE,
    ActionType.LAYOUT_CHANGE,
    ActionType.LIBRARY_CONFIG,
}


def kernel_id(index: int, name: str) -> str:
    return f"k{index}"


def build_catalog(
    events: TraceEvents,
    run_id: str,
    *,
    gpu_busy_percent: float | None = None,
    launch_gap_total_us: float | None = None,
    minimum_detectable_effect: float = 0.0,
    estimated_speedup: float = DEFAULT_ESTIMATED_SPEEDUP,
    wall_time_us: float | None = None,
    device_name: str | None = None,
    kernel_costs: Mapping[str, KernelCost] | None = None,
    with_regions: bool = True,
) -> KernelCatalog:
    """Aggregate trace events into a ranked, gated kernel catalog.

    `gpu_busy_percent` and `launch_gap_total_us` come from unitrace when it is
    available. When it is not, GPU-busy is estimated from the trace span, and the
    catalog records that the estimate is weaker — it cannot see time lost *between*
    launches, which is exactly the signal that identifies a host-bound workload.

    `device_name` and `kernel_costs` feed the roofline term of the ranking.
    `kernel_costs` maps a runtime kernel name to its measured per-call FLOP and byte
    counts; a `torch.profiler` trace carries neither, so the common case is that it is
    absent and every kernel gets neutral headroom, recorded as unmeasured rather than
    filled in with an estimate.
    """
    aggregated: dict[str, dict] = {}
    for event in events.kernels:
        entry = aggregated.setdefault(
            event.name,
            {"calls": 0, "total_us": 0.0, "shapes": {}},
        )
        entry["calls"] += 1
        entry["total_us"] += event.duration_us
        shape = _shape_from_event(event.args)
        if shape is not None:
            entry["shapes"][shape.key] = entry["shapes"].get(shape.key, 0) + 1

    total_gpu_us = sum(e["total_us"] for e in aggregated.values())
    span_us = wall_time_us if wall_time_us is not None else events.wall_span_us

    estimated_busy = False
    if gpu_busy_percent is None:
        estimated_busy = True
        gpu_busy_percent = (total_gpu_us / span_us * 100.0) if span_us > 0 else 0.0
        gpu_busy_percent = min(100.0, gpu_busy_percent)

    gpu_busy_fraction = max(0.0, min(1.0, gpu_busy_percent / 100.0))
    host_bound_fraction = max(0.0, 1.0 - gpu_busy_fraction)

    if launch_gap_total_us is None:
        launch_gap_total_us = max(0.0, span_us - total_gpu_us) if span_us > 0 else 0.0

    records: list[KernelRecord] = []
    skipped: list[dict[str, str]] = []

    for index, (name, entry) in enumerate(
        sorted(aggregated.items(), key=lambda kv: kv[1]["total_us"], reverse=True)
    ):
        provenance = resolvers.resolve(name)
        share = (entry["total_us"] / total_gpu_us) if total_gpu_us > 0 else 0.0
        ceiling = stats.amdahl_ceiling(share, estimated_speedup, gpu_busy_fraction)

        tractability = max(
            TRACTABILITY_FLOOR,
            min(
                TRACTABILITY_CAP,
                resolvers.extraction_tractability(provenance.default_extraction),
            ),
        )
        availability = 1.0 if _has_productive_action(provenance.actions) else 0.0

        record = KernelRecord(
            id=kernel_id(index, name),
            runtime_name=name,
            demangled_name=None,
            framework_op=provenance.framework_op,
            provider=provenance.provider,
            language=provenance.language,
            build_system=provenance.build_system,
            aot=provenance.aot,
            source_file=provenance.source.file,
            source_symbol=provenance.source.symbol,
            dispatch_chain=provenance.dispatch_chain,
            calls=entry["calls"],
            total_time_us=entry["total_us"],
            avg_time_us=entry["total_us"] / entry["calls"] if entry["calls"] else 0.0,
            shapes=_rebuild_shapes(entry["shapes"]),
            actions_available=provenance.actions,
            provenance_confidence=provenance.source.confidence
            if provenance.source.confidence is not None
            else provenance.confidence,
            resolution_method=provenance.source.method,
            extraction_level=provenance.default_extraction,
            gpu_time_share=share,
            max_e2e_gain=ceiling,
            roofline_headroom=NEUTRAL_HEADROOM,
            extraction_tractability=tractability,
            priority=0.0,
        )

        # Roofline headroom: measured achieved-vs-ceiling where FLOP and byte counts
        # exist, neutral-and-flagged where they do not. >= 1.0 always; 1.0 means
        # "already at the roof" (see analysis/roofline.py for the direction).
        headroom = headroom_estimate_for(record, device_name, (kernel_costs or {}).get(name))
        record.roofline_headroom = headroom.value
        record.priority = (
            ceiling * headroom.value * availability * record.confidence_factor * tractability
        )

        reason = _skip_reason(record, minimum_detectable_effect)
        if reason:
            record.skip_reason = reason
            skipped.append(
                {
                    "id": record.id,
                    "kernel": name,
                    "gpu_percent": f"{share * 100:.1f}",
                    "reason": reason,
                }
            )

        records.append(record)

    records.sort(key=lambda r: r.priority, reverse=True)

    gating_action, gating_reason = _gate(
        records,
        gpu_busy_percent=gpu_busy_percent,
        host_bound_fraction=host_bound_fraction,
        minimum_detectable_effect=minimum_detectable_effect,
        estimated_busy=estimated_busy,
    )

    # Regions are the other optimization unit: only a region can express a
    # many-to-one replacement.
    regions = (
        detect_regions(events, records, total_gpu_time_us=total_gpu_us) if with_regions else []
    )

    catalog = KernelCatalog(
        run_id=run_id,
        kernels=records,
        regions=regions,
        gpu_busy_percent=gpu_busy_percent,
        launch_gap_total_us=launch_gap_total_us,
        host_bound_fraction=host_bound_fraction,
        minimum_detectable_effect=minimum_detectable_effect,
        total_gpu_time_us=total_gpu_us,
        total_wall_time_us=span_us,
        gating_action=gating_action,
        gating_reason=gating_reason,
        considered_but_not_attempted=skipped,
    )
    return catalog


def _gate(
    records: list[KernelRecord],
    *,
    gpu_busy_percent: float,
    host_bound_fraction: float,
    minimum_detectable_effect: float,
    estimated_busy: bool,
) -> tuple[ActionType, str]:
    """Decide whether kernel optimization is the right activity at all."""
    if not records:
        return (
            ActionType.PROFILE_MORE,
            "no device kernels observed; either the workload is entirely host-bound or "
            "the trace captured no device activity source",
        )

    qualifier = " (estimated from trace span; unitrace unavailable)" if estimated_busy else ""

    if host_bound_fraction >= 0.5:
        return (
            ActionType.HOST_OPTIMIZATION,
            f"workload is host-bound: GPU busy {gpu_busy_percent:.1f}%{qualifier}, "
            f"host-bound fraction {host_bound_fraction:.2f}. Launch overhead and "
            f"scheduler work dominate; a faster kernel cannot move end-to-end time.",
        )

    best = max(records, key=lambda r: r.max_e2e_gain)
    if minimum_detectable_effect > 0 and best.max_e2e_gain < minimum_detectable_effect:
        return (
            ActionType.NO_ACTION,
            f"no kernel clears the noise floor: best Amdahl ceiling is "
            f"{best.max_e2e_gain:.2f}% ({best.runtime_name}) against a minimum "
            f"detectable effect of {minimum_detectable_effect:.2f}%. Any gain would be "
            f"unmeasurable in this workload.",
        )

    if not any(_has_productive_action(r.actions_available) for r in records):
        return (
            ActionType.PROFILE_MORE,
            "kernels were observed but none has an available optimization action; "
            "provenance is too weak to act on",
        )

    return (
        ActionType.KERNEL_REWRITE,
        f"GPU-bound: GPU busy {gpu_busy_percent:.1f}%{qualifier}; top candidate "
        f"{best.runtime_name} has an Amdahl ceiling of {best.max_e2e_gain:.2f}%",
    )


def _skip_reason(record: KernelRecord, mde: float) -> str | None:
    """Why this kernel will not be attempted, if it will not be."""
    if not _has_productive_action(record.actions_available):
        if ActionType.PROFILE_MORE in record.actions_available:
            return "no provenance; needs more profiling before any action"
        return "no optimization action available"
    if record.confidence_factor < 0.5:
        return f"provenance confidence too low ({record.describe_confidence()})"
    if mde > 0 and record.max_e2e_gain < mde:
        return (
            f"Amdahl ceiling {record.max_e2e_gain:.2f}% is below the minimum "
            f"detectable effect {mde:.2f}%"
        )
    return None


def _has_productive_action(actions: list[ActionType]) -> bool:
    return any(a in _PRODUCTIVE_ACTIONS for a in actions)


def _shape_from_event(args: dict) -> ShapeObservation | None:
    """Recover an input shape from a trace event's recorded arguments."""
    raw = args.get("Input Dims") or args.get("input_dims")
    if not raw or not isinstance(raw, list):
        return None
    dims: dict[str, int] = {}
    for position, entry in enumerate(raw):
        if isinstance(entry, list):
            for axis, value in enumerate(entry):
                try:
                    dims[f"a{position}_d{axis}"] = int(value)
                except (TypeError, ValueError):
                    continue
    if not dims:
        return None
    types = args.get("Input type") or args.get("input_type")
    dtypes: dict[str, str] = {}
    if isinstance(types, list):
        for position, entry in enumerate(types):
            if isinstance(entry, str) and entry:
                dtypes[f"a{position}"] = entry
    return ShapeObservation(dims=dims, dtypes=dtypes, count=1)


def _rebuild_shapes(counts: dict[str, int]) -> list[ShapeObservation]:
    """Turn the aggregation map back into weighted shape observations."""
    shapes: list[ShapeObservation] = []
    for key, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        dims: dict[str, int] = {}
        dtypes: dict[str, str] = {}
        dim_part, _, dtype_part = key.partition("|")
        for token in filter(None, dim_part.split(",")):
            name, _, value = token.partition("=")
            try:
                dims[name] = int(value)
            except ValueError:
                continue
        for token in filter(None, dtype_part.split(",")):
            name, _, value = token.partition(":")
            if name:
                dtypes[name] = value
        shapes.append(ShapeObservation(dims=dims, dtypes=dtypes, count=count))
    return shapes


def format_catalog(catalog: KernelCatalog, limit: int = 20) -> str:
    """Render the catalog as the table the CLI prints."""
    lines = [
        f"{'ID':<5} {'GPU%':>6} {'Calls':>8}  {'Operator':<18} {'Provider':<12} "
        f"{'Extract':<8} {'Actions':<26} {'Conf':>5}",
        "-" * 100,
    ]
    for record in catalog.kernels[:limit]:
        actions = ",".join(a.value.replace("kernel_", "") for a in record.actions_available[:3])
        operator = (record.framework_op or record.runtime_name)[:18]
        lines.append(
            f"{record.id:<5} {record.gpu_time_share * 100:>6.1f} {record.calls:>8}  "
            f"{operator:<18} {record.provider.value:<12} "
            f"{(record.extraction_level.value if record.extraction_level else '--'):<8} "
            f"{actions:<26} {record.describe_confidence():>5}"
        )

    lines.append("")
    lines.append(
        f"GPU busy: {catalog.gpu_busy_percent:.1f}%   "
        f"launch gaps: {catalog.launch_gap_total_us:.0f}us   "
        f"MDE (e2e): {catalog.minimum_detectable_effect:.2f}%"
    )
    lines.append(f"Gate: {catalog.gating_action.value} — {catalog.gating_reason}")

    # Say whether the roofline term did any work: a reader cannot otherwise tell a
    # measured headroom from a neutral one.
    if catalog.kernels:
        measured = sum(1 for k in catalog.kernels if k.roofline_headroom != NEUTRAL_HEADROOM)
        if measured:
            lines.append(
                f"Roofline: headroom measured for {measured}/{len(catalog.kernels)} kernels"
            )
        else:
            lines.append(
                "Roofline: headroom unmeasured for every kernel (the trace carries no FLOP "
                "or byte counts); neutral 1.0 applied rather than an estimate."
            )

    if catalog.considered_but_not_attempted:
        lines.append("")
        lines.append("Considered but not attempted:")
        for item in catalog.considered_but_not_attempted[:10]:
            lines.append(f"  {item['id']} ({item['gpu_percent']}% GPU) — {item['reason']}")

    return "\n".join(lines)
