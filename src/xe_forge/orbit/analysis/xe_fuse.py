"""
Routing a fusable region to Xe-Fuse, an external sibling project that is never a
dependency: builds the handoff description, determines availability, and reports
preset matches as candidate lists. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass

from xe_forge.orbit.models import ActionType, ExtractionLevel, RegionRecord

# The import name Xe-Fuse would publish. Nothing in this package may add xe-fuse to
# its dependency set.
XE_FUSE_MODULE = "xe_fuse"

# Region bundles are almost always E3: the region is defined by how the framework
# strings the kernels together, so the framework is the most faithful driver.
REGION_EXTRACTION_LEVEL = ExtractionLevel.E3


@dataclass(frozen=True)
class ModelPreset:
    """One transformer architecture as Xe-Fuse's presets encode it.

    This table exists to *detect* which preset an observed region matches; it is not
    authoritative over Xe-Fuse's own table, and a match should be re-checked against
    that project before anything is generated from it.
    """

    key: str
    name: str
    hidden_size: int
    kv_hidden_size: int
    ffn_hidden_size: int
    activation: str
    rope: bool = True


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset("llama-2-7b", "LLaMA 2 7B", 4096, 4096, 11008, "swiglu"),
    ModelPreset("llama-3-8b", "LLaMA 3 8B", 4096, 1024, 14336, "swiglu"),
    ModelPreset("llama-3-70b", "LLaMA 3 70B", 8192, 1024, 28672, "swiglu"),
    ModelPreset("mistral-7b", "Mistral 7B", 4096, 1024, 14336, "swiglu"),
    ModelPreset("qwen2.5-7b", "Qwen 2.5 7B", 3584, 512, 18944, "swiglu"),
    ModelPreset("gemma-2-9b", "Gemma 2 9B", 3584, 2048, 14336, "geglu"),
    ModelPreset("phi-3-mini", "Phi-3 Mini 3.8B", 3072, 3072, 8192, "swiglu"),
)

# Which activation a fusion pattern implies, used to narrow an ambiguous preset match.
_PATTERN_ACTIVATIONS: dict[str, str] = {
    "swiglu": "swiglu",
    "geglu": "geglu",
}


def xe_fuse_available() -> bool:
    """True when Xe-Fuse is usable here: importable, or present as a checkout.

    `find_spec` rather than a real import, so probing for an optional optimizer never
    executes third-party module-level code as a side effect of printing a table.
    """
    try:
        if importlib.util.find_spec(XE_FUSE_MODULE) is not None:
            return True
    except (ImportError, ValueError):
        # A broken or shadowed install is "not available", not a crash in `xe-orbit
        # regions`.
        pass
    try:
        from xe_forge.orbit.optimize.xe_fuse_executor import checkout_available

        return checkout_available()
    except Exception:
        return False


def xe_fuse_version() -> str | None:
    """The installed Xe-Fuse version, or None when it is not installed."""
    try:
        return importlib.metadata.version(XE_FUSE_MODULE.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - metadata backends vary
        return None


def observed_dims(region: RegionRecord) -> set[int]:
    """Every dimension seen on the region's intermediate tensors.

    These are the tensors fusion would eliminate, so their shapes are the ones that
    carry the architecture: a `[tokens, H]` activation names H, and an FFN intermediate
    names the FFN width.
    """
    dims: set[int] = set()
    for tensor in region.intermediate_tensors:
        dims.update(d for d in tensor.shape if d > 1)
    return dims


def match_model_preset(region: RegionRecord) -> list[ModelPreset]:
    """Presets whose hidden or FFN width appears in the region's tensors.

    Returns *every* candidate, most specific first. A single-entry list is a confident
    match; several entries mean the observed dimensions do not discriminate and the
    caller must not pick one.
    """
    dims = observed_dims(region)
    if not dims:
        return []

    activation = None
    for token, name in _PATTERN_ACTIVATIONS.items():
        if region.fusion_pattern and token in region.fusion_pattern:
            activation = name
            break

    scored: list[tuple[int, ModelPreset]] = []
    for preset in MODEL_PRESETS:
        score = 0
        if preset.hidden_size in dims:
            score += 2
        if preset.ffn_hidden_size in dims:
            score += 2
        if preset.kv_hidden_size in dims:
            score += 1
        if score == 0:
            continue
        if activation is not None and preset.activation == activation:
            score += 1
        scored.append((score, preset))

    scored.sort(key=lambda item: (-item[0], item[1].key))
    return [preset for _, preset in scored]


def requirements_for(region: RegionRecord) -> list[str]:
    """What Xe-Fuse would need in hand before it could fuse this region.

    Written as a checklist rather than prose because it is also the list of artifacts
    Orbit still has to produce; anything unticked here is integration work, not a
    detail.
    """
    eliminated = sum(t.bytes for t in region.intermediate_tensors)
    return [
        f"region bundle at {REGION_EXTRACTION_LEVEL.value} with a driver that runs the "
        f"{len(region.kernel_ids)} kernels unfused, so the fused replacement can be "
        f"compared against the sequence as a unit",
        f"the {len(region.intermediate_tensors)} intermediate tensor(s) fusion "
        f"eliminates, with shape, dtype and stride ({eliminated} bytes per pass)",
        f"producer-consumer edges: {_render_edges(region)}",
        "the matched model preset (H, H_kv, FFN width, activation, RoPE) so the "
        "sycl-tla epilogue is instantiated for the right architecture",
        "captured real tensors for the region entry point, not synthesized inputs",
        "the per-variant rtol/atol derived from the end-to-end numerical budget",
        "the accept threshold implied by the region's Amdahl ceiling",
    ]


def route_region(region: RegionRecord) -> dict[str, object]:
    """Describe the handoff of one region to Xe-Fuse.

    Returns a plain description — it never calls Xe-Fuse, and it never claims a result.
    `xe_fuse_available` is the field that matters: when it is False the route is
    `blocked` and the reason names the external project explicitly, because a report
    that quietly implies a fusion happened is worse than one that says it cannot.
    """
    available = xe_fuse_available()
    candidates = match_model_preset(region)

    if not candidates:
        preset_match: str | None = None
        preset_confidence = "none"
    elif len(candidates) == 1:
        preset_match = candidates[0].key
        preset_confidence = "unique"
    else:
        preset_match = None
        preset_confidence = "ambiguous"

    if available:
        status = "ready"
        reason = (
            f"xe_fuse {xe_fuse_version() or '(version unknown)'} is importable; the "
            f"region can be handed over through the engine seam."
        )
    else:
        status = "blocked"
        reason = (
            "xe_fuse is not importable in this environment. Xe-Fuse is an external "
            "sibling project: it is not a dependency, submodule or import "
            "of Xe-Forge, so REGION_FUSION has no executor here. The handoff below is "
            "what it would receive; nothing was executed and no speedup is implied."
        )

    return {
        "region_id": region.id,
        "action": ActionType.REGION_FUSION.value,
        "optimizer": "xe-fuse",
        "external": True,
        "xe_fuse_available": available,
        "xe_fuse_version": xe_fuse_version(),
        "status": status,
        "reason": reason,
        "fusion_pattern": region.fusion_pattern,
        "kernel_ids": list(region.kernel_ids),
        "aten_ops": list(region.aten_ops),
        "producer_consumer_edges": [list(edge) for edge in region.producer_consumer_edges],
        "combined_time_us": region.combined_time_us,
        "gpu_time_share": region.gpu_time_share,
        "extraction_level": REGION_EXTRACTION_LEVEL.value,
        "eliminated_bytes": sum(t.bytes for t in region.intermediate_tensors),
        "intermediate_tensors": [t.model_dump(mode="json") for t in region.intermediate_tensors],
        "preset_match": preset_match,
        "preset_confidence": preset_confidence,
        "preset_candidates": [preset.key for preset in candidates],
        "observed_dims": sorted(observed_dims(region)),
        "requires": requirements_for(region),
    }


def format_route(route: dict[str, object]) -> str:
    """Render one route as the block `xe-orbit fuse-region` would print."""
    kernels = "+".join(str(k) for k in _as_list(route.get("kernel_ids")))
    candidates = ", ".join(str(c) for c in _as_list(route.get("preset_candidates")))
    preset = route.get("preset_match") or f"({route.get('preset_confidence')})"
    share = float(route.get("gpu_time_share") or 0.0) * 100.0
    combined = float(route.get("combined_time_us") or 0.0)

    lines = [
        f"{route.get('region_id')}: {route.get('fusion_pattern') or '(unnamed chain)'} "
        f"-> {route.get('optimizer')}  [{route.get('status')}]",
        f"  kernels:  {kernels}  ({combined:.0f}us, {share:.1f}% of GPU time)",
        f"  preset:   {preset}" + (f"  candidates: {candidates}" if candidates else ""),
        f"  reason:   {route.get('reason')}",
        "  requires:",
    ]
    lines.extend(f"    - {item}" for item in _as_list(route.get("requires")))
    return "\n".join(lines)


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _render_edges(region: RegionRecord) -> str:
    if not region.producer_consumer_edges:
        return "none recorded"
    return ", ".join(
        f"{producer}->{consumer}" for producer, consumer in region.producer_consumer_edges
    )
