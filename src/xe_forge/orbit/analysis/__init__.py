"""Analysis: kernel cataloguing, region detection, host-bound gating and ranking."""

from xe_forge.orbit.analysis.catalog import build_catalog, format_catalog
from xe_forge.orbit.analysis.regions import (
    FUSION_PATTERNS,
    FusionPattern,
    detect_regions,
    format_regions,
)
from xe_forge.orbit.analysis.roofline import (
    HARDWARE_PRESETS,
    NEUTRAL_HEADROOM,
    HeadroomEstimate,
    KernelCost,
    estimate_headroom,
    headroom_for,
    resolve_hardware,
)
from xe_forge.orbit.analysis.xe_fuse import (
    MODEL_PRESETS,
    match_model_preset,
    route_region,
    xe_fuse_available,
)

__all__ = [
    "FUSION_PATTERNS",
    "HARDWARE_PRESETS",
    "MODEL_PRESETS",
    "NEUTRAL_HEADROOM",
    "FusionPattern",
    "HeadroomEstimate",
    "KernelCost",
    "build_catalog",
    "detect_regions",
    "estimate_headroom",
    "format_catalog",
    "format_regions",
    "headroom_for",
    "match_model_preset",
    "resolve_hardware",
    "route_region",
    "xe_fuse_available",
]
