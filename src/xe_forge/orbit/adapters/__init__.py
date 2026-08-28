"""
Adapter registry (plan §10.3).

Adapters are resolved by `detect()` and registered through the `xe_orbit.frameworks`
entry-point group, so internal or proprietary frameworks can plug in without forking
the repository. Built-in adapters are registered directly; third-party ones are
discovered from installed distributions.

Resolution tries the highest tier first, so `GenericTorchAdapter` — which claims every
workload — is the guaranteed fallback rather than a shadow over real adapters.
"""

from __future__ import annotations

import logging
from importlib import metadata

from xe_forge.orbit.adapters.base import (
    AdapterError,
    BaseAdapter,
    FrameworkAdapter,
    Handle,
    LoadSpec,
    MetricSpec,
    PreparedWorkload,
)
from xe_forge.orbit.adapters.generic_torch import GenericTorchAdapter
from xe_forge.orbit.adapters.sglang import SGLangAdapter
from xe_forge.orbit.adapters.vllm import VLLMAdapter
from xe_forge.orbit.models import WorkloadSpec

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "xe_orbit.frameworks"

# Built-in adapters. v0.1 ships Tier 0 alongside the first Tier 1 adapter on purpose
# (§10.9): building the generic path at the same time keeps one framework's shape from
# becoming the core's shape. SGLang is the scheduled v0.2 portability test — the
# second Tier 1 adapter, whose cost outside `adapters/` is a reported metric (§10.8).
_BUILTIN: dict[str, type] = {
    "generic_torch": GenericTorchAdapter,
    "vllm": VLLMAdapter,
    "sglang": SGLangAdapter,
}


def _load_entry_point_adapters() -> dict[str, type]:
    """Discover out-of-tree adapters without importing them eagerly on failure."""
    discovered: dict[str, type] = {}
    try:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        entry_points = metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[assignment]
    for entry in entry_points:
        try:
            discovered[entry.name] = entry.load()
        except Exception as exc:
            # A broken third-party adapter must not take the whole CLI down.
            logger.warning("failed to load framework adapter %r: %s", entry.name, exc)
    return discovered


def available_adapters() -> dict[str, type]:
    """Every registered adapter class, built-in and entry-point supplied."""
    adapters = dict(_BUILTIN)
    adapters.update(_load_entry_point_adapters())
    return adapters


def get_adapter(name: str, **kwargs) -> FrameworkAdapter:
    """Instantiate one adapter by name."""
    adapters = available_adapters()
    if name not in adapters:
        raise AdapterError(f"unknown framework adapter {name!r}; available: {sorted(adapters)}")
    return adapters[name](**kwargs)


def resolve_adapter(spec: WorkloadSpec, requested: str | None = None) -> FrameworkAdapter:
    """Pick the adapter for a workload.

    An explicit request wins. Otherwise the highest-tier adapter whose `detect()`
    accepts the workload is chosen, which means a real framework adapter always beats
    the Tier 0 fallback.
    """
    if requested and requested != "auto":
        return get_adapter(requested)

    candidates: list[FrameworkAdapter] = []
    for name, cls in available_adapters().items():
        try:
            adapter = cls()
            if adapter.detect(spec):
                candidates.append(adapter)
        except Exception as exc:
            logger.warning("adapter %r failed during detection: %s", name, exc)

    if not candidates:
        return GenericTorchAdapter()

    candidates.sort(key=lambda a: getattr(a, "tier", 0), reverse=True)
    return candidates[0]


def describe_adapters() -> list[dict[str, object]]:
    """Rows for `xe-orbit frameworks` — name, tier and declared capabilities."""
    rows: list[dict[str, object]] = []
    for name, cls in sorted(available_adapters().items()):
        try:
            adapter = cls()
            caps = adapter.capabilities
            rows.append(
                {
                    "name": name,
                    "tier": getattr(adapter, "tier", 0),
                    "metrics": sorted(caps.metrics),
                    "can_reset_state": caps.can_reset_state,
                    "can_construct_single_layer": caps.can_construct_single_layer,
                    "patchable_layers": sorted(caps.patchable_layers),
                }
            )
        except Exception as exc:
            rows.append({"name": name, "tier": -1, "error": str(exc)})
    return rows


__all__ = [
    "ENTRY_POINT_GROUP",
    "AdapterError",
    "BaseAdapter",
    "FrameworkAdapter",
    "GenericTorchAdapter",
    "Handle",
    "LoadSpec",
    "MetricSpec",
    "PreparedWorkload",
    "SGLangAdapter",
    "VLLMAdapter",
    "available_adapters",
    "describe_adapters",
    "get_adapter",
    "resolve_adapter",
]
