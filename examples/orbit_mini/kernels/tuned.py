"""Tuned-config lookup — the workload's non-code dependency (plan §12.8, §15.2).

DELIBERATE TRAP. ``tuned_configs.json`` sits next to this module and is read at
launch time, keyed by device name. There is **no in-code default**: if the file
is missing, malformed, or missing the ``default`` entry, the launch wrappers
raise. That is deliberate, and it is what makes the §12.12 step-5 check real —

    "remove each declared data file in turn; each removal must produce a
     failure. A data dep that can be deleted without effect was not actually a
     dependency."

An extractor that reconstructs this kernel from synthetic inputs and forgets the
JSON gets a clean crash, which is the good outcome. An extractor that silently
substitutes defaults gets plausible numbers from the wrong configuration, which
is the outcome the plan is written to prevent.

The device-name keying is the vLLM pattern (``E=...,N=...,device_name=...json``)
in miniature: the *same* source file behaves differently on two machines, so a
bundle that captures the code but not the data is not reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

#: The data dependency itself. Declared as a module-level path so extraction can
#: find it without executing anything (§12.8 wants these copied, not regenerated).
TUNED_CONFIG_PATH: Path = Path(__file__).resolve().parent / "tuned_configs.json"

#: Every key a device entry must provide. Missing keys are an error, not a default.
REQUIRED_FIELDS: tuple[str, ...] = (
    "block_n",
    "num_warps",
    "num_stages",
    "rms_eps",
    "clamp_limit",
    "prefer_sycl_op",
)


class TunedConfigError(RuntimeError):
    """Raised when the tuned-config data dependency is missing or unusable."""


@dataclass(frozen=True)
class TunedEntry:
    """One device's tuned entry, as read from ``tuned_configs.json``."""

    device_key: str
    block_n: int
    num_warps: int
    num_stages: int
    rms_eps: float
    clamp_limit: float
    prefer_sycl_op: bool

    def describe(self) -> str:
        return (
            f"{self.device_key}: block_n={self.block_n} num_warps={self.num_warps} "
            f"num_stages={self.num_stages} rms_eps={self.rms_eps:g} "
            f"clamp={self.clamp_limit:g}"
        )


@lru_cache(maxsize=1)
def _load_table() -> dict[str, Any]:
    """Read and validate ``tuned_configs.json``.

    Cached, but the cache is on the *parsed* table only — the file is still a
    declared dependency of the bundle, and §12.12 removes it to prove that.
    """
    try:
        raw = TUNED_CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # the §12.12 step-5 failure mode
        raise TunedConfigError(
            f"tuned config data dependency missing: {TUNED_CONFIG_PATH}. "
            "orbit_mini has no in-code fallback for this on purpose (plan §12.8)."
        ) from exc

    try:
        table = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TunedConfigError(f"tuned config is not valid JSON: {TUNED_CONFIG_PATH}") from exc

    devices = table.get("devices")
    if not isinstance(devices, dict) or "default" not in devices:
        raise TunedConfigError(
            f"tuned config {TUNED_CONFIG_PATH} has no 'devices.default' entry; "
            "refusing to guess a configuration."
        )
    return devices


def clear_cache() -> None:
    """Drop the parsed-table cache. Used by the data-dependency check."""
    _load_table.cache_clear()


def device_key(device: torch.device | str) -> str:
    """Map a torch device onto a key in ``tuned_configs.json``.

    Uses the vendor device *name*, not the device type, because that is what the
    file is keyed by and what makes the dependency machine-specific.
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        return "cpu"
    index = 0 if dev.index is None else dev.index
    if dev.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(index)
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(index)
    return dev.type


def lookup(device: torch.device | str) -> TunedEntry:
    """Resolve the tuned entry for ``device``, falling back to ``default``.

    Falling back to the ``default`` *entry in the file* is fine. Falling back to
    a default baked into the code is not, and there isn't one.
    """
    devices = _load_table()
    key = device_key(device)
    entry = devices.get(key)
    resolved_key = key
    if entry is None:
        entry = devices["default"]
        resolved_key = f"default (no entry for {key!r})"

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise TunedConfigError(f"tuned config entry {key!r} is missing required fields: {missing}")

    return TunedEntry(
        device_key=resolved_key,
        block_n=int(entry["block_n"]),
        num_warps=int(entry["num_warps"]),
        num_stages=int(entry["num_stages"]),
        rms_eps=float(entry["rms_eps"]),
        clamp_limit=float(entry["clamp_limit"]),
        prefer_sycl_op=bool(entry["prefer_sycl_op"]),
    )
