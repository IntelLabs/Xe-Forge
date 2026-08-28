"""
Roofline headroom for the ranking function (plan §18).

§18's ranking multiplies the Amdahl ceiling by `roofline_headroom(k)`:

    priority(k) = max_e2e_gain(k, s_est)
                * roofline_headroom(k)
                * action_availability(k)
                * provenance_confidence(k)
                * min(extraction_tractability(k), TRACTABILITY_CAP)

v1 filled that slot with an unspecified `estimated_headroom` fudge factor. §18 removed
it and said to use the real roofline: **measured** achieved TFLOPS and bandwidth against
the hardware ceiling. This module is that replacement, and it refuses to guess.

Direction convention — read this before changing anything here
--------------------------------------------------------------
`headroom` is a **multiplier on priority expressing how much of the hardware the kernel
is leaving on the table**, defined as::

    headroom = roofline_ceiling_tflops / achieved_tflops        (>= 1.0)

* A kernel sitting *on* the roof achieves the ceiling, so the ratio is **1.0** — no
  room, and the ranking is left untouched by this term.
* A kernel at a quarter of its ceiling scores **4.0** — four times the achievable
  performance is unclaimed, so it outranks an equally hot kernel that is already
  saturating the hardware.

Getting this backwards inverts the ranking: it would promote exactly the kernels that
have nothing left to give. If you find yourself writing `achieved / ceiling`, stop.

`NEUTRAL_HEADROOM` is 1.0, deliberately the same value as "already at the roof". When
FLOP and byte counts are unavailable — the common case, because a `torch.profiler`
trace does not carry them — the estimate is neutral *and flagged* (`measured=False`).
Neutral is the conservative end of the range: it never promotes a kernel on the
strength of data we do not have. That is the whole difference between this module and
the fudge factor it replaces.

Both FLOPs and bytes are required for a measured result. With only one of them the
roofline cannot tell whether the kernel sits under the flat compute roof or the sloped
memory roof, and picking a roof on a guess systematically *overstates* headroom. That
is the failure §18 removed, so the answer is neutral-and-say-so instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from xe_forge.orbit.models import KernelRecord

# ---------------------------------------------------------------------------
# Hardware presets.
#
# SOURCE: copied verbatim from `scripts/roofline.py`'s HARDWARE_PRESETS. That script
# is a standalone PEP-723 tool that deliberately does not import this package, so the
# constants are duplicated rather than shared. Duplication is the point: it makes
# drift visible, and `tests/orbit/test_roofline.py` parses the script and asserts the
# two tables still agree. If you change a number here, change it there.
#
# Peak figures are FP16/BF16 dense throughput and peak DRAM bandwidth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """One device's roof: a flat compute ceiling and a sloped bandwidth ceiling."""

    key: str
    name: str
    peak_tflops: float
    peak_bandwidth_gbps: float

    def ceiling_tflops(self, arithmetic_intensity: float) -> float:
        """The roofline at a given FLOP/byte: min(compute roof, bandwidth roof)."""
        return min(self.peak_tflops, arithmetic_intensity * self.peak_bandwidth_gbps / 1000.0)

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity where the two roofs meet; below it a kernel is
        memory-bound, above it compute-bound."""
        return self.peak_tflops / (self.peak_bandwidth_gbps / 1000.0)


HARDWARE_PRESETS: dict[str, Hardware] = {
    "arc-pro-b70": Hardware("arc-pro-b70", "Intel Arc Pro B70", 160.0, 608.0),
    "arc-b580": Hardware("arc-b580", "Intel Arc B580", 117.0, 456.0),
    "max-1550": Hardware("max-1550", "Intel Data Center GPU Max 1550 (PVC)", 839.0, 3276.0),
    "max-1100": Hardware("max-1100", "Intel Data Center GPU Max 1100 (PVC)", 362.0, 1228.0),
    "flex-170": Hardware("flex-170", "Intel Data Center GPU Flex 170", 137.0, 576.0),
}

# Substrings of a reported device name that identify a preset. Only real products
# appear here — an unrecognized device resolves to None and the headroom is reported
# as unmeasured, never approximated with "something close". Longest/most specific
# tokens first, because matching is first-hit.
_DEVICE_ALIASES: tuple[tuple[str, str], ...] = (
    ("arc pro b70", "arc-pro-b70"),
    ("pro b70", "arc-pro-b70"),
    ("b70", "arc-pro-b70"),
    ("arc b580", "arc-b580"),
    ("b580", "arc-b580"),
    ("max 1550", "max-1550"),
    ("1550", "max-1550"),
    ("max 1100", "max-1100"),
    ("1100", "max-1100"),
    ("flex 170", "flex-170"),
)

_PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")

# The value applied when nothing was measured. Identical to "already at the roof", so
# an unmeasured kernel is never promoted over a measured one on missing evidence.
NEUTRAL_HEADROOM = 1.0

# Ceiling on the ratio. Unbounded, a kernel at 1% of the roof would score 100x and
# overturn an order-of-magnitude difference in the Amdahl ceiling — the §11.10 language
# bias failure in a new costume, with "badly optimized and tiny" beating "dominant".
# Eight is already four times the DEFAULT_ESTIMATED_SPEEDUP the ceiling assumes.
MAX_HEADROOM = 8.0


@dataclass(frozen=True)
class KernelCost:
    """Measured FLOP and byte counts for one invocation of a kernel.

    Both are *per call*, matching `KernelRecord.avg_time_us`. A shape-derived FLOP
    formula produces per-call numbers, so this is the unit that needs no conversion at
    the call site. `None` means "not counted", which is not the same as zero.
    """

    flops_per_call: float | None = None
    bytes_per_call: float | None = None


@dataclass(frozen=True)
class HeadroomEstimate:
    """A headroom value plus the evidence for it — including its absence."""

    value: float
    measured: bool
    basis: str  # "roofline" | "unmeasured"
    reason: str
    device: str | None = None
    hardware: str | None = None
    achieved_tflops: float | None = None
    achieved_gbps: float | None = None
    ceiling_tflops: float | None = None
    arithmetic_intensity: float | None = None

    @property
    def compute_bound(self) -> bool | None:
        """True when the kernel sits right of the ridge point (under the flat roof)."""
        if not self.measured or self.arithmetic_intensity is None or self.hardware is None:
            return None
        hardware = HARDWARE_PRESETS.get(self.hardware)
        if hardware is None:
            return None
        return self.arithmetic_intensity >= hardware.ridge_point


def normalize_device_name(device_name: str) -> str:
    """Lower-case, strip the `(R)`/`(TM)` noise, collapse separators to spaces."""
    return _PUNCTUATION_RE.sub(" ", device_name.lower()).strip()


def resolve_hardware(device_name: str | None) -> Hardware | None:
    """Map a reported device name onto a preset, or None if we do not know it.

    Returning None is a real answer. Substituting the "nearest" GPU would put a wrong
    roof under every headroom number computed for that run.
    """
    if not device_name:
        return None
    normalized = normalize_device_name(device_name)
    if not normalized:
        return None
    key = normalized.replace(" ", "-")
    if key in HARDWARE_PRESETS:
        return HARDWARE_PRESETS[key]
    for token, preset in _DEVICE_ALIASES:
        if token in normalized:
            return HARDWARE_PRESETS[preset]
    return None


def unmeasured(reason: str, device: str | None = None) -> HeadroomEstimate:
    """Neutral headroom with the reason it could not be measured recorded."""
    return HeadroomEstimate(
        value=NEUTRAL_HEADROOM,
        measured=False,
        basis="unmeasured",
        reason=reason,
        device=device,
    )


def estimate_headroom(
    *,
    time_us: float,
    device_name: str | None = None,
    flops: float | None = None,
    bytes_moved: float | None = None,
) -> HeadroomEstimate:
    """Achieved performance against the hardware roof, as a priority multiplier.

    `flops` and `bytes_moved` are per invocation and `time_us` is that invocation's
    duration; the ratio is scale-free, so totals work equally well as long as both come
    from the same window. Returns >= 1.0: 1.0 means the kernel is at the roof (no room),
    larger means that much of the hardware is unclaimed. See the module docstring for
    why the direction is this way round.
    """
    device = device_name
    hardware = resolve_hardware(device_name)
    if hardware is None:
        if not device_name:
            return unmeasured("no device name supplied; no roof to measure against", device)
        return unmeasured(
            f"unknown device {device_name!r}: no roofline preset "
            f"(known: {', '.join(sorted(HARDWARE_PRESETS))})",
            device,
        )

    if time_us is None or time_us <= 0:
        return unmeasured("kernel duration is zero or unknown", device)

    if flops is None or bytes_moved is None:
        missing = "FLOP and byte counts" if flops is None and bytes_moved is None else None
        if missing is None:
            missing = "byte counts" if bytes_moved is None else "FLOP counts"
        return unmeasured(
            f"{missing} unavailable (the trace does not carry them); the roofline needs "
            f"both to know which roof applies, and choosing one on a guess overstates "
            f"headroom — see plan §18",
            device,
        )

    if flops <= 0 or bytes_moved <= 0:
        return unmeasured("FLOP or byte count is non-positive; cannot place on the roof", device)

    # Written out rather than folded so the units stay checkable:
    #   FLOP / us  -> FLOP/s (x1e6) -> TFLOP/s (/1e12)
    #   byte / us  -> byte/s (x1e6) -> GB/s    (/1e9)
    achieved_tflops = (flops / time_us) * 1e6 / 1e12
    achieved_gbps = (bytes_moved / time_us) * 1e6 / 1e9
    arithmetic_intensity = flops / bytes_moved
    ceiling = hardware.ceiling_tflops(arithmetic_intensity)

    if achieved_tflops <= 0:
        return unmeasured("achieved throughput computed as zero", device)

    ratio = ceiling / achieved_tflops
    if ratio < 1.0:
        # Above the roof is physically impossible, so the inputs are wrong: a bad FLOP
        # formula, a mis-scaled byte count, or the wrong device. Clamp to "no room"
        # rather than inverting the term, and say so.
        return HeadroomEstimate(
            value=NEUTRAL_HEADROOM,
            measured=True,
            basis="roofline",
            reason=(
                f"achieved {achieved_tflops:.1f} TFLOPS exceeds the {hardware.name} roof of "
                f"{ceiling:.1f} TFLOPS at AI {arithmetic_intensity:.1f}; the FLOP/byte counts "
                f"or the device are wrong. Clamped to neutral rather than trusted."
            ),
            device=device,
            hardware=hardware.key,
            achieved_tflops=achieved_tflops,
            achieved_gbps=achieved_gbps,
            ceiling_tflops=ceiling,
            arithmetic_intensity=arithmetic_intensity,
        )

    value = min(MAX_HEADROOM, ratio)
    capped = " (capped)" if ratio > MAX_HEADROOM else ""
    bound = "compute-bound" if arithmetic_intensity >= hardware.ridge_point else "memory-bound"
    return HeadroomEstimate(
        value=value,
        measured=True,
        basis="roofline",
        reason=(
            f"{achieved_tflops:.1f} of {ceiling:.1f} TFLOPS on {hardware.name} "
            f"(AI {arithmetic_intensity:.1f} FLOP/byte, {bound}, "
            f"{achieved_gbps:.0f} GB/s): {value:.2f}x unclaimed{capped}"
        ),
        device=device,
        hardware=hardware.key,
        achieved_tflops=achieved_tflops,
        achieved_gbps=achieved_gbps,
        ceiling_tflops=ceiling,
        arithmetic_intensity=arithmetic_intensity,
    )


def headroom_for(
    kernel: KernelRecord,
    device_name: str | None,
    flops: float | None = None,
    bytes_moved: float | None = None,
) -> float:
    """`estimate_headroom` for one catalogued kernel, as the bare ranking multiplier.

    Uses the kernel's average per-call duration, so `flops` and `bytes_moved` are
    per-call counts. Returns `NEUTRAL_HEADROOM` whenever the roofline cannot be
    computed; call `estimate_headroom` directly when the caller needs to know *why*.
    """
    return estimate_headroom(
        time_us=_per_call_time_us(kernel),
        device_name=device_name,
        flops=flops,
        bytes_moved=bytes_moved,
    ).value


def headroom_estimate_for(
    kernel: KernelRecord,
    device_name: str | None,
    cost: KernelCost | None = None,
) -> HeadroomEstimate:
    """The full estimate for a kernel, including whether it was measured at all."""
    return estimate_headroom(
        time_us=_per_call_time_us(kernel),
        device_name=device_name,
        flops=cost.flops_per_call if cost else None,
        bytes_moved=cost.bytes_per_call if cost else None,
    )


def _per_call_time_us(kernel: KernelRecord) -> float:
    if kernel.avg_time_us > 0:
        return kernel.avg_time_us
    if kernel.calls > 0:
        return kernel.total_time_us / kernel.calls
    return 0.0


def format_headroom(estimate: HeadroomEstimate) -> str:
    """One line describing a headroom estimate, measured or not."""
    if not estimate.measured:
        return f"headroom {estimate.value:.2f}x (unmeasured) — {estimate.reason}"
    return f"headroom {estimate.value:.2f}x — {estimate.reason}"


def list_presets() -> str:
    """The preset table, for `--list-hardware`-style output."""
    lines = [f"{'Preset':<14} {'Device':<42} {'TFLOPS':>8} {'GB/s':>8} {'Ridge':>8}"]
    for key in sorted(HARDWARE_PRESETS):
        hardware = HARDWARE_PRESETS[key]
        lines.append(
            f"{key:<14} {hardware.name:<42} {hardware.peak_tflops:>8g} "
            f"{hardware.peak_bandwidth_gbps:>8g} {hardware.ridge_point:>8.1f}"
        )
    return "\n".join(lines)
