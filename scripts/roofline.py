#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
# ]
# ///
"""Generate roofline performance plots from a CSV of kernel measurements.

This is a standalone script — it does NOT import the xe-forge package, so it
can be run on any machine that has the measurement CSV (e.g. copy results off
an XPU box and plot locally).

A roofline plot shows achieved performance (TFLOPS, y) against arithmetic
intensity (FLOP/byte, x) on log-log axes, overlaid with the hardware "roof":

    performance_ceiling(AI) = min(peak_compute_tflops,            # flat compute roof
                                   AI * peak_bandwidth_gbps / 1000) # sloped memory roof

Points to the left of the ridge are memory-bound (sit under the sloped roof);
points to the right are compute-bound (sit under the flat roof). The closer a
point is to the roof, the better the kernel uses the hardware.

------------------------------------------------------------------------------
CSV format
------------------------------------------------------------------------------
One row per measurement. Header row required. Columns (aliases accepted):

  tflops                achieved throughput in TFLOPS        (aliases: perf, gflops*)
  arithmetic_intensity  FLOP / byte                          (aliases: ai, intensity, flop_per_byte)
  series                grouping / legend label              (aliases: backend, kind)   [optional]
  label                 per-point annotation                 (aliases: name, shape)     [optional]
  pair                  id linking two points with a line    (aliases: group, kernel)   [optional]

You don't have to precompute tflops / arithmetic_intensity — if you instead
provide raw measurements they are derived:

  flop                  total floating-point ops    -> with time_us gives tflops
  time_us               kernel time in microseconds    (tflops = flop / time_us / 1e6)
  bytes                 bytes moved to/from memory  -> with flop gives arithmetic_intensity
                                                       (ai = flop / bytes)

* gflops is divided by 1000 to get tflops.

------------------------------------------------------------------------------
Examples
------------------------------------------------------------------------------
matplotlib/numpy are declared in the PEP 723 header above, so `uv run` pulls
them ephemerally — no need to install anything into the project venv.

  # plot with the Arc Pro B70 preset roof
  uv run scripts/roofline.py scripts/sample_roofline.csv \
      --hardware arc-pro-b70 --connect --annotate \
      --title "Roofline — FlashAttention Fwd" -o plots/fa.png

  # list the built-in hardware presets
  uv run scripts/roofline.py --list-hardware

  # override the roof explicitly (skips the preset table)
  uv run scripts/roofline.py results.csv \
      --peak-tflops 160 --peak-bandwidth 608 -o plots/gemm.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Hardware presets: peak FP16/BF16 dense throughput and peak DRAM bandwidth.
#
# These are the *roof* values drawn on the plot. They are best-effort published
# / measured numbers and can always be overridden with --peak-tflops /
# --peak-bandwidth. The Arc Pro B70 figures match the ceilings used in the
# repo's existing roofline plots (160 TFLOPS, 608 GB/s).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hardware:
    name: str
    peak_tflops: float  # peak compute, half precision (FP16/BF16)
    peak_bandwidth_gbps: float  # peak DRAM bandwidth, GB/s


HARDWARE_PRESETS: dict[str, Hardware] = {
    "arc-pro-b70": Hardware("Intel Arc Pro B70", peak_tflops=160.0, peak_bandwidth_gbps=608.0),
    "arc-b580": Hardware("Intel Arc B580", peak_tflops=117.0, peak_bandwidth_gbps=456.0),
    "max-1550": Hardware(
        "Intel Data Center GPU Max 1550 (PVC)", peak_tflops=839.0, peak_bandwidth_gbps=3276.0
    ),
    "max-1100": Hardware(
        "Intel Data Center GPU Max 1100 (PVC)", peak_tflops=362.0, peak_bandwidth_gbps=1228.0
    ),
    "flex-170": Hardware(
        "Intel Data Center GPU Flex 170", peak_tflops=137.0, peak_bandwidth_gbps=576.0
    ),
}


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
# Map each canonical field to the set of accepted header names (lower-cased).
COLUMN_ALIASES: dict[str, set[str]] = {
    "tflops": {"tflops", "perf", "performance", "throughput"},
    "gflops": {"gflops"},
    "arithmetic_intensity": {
        "arithmetic_intensity",
        "ai",
        "intensity",
        "flop_per_byte",
        "flops_per_byte",
    },
    "series": {"series", "backend", "kind", "category", "engine"},
    "family": {"family", "group_color", "op", "kernel_family"},
    "label": {"label", "name", "shape", "config"},
    "pair": {"pair", "group", "kernel", "link"},
    "flop": {"flop", "flops", "total_flop", "total_flops"},
    "time_us": {"time_us", "us", "time", "latency_us"},
    "bytes": {"bytes", "bytes_moved", "mem_bytes", "memory_bytes"},
}


@dataclass
class Point:
    arithmetic_intensity: float
    tflops: float | None  # None => "AI-only", plotted at its roofline ceiling
    series: str = "kernel"
    family: str = ""  # optional grouping that drives marker COLOR (series drives shape)
    label: str = ""
    pair: str = ""


def _resolve_headers(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical field name -> actual CSV header present in the file."""
    lower_to_actual = {fn.strip().lower(): fn for fn in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_actual:
                resolved[canonical] = lower_to_actual[alias]
                break
    return resolved


def _to_float(row: dict, header: str | None) -> float | None:
    if not header:
        return None
    raw = (row.get(header) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_points(csv_path: str) -> list[Point]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"error: {csv_path} is empty or has no header row")
        cols = _resolve_headers(reader.fieldnames)

        points: list[Point] = []
        for lineno, row in enumerate(reader, start=2):  # line 1 is the header
            tflops = _to_float(row, cols.get("tflops"))
            if tflops is None:
                gflops = _to_float(row, cols.get("gflops"))
                if gflops is not None:
                    tflops = gflops / 1000.0
            if tflops is None:
                flop = _to_float(row, cols.get("flop"))
                time_us = _to_float(row, cols.get("time_us"))
                if flop is not None and time_us and time_us > 0:
                    tflops = flop / time_us / 1e6

            ai = _to_float(row, cols.get("arithmetic_intensity"))
            if ai is None:
                flop = _to_float(row, cols.get("flop"))
                nbytes = _to_float(row, cols.get("bytes"))
                if flop is not None and nbytes and nbytes > 0:
                    ai = flop / nbytes

            # AI is always required; tflops is optional (AI-only points get
            # placed on the roof, showing the theoretical ceiling for that shape).
            if ai is None:
                print(
                    f"warning: skipping row {lineno} — could not determine arithmetic_intensity",
                    file=sys.stderr,
                )
                continue
            if ai <= 0 or (tflops is not None and tflops <= 0):
                print(
                    f"warning: skipping row {lineno} — non-positive value "
                    f"(tflops={tflops}, ai={ai}); log axes need > 0",
                    file=sys.stderr,
                )
                continue

            points.append(
                Point(
                    arithmetic_intensity=ai,
                    tflops=tflops,
                    series=(row.get(cols["series"], "").strip() if "series" in cols else "")
                    or "kernel",
                    family=(row.get(cols["family"], "").strip() if "family" in cols else ""),
                    label=(row.get(cols["label"], "").strip() if "label" in cols else ""),
                    pair=(row.get(cols["pair"], "").strip() if "pair" in cols else ""),
                )
            )
    if not points:
        raise SystemExit(f"error: no usable rows found in {csv_path}")
    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
@dataclass
class PlotConfig:
    title: str = "Roofline"
    output: str = "roofline.png"
    connect: bool = False
    annotate: bool = False
    key: bool = False  # number each point and list labels in a side key
    dpi: int = 200
    series_order: list[str] = field(default_factory=list)


def _resolve_hardware(args) -> Hardware:
    if args.peak_tflops is not None and args.peak_bandwidth is not None:
        return Hardware("custom", args.peak_tflops, args.peak_bandwidth)
    if args.hardware is None:
        raise SystemExit(
            "error: specify --hardware <preset> (see --list-hardware), "
            "or both --peak-tflops and --peak-bandwidth"
        )
    key = args.hardware.lower()
    if key not in HARDWARE_PRESETS:
        raise SystemExit(
            f"error: unknown hardware preset '{args.hardware}'. "
            f"Choices: {', '.join(HARDWARE_PRESETS)}"
        )
    hw = HARDWARE_PRESETS[key]
    # Allow partial override of a preset.
    if args.peak_tflops is not None:
        hw = Hardware(hw.name, args.peak_tflops, hw.peak_bandwidth_gbps)
    if args.peak_bandwidth is not None:
        hw = Hardware(hw.name, hw.peak_tflops, args.peak_bandwidth)
    return hw


def plot_roofline(points: list[Point], hw: Hardware, cfg: PlotConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless / no display needed
    import matplotlib.pyplot as plt
    import numpy as np

    # Widen the canvas when a side key is shown so labels have room.
    figsize = (10.5, 5.5) if cfg.key else (7, 5)
    fig, ax = plt.subplots(figsize=figsize)

    def ceiling(ai: float) -> float:
        """Roofline performance ceiling at a given arithmetic intensity."""
        return min(hw.peak_tflops, ai * hw.peak_bandwidth_gbps / 1000.0)

    def yval(p: Point) -> float:
        """Measured TFLOPS, or the roof ceiling for AI-only points."""
        return p.tflops if p.tflops is not None else ceiling(p.arithmetic_intensity)

    ais = [p.arithmetic_intensity for p in points]
    ys = [yval(p) for p in points]

    # --- axis ranges with a little padding (log space) ---
    ridge_ai = hw.peak_tflops / (hw.peak_bandwidth_gbps / 1000.0)  # AI where roofs meet
    # Gentle left padding so the axis starts just below the left-most point
    # (a tighter factor than the other edges, which keep the roomier /3 ·3 pad).
    x_min = min(min(ais), ridge_ai) / 1.4
    x_max = max(max(ais), ridge_ai) * 1.8
    y_min = min(min(ys), hw.peak_tflops) / 3.0
    y_max = hw.peak_tflops * 1.6

    # --- the roof itself ---
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    roof = np.minimum(hw.peak_tflops, xs * hw.peak_bandwidth_gbps / 1000.0)
    # Shade the region above the roof light gray: no kernel can exceed the
    # roofline, so everything above it is physically unattainable.
    ax.fill_between(xs, roof, y_max, color="lightgray", alpha=0.4, zorder=0)
    ax.plot(xs, roof, color="black", linewidth=1.6, zorder=5)

    # Roof annotations: compute ceiling (flat) and bandwidth ceiling (slope).
    ax.annotate(
        f"{hw.peak_tflops:g} TFLOPS (peak)",
        xy=(x_max, hw.peak_tflops),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
        style="italic",
    )
    # Place the bandwidth label on the sloped part, a bit left of the ridge.
    # Clamp into the visible window so a tight x_min can't push it off-screen:
    # keep it left of the ridge but no further left than the axis start.
    bw_label_x = max(ridge_ai / 6.0, x_min * 1.15)
    if bw_label_x < ridge_ai:
        bw_label_y = bw_label_x * hw.peak_bandwidth_gbps / 1000.0
        ax.annotate(
            f"{hw.peak_bandwidth_gbps:g} GB/s",
            xy=(bw_label_x, bw_label_y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color="dimgray",
            style="italic",
            rotation=23,
            rotation_mode="anchor",
        )

    # --- group points by series for coloring/legend ---
    series_seen: list[str] = []
    for p in points:
        if p.series not in series_seen:
            series_seen.append(p.series)
    if cfg.series_order:
        ordered = [s for s in cfg.series_order if s in series_seen]
        ordered += [s for s in series_seen if s not in ordered]
        series_seen = ordered

    # `family` (if present) is an independent grouping that drives COLOR, while
    # `series` drives MARKER SHAPE. This lets one point encode two facts at once
    # — e.g. color = kernel family (BatchedMoE/FusedMoE/UnifiedAttention), shape
    # = Original vs Optimized. With no family column we fall back to the old
    # behavior: series alone drives both color and shape.
    families_seen: list[str] = []
    for p in points:
        if p.family and p.family not in families_seen:
            families_seen.append(p.family)
    use_family = bool(families_seen)

    cmap = plt.get_cmap("tab10")
    markers = ["o", "*", "D", "s", "^", "v", "P", "X"]
    if use_family:
        color_of = {f: cmap(i % 10) for i, f in enumerate(families_seen)}
        marker_of = {s: markers[i % len(markers)] for i, s in enumerate(series_seen)}

        def point_color(p: Point):
            return color_of[p.family]

        def point_marker(p: Point) -> str:
            return marker_of[p.series]
    else:
        style = {s: (cmap(i % 10), markers[i % len(markers)]) for i, s in enumerate(series_seen)}

        def point_color(p: Point):
            return style[p.series][0]

        def point_marker(p: Point) -> str:
            return style[p.series][1]

    # --- optional connector arrows between points sharing a `pair` id ---
    # The arrow points from the baseline ("Original") to the "Optimized" point,
    # so direction reads as "where the optimization moved the kernel".
    if cfg.connect:
        from collections import defaultdict

        groups: dict[str, list[Point]] = defaultdict(list)
        for p in points:
            if p.pair:
                groups[p.pair].append(p)
        for members in groups.values():
            if len(members) < 2:
                continue
            src = next((m for m in members if m.series.lower().startswith("orig")), None)
            dst = next((m for m in members if m.series.lower().startswith("opt")), None)
            if src is None or dst is None:
                # No explicit Original/Optimized series — fall back to low->high.
                lo, hi = sorted(members, key=yval)[0], sorted(members, key=yval)[-1]
                src, dst = lo, hi
            ax.annotate(
                "",
                xy=(dst.arithmetic_intensity, yval(dst)),
                xytext=(src.arithmetic_intensity, yval(src)),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "gray",
                    "linewidth": 0.8,
                    "alpha": 0.7,
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=2,
            )

    # --- scatter points ---
    # Group by (family, series) so each combination gets its consistent
    # color (family) + marker (series). Legend entries are added separately
    # below so color and shape are explained independently.
    if use_family:
        for f in families_seen:
            for s in series_seen:
                sx = [p.arithmetic_intensity for p in points if p.family == f and p.series == s]
                sy = [yval(p) for p in points if p.family == f and p.series == s]
                if not sx:
                    continue
                marker = marker_of[s]
                ax.scatter(
                    sx,
                    sy,
                    color=color_of[f],
                    marker=marker,
                    s=90 if marker == "*" else 55,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=6,
                    alpha=0.9,
                )
    else:
        for s in series_seen:
            color, marker = style[s]
            sx = [p.arithmetic_intensity for p in points if p.series == s]
            sy = [yval(p) for p in points if p.series == s]
            ax.scatter(
                sx,
                sy,
                label=s,
                color=color,
                marker=marker,
                s=90 if marker == "*" else 55,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
                alpha=0.9,
            )

    # --- optional per-point annotations ---
    if cfg.key:
        # Number each spec; list "<n>. <label>" in a key box to the right.
        # Points sharing a `pair` id are the same spec (e.g. baseline vs
        # optimized), so they get ONE number / key line, not two. Points with
        # no pair id are each their own spec.
        # First pass: collect each unique spec (one per `pair`; unpaired points
        # are their own spec) with the point that anchors its number. The anchor
        # is the baseline ("Original") point so the number sits at the arrow's
        # tail, not its head.
        spec_order: list[str] = []  # gids in first-seen order
        anchor_pt: dict[str, Point] = {}
        label_pt: dict[str, Point] = {}  # representative for family/label
        for i, p in enumerate(points):
            gid = p.pair or f"__point_{i}"  # unpaired points are unique
            if gid not in label_pt:
                spec_order.append(gid)
                label_pt[gid] = p
                anchor_pt[gid] = p
            if p.series.lower().startswith("orig"):
                anchor_pt[gid] = p  # prefer the baseline as the anchor

        # Number specs in alphabetical order of their label, so the key box
        # reads A->Z and a point can be found by name. Specs with no label keep
        # first-seen (CSV) order at the end.
        spec_order.sort(key=lambda gid: (label_pt[gid].label == "", label_pt[gid].label.lower()))

        key_lines = []
        group_number: dict[str, int] = {}
        anchors: list[tuple[int, float, float]] = []  # (number, ai, yval)
        for n, gid in enumerate(spec_order, start=1):
            group_number[gid] = n
            p = label_pt[gid]
            key_lines.append(f"{n}. {p.label or p.series}")
            a = anchor_pt[gid]
            anchors.append((n, a.arithmetic_intensity, yval(a)))

        # Second pass: numbers whose anchors nearly coincide (in log space)
        # would print on top of each other. Bucket them and fan the text out
        # radially so each number lands in its own slot.
        import math

        TOL = 0.04  # log10 distance under which two anchors are "the same spot"
        buckets: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
        for n, ai, y in anchors:
            key = (round(math.log10(ai) / TOL), round(math.log10(y) / TOL))
            buckets.setdefault(key, []).append((n, ai, y))

        for members in buckets.values():
            members.sort(key=lambda m: m[0])
            count = len(members)
            for j, (n, ai, y) in enumerate(members):
                if count == 1:
                    dx, dy = 9, 5
                else:
                    # Fan the cluster around the anchor (radius grows slightly
                    # with cluster size so big stacks still separate).
                    angle = 2 * math.pi * j / count
                    radius = 9 + 2.5 * count
                    dx = 4 + radius * math.cos(angle)
                    dy = 4 + radius * math.sin(angle)
                ax.annotate(
                    str(n),
                    xy=(ai, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    ha="center",
                    va="center",
                )
        fig.text(
            0.80,
            0.5,
            "\n".join(key_lines),
            fontsize=7.5,
            va="center",
            ha="left",
            family="monospace",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )
        # Leave room on the right for the key box.
        fig.subplots_adjust(right=0.78)
    elif cfg.annotate:
        for p in points:
            if p.label:
                ax.annotate(
                    p.label,
                    xy=(p.arithmetic_intensity, yval(p)),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="dimgray",
                    rotation=30,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(cfg.title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    if use_family:
        # Two-part legend: color explains the family, marker shape explains the
        # series (Original vs Optimized). Proxy handles carry no data.
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                color=color_of[f],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=7,
                label=f,
            )
            for f in families_seen
        ]
        handles += [
            Line2D(
                [],
                [],
                linestyle="",
                marker=marker_of[s],
                color="dimgray",
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=9 if marker_of[s] == "*" else 7,
                label=s,
            )
            for s in series_seen
        ]
        ax.legend(handles=handles, frameon=True, fontsize=8, loc="lower right")
    # Only show a legend if there's more than the default single series.
    elif len(series_seen) > 1 or series_seen[0] != "kernel":
        ax.legend(frameon=True, fontsize=8, loc="lower right")

    if not cfg.key:
        # tight_layout fights the manual right-margin reserved for the key box.
        fig.tight_layout()
    fig.savefig(cfg.output, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cfg.output}  ({len(points)} points, roof: {hw.name})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a roofline plot from a CSV of kernel measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", nargs="?", help="Input CSV file of measurements")
    p.add_argument(
        "-o", "--output", default="roofline.png", help="Output image path (default: roofline.png)"
    )
    p.add_argument("-t", "--title", default="Roofline", help="Plot title")
    p.add_argument(
        "--hardware",
        help="Hardware preset for the roof (see --list-hardware)",
    )
    p.add_argument("--peak-tflops", type=float, help="Override peak compute (TFLOPS)")
    p.add_argument("--peak-bandwidth", type=float, help="Override peak DRAM bandwidth (GB/s)")
    p.add_argument(
        "--connect",
        action="store_true",
        help="Draw a line between points sharing a `pair` id (e.g. baseline->optimized)",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each point with its `label` (inline; best for few points)",
    )
    p.add_argument(
        "--key",
        action="store_true",
        help="Number each point and list labels in a side key (best for many points)",
    )
    p.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    p.add_argument(
        "--series-order",
        help="Comma-separated series names to fix legend/color order",
    )
    p.add_argument(
        "--list-hardware",
        action="store_true",
        help="List built-in hardware presets and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_hardware:
        print("Built-in hardware presets:")
        for key, hw in HARDWARE_PRESETS.items():
            print(
                f"  {key:<14} {hw.name:<42} "
                f"{hw.peak_tflops:>7g} TFLOPS  {hw.peak_bandwidth_gbps:>7g} GB/s"
            )
        return 0

    if not args.csv:
        build_parser().error("the CSV argument is required (or use --list-hardware)")

    hw = _resolve_hardware(args)
    points = load_points(args.csv)
    cfg = PlotConfig(
        title=args.title,
        output=args.output,
        connect=args.connect,
        annotate=args.annotate,
        key=args.key,
        dpi=args.dpi,
        series_order=[s.strip() for s in args.series_order.split(",")] if args.series_order else [],
    )
    plot_roofline(points, hw, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
