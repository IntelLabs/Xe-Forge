#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "PyYAML>=6.0.0",
# ]
# ///
"""Convert a benchmark results CSV into the roofline.py input format.

The benchmark CSV (one row per kernel variant) carries *measured* times for
both the reference (`baseline_us`) and the optimized Triton kernel
(`triton_us`), plus the optimized throughput (`tflops`). roofline.py instead
wants one row per *point* with an `arithmetic_intensity` column, and pairs of
points (baseline vs optimized) linked by a `pair` id.

So each input row is expanded into TWO output rows sharing a `pair`:

  Optimized : tflops = <tflops>                       (as measured)
  Original  : tflops = <tflops> * triton_us/baseline_us
              (same FLOP, scaled by the inverse time ratio == tflops/speedup)

The arithmetic intensity is FLOP / compulsory-DRAM-bytes, computed per family
from each spec's FLOP formula and a once-through traffic model (see below). The
quant element widths (fp8/int8) and MoE TOPK come from the spec YAMLs, since the
CSV `config` strings omit them; attention configs are fully specified inline.

------------------------------------------------------------------------------
Byte / arithmetic-intensity model  (compulsory traffic: each tensor once)
------------------------------------------------------------------------------
BatchedMoE   FLOP = E*M*N*K*2  (per-expert dense GEMM, all E experts active)
  A      E*M*K*act   B  E*N*K*w   out  E*M*N*act   ids  E*4

FusedMoE     FLOP = M*TOPK*N*K*2  (token-routed grouped GEMM)
  A   M*TOPK*K*act   B  Eact*N*K*w   out  M*TOPK*N*act   ids  M*TOPK*4
  Eact = min(E, M*TOPK)

UnifiedAttention  FLOP = 4*TQ*QH*D*MKV  (QK^T + softmax·V)
  query  TQ*QH*D*2   K,V  2*MKV*KH*D*kv   out  TQ*QH*D*2

Element widths are selected per family by the quant code (see *_QUANT below).
This is an upper bound on AI (perfect reuse), so points sit at their most
compute-bound position; real kernels with imperfect reuse sit further left.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
  uv run scripts/benchmark_to_roofline_csv.py benchmark.csv -o roofline_input.csv
  uv run scripts/roofline.py roofline_input.csv --hardware arc-pro-b70 \
      --connect --key --title "Roofline (baseline vs Triton)" \
      -o plots/roofline.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

# CSV family name -> spec YAML, relative to repo root (one above scripts/).
FAMILY_SPEC: dict[str, str] = {
    "2_BatchedMoE": "examples/vllm/1_BatchedMoE.yaml",
    "3_FusedMoE": "examples/vllm/2_FusedMoE.yaml",
    "4_UnifiedAttention": "examples/vllm/3_UnifiedAttention.yaml",
}

# CSV family name -> clean display name (drops the leading index).
FAMILY_NAME: dict[str, str] = {
    "2_BatchedMoE": "BatchedMoE",
    "3_FusedMoE": "FusedMoE",
    "4_UnifiedAttention": "UnifiedAttention",
}

# Per-family quant code -> (activation bytes, weight bytes). The codes are
# family-specific: the two MoE specs assign different meanings to the same int.
BATCHEDMOE_QUANT = {0: (2, 2), 1: (1, 1), 2: (2, 1)}  # bf16, fp8 w8a8, int8 w8a16
FUSEDMOE_QUANT = {0: (2, 2), 1: (1, 1), 2: (1, 1), 3: (2, 1)}  # +int8 w8a8, int8 w8a16
# Attention KV-cache dtype bytes by KQM code (0 bf16, else quantized 1B).
KQM_KV_BYTES = {0: 2, 1: 1, 2: 1, 3: 1}


def batchedmoe_ai(d: dict) -> float:
    act, w = BATCHEDMOE_QUANT.get(d.get("QUANT", 0), (2, 2))
    E, M, K, N = d["E"], d["M"], d["K"], d["N"]
    flop = E * M * N * K * 2
    nbytes = E * M * K * act + E * N * K * w + E * M * N * act + E * 4
    return flop / nbytes


def fusedmoe_ai(d: dict) -> float:
    act, w = FUSEDMOE_QUANT.get(d.get("QUANT", 0), (2, 2))
    M, K, N, E, topk = d["M"], d["K"], d["N"], d["E"], d["TOPK"]
    flop = M * topk * N * K * 2
    e_active = min(E, M * topk)
    nbytes = M * topk * K * act + e_active * N * K * w + M * topk * N * act + M * topk * 4
    return flop / nbytes


def attention_ai(d: dict) -> float:
    kv = KQM_KV_BYTES.get(d.get("KQM", 0), 2)
    TQ, QH, D, KH, MKV = d["TQ"], d["QH"], d["D"], d["KH"], d["MKV"]
    flop = 4 * TQ * QH * D * MKV
    nbytes = TQ * QH * D * 2 + 2 * MKV * KH * D * kv + TQ * QH * D * 2
    return flop / nbytes


AI_FN = {
    "2_BatchedMoE": batchedmoe_ai,
    "3_FusedMoE": fusedmoe_ai,
    "4_UnifiedAttention": attention_ai,
}


def parse_config(cfg: str) -> dict[str, int]:
    """'E=128,M=64,K=768' -> {'E':128,'M':64,'K':768}."""
    out: dict[str, int] = {}
    for part in cfg.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = int(v.strip())
    return out


def load_spec_dims(spec_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Return (variant -> dims) and (variant -> comment label) from a spec YAML."""
    text = spec_path.read_text()
    spec = yaml.safe_load(text)
    dims_by_key: dict[str, dict] = {}
    for key, value in spec.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            d = value[0].get("dims")
            if d:
                dims_by_key[key] = d

    # Grab the human-readable comment under each variant key for nicer labels.
    comments: dict[str, str] = {}
    lines = text.splitlines()
    key_re = re.compile(r"^([A-Za-z0-9_\-]+):\s*$")
    for i, line in enumerate(lines):
        m = key_re.match(line)
        if not m:
            continue
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                continue
            if nxt.strip().startswith("#"):
                comments[m.group(1)] = nxt.strip().lstrip("#").strip()
            break
    return dims_by_key, comments


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", help="Input benchmark results CSV")
    p.add_argument("-o", "--output", default="-", help="Output roofline CSV (default: stdout)")
    p.add_argument(
        "--min-tflops",
        type=float,
        default=0.0,
        help="Drop a pair if its Optimized throughput is below this (declutters "
        "the plot by removing the dense band of memory-bound decode/small-batch "
        "shapes). Both rows of a surviving pair are kept. Default: 0 (keep all).",
    )
    p.add_argument(
        "--min-ai",
        type=float,
        default=0.0,
        help="Drop a pair whose arithmetic intensity is below this (FLOP/byte). "
        "Removes the left-most memory-bound shapes. Default: 0 (keep all).",
    )
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    specs = {fam: load_spec_dims(repo_root / rel) for fam, rel in FAMILY_SPEC.items()}

    rows: list[dict] = []
    skipped: list[str] = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            fam = r["family"]
            key = r["shape_key"]
            if fam not in AI_FN:
                skipped.append(f"{fam}/{key}: unknown family")
                continue

            # Need both measured times and the optimized throughput.
            try:
                base_us = float(r["baseline_us"])
                tri_us = float(r["triton_us"])
                opt_tflops = float(r["tflops"])
            except (ValueError, KeyError):
                skipped.append(f"{fam}/{key}: no measurement")
                continue
            if base_us <= 0 or tri_us <= 0 or opt_tflops <= 0:
                skipped.append(f"{fam}/{key}: non-positive measurement")
                continue

            # Dims: start from the spec, overlay anything in the config string.
            dims_by_key, comments = specs[fam]
            dims = dict(dims_by_key.get(key, {}))
            dims.update(parse_config(r.get("config", "")))
            try:
                ai = AI_FN[fam](dims)
            except KeyError as e:
                skipped.append(f"{fam}/{key}: missing dim {e}")
                continue

            # Baseline throughput: same FLOP, slower by the time ratio.
            base_tflops = opt_tflops * tri_us / base_us

            label = comments.get(key, key)
            pair = f"{fam}:{key}"
            family = FAMILY_NAME.get(fam, fam)
            rows.append(
                {
                    "series": "Original",
                    "family": family,
                    "label": label,
                    "pair": pair,
                    "arithmetic_intensity": f"{ai:.2f}",
                    "tflops": f"{base_tflops:.3f}",
                }
            )
            rows.append(
                {
                    "series": "Optimized",
                    "family": family,
                    "label": label,
                    "pair": pair,
                    "arithmetic_intensity": f"{ai:.2f}",
                    "tflops": f"{opt_tflops:.3f}",
                }
            )

    if not rows:
        raise SystemExit(f"error: no usable rows in {args.csv}")

    # Optional declutter: drop a whole pair when its Optimized point is below
    # the threshold. Filtering on Optimized (the headline result) keeps the
    # baseline alongside it so the baseline->optimized connector still draws.
    if args.min_tflops > 0:
        opt_tflops_by_pair = {
            r["pair"]: float(r["tflops"]) for r in rows if r["series"] == "Optimized"
        }
        dropped = sorted(
            (p for p, t in opt_tflops_by_pair.items() if t < args.min_tflops),
            key=lambda p: opt_tflops_by_pair[p],
        )
        rows = [r for r in rows if opt_tflops_by_pair.get(r["pair"], 0.0) >= args.min_tflops]
        for p in dropped:
            print(
                f"filtered {p}: {opt_tflops_by_pair[p]:.3f} < {args.min_tflops:g} TFLOPS",
                file=sys.stderr,
            )
        if not rows:
            raise SystemExit(f"error: --min-tflops {args.min_tflops:g} filtered out every pair")

    # Optional declutter on the x-axis: drop the left-most (low arithmetic
    # intensity) pairs. Both rows of a pair share the same AI, so filtering
    # row-by-row keeps pairs intact.
    if args.min_ai > 0:
        dropped_ai = sorted(
            {
                (r["pair"], float(r["arithmetic_intensity"]))
                for r in rows
                if float(r["arithmetic_intensity"]) < args.min_ai
            },
            key=lambda pa: pa[1],
        )
        rows = [r for r in rows if float(r["arithmetic_intensity"]) >= args.min_ai]
        for p, ai in dropped_ai:
            print(f"filtered {p}: AI {ai:.2f} < {args.min_ai:g} FLOP/byte", file=sys.stderr)
        if not rows:
            raise SystemExit(f"error: --min-ai {args.min_ai:g} filtered out every pair")

    fieldnames = ["series", "family", "label", "pair", "arithmetic_intensity", "tflops"]
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if out is not sys.stdout:
        out.close()
        print(f"wrote {args.output}  ({len(rows)} rows, {len(rows) // 2} pairs)", file=sys.stderr)
    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
