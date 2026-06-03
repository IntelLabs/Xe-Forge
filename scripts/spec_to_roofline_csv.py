#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "PyYAML>=6.0.0",
# ]
# ///
"""Convert an xe-forge kernel spec (YAML) into a roofline CSV.

The spec only carries *problem shapes* and a FLOP formula — there are no
measured kernel times in it. So this emits, for each bench variant, its
**arithmetic intensity** (FLOP / bytes moved). Feed the CSV to roofline.py
*without* a tflops column and each variant lands on the roofline ceiling,
showing where that shape sits (memory- vs compute-bound) and the best the
hardware could theoretically deliver for it.

------------------------------------------------------------------------------
Bytes / arithmetic-intensity model  (Fused MoE grouped GEMM)
------------------------------------------------------------------------------
For a token-routed MoE matmul with dims M (tokens), K (in), N (out),
E (experts), TOPK (experts per token):

  FLOP    = M * TOPK * N * K * 2                 (from the spec's `flop`)

  DRAM bytes (compulsory traffic, each tensor read/written once):
    activations A   : M * TOPK * K   * act_bytes   (a token's row feeds each of
                                                     its TOPK experts)
    weights B       : E_active * N * K * w_bytes   (each activated expert's
                                                     weight loaded once)
    routing ids     : M * TOPK       * 4           (int32)
    output          : M * TOPK * N   * act_bytes

  E_active = min(E, M * TOPK)   # can't activate more experts than routings

`QUANT` selects element widths (matches the spec's w13/w2 quant variants):
    0  bf16          act=2  w=2
    1  fp8  w8a8     act=1  w=1
    2  int8 w8a8     act=1  w=1
    3  int8 w8a16    act=2  w=1

This is the standard compulsory-traffic estimate: it assumes each tensor is
streamed from DRAM exactly once (perfect weight reuse across tokens sharing an
expert). It is an upper bound on AI, so the plotted point is the most
compute-bound / highest-ceiling position the shape can occupy. Real kernels
with imperfect reuse sit at lower AI (further left).

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
  uv run scripts/spec_to_roofline_csv.py examples/vllm/2_FusedMoE.yaml -o moe.csv
  uv run scripts/roofline.py moe.csv --hardware arc-pro-b70 --annotate \
      --title "Roofline — Fused MoE (theoretical ceilings)" -o plots/moe.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

import yaml

# QUANT code -> (activation bytes, weight bytes, label)
QUANT_MODEL: dict[int, tuple[int, int, str]] = {
    0: (2, 2, "bf16"),
    1: (1, 1, "fp8-w8a8"),
    2: (1, 1, "int8-w8a8"),
    3: (2, 1, "int8-w8a16"),
}


def eval_formula(formula: str, dims: dict[str, int]) -> float:
    """Evaluate a flop formula like 'M*TOPK*N*K*2' against the dims dict."""
    # Only names from dims and integer arithmetic — no builtins.
    return float(eval(formula, {"__builtins__": {}}, dict(dims)))


def moe_bytes(dims: dict[str, int]) -> float:
    """Compulsory DRAM bytes for one Fused MoE variant (see module docstring)."""
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]
    E = dims["E"]
    topk = dims["TOPK"]
    quant = dims.get("QUANT", 0)
    act_b, w_b, _ = QUANT_MODEL.get(quant, QUANT_MODEL[0])

    e_active = min(E, M * topk)
    a_bytes = M * topk * K * act_b
    b_bytes = e_active * N * K * w_b
    id_bytes = M * topk * 4
    out_bytes = M * topk * N * act_b
    return a_bytes + b_bytes + id_bytes + out_bytes


def extract_comments(text: str) -> dict[str, str]:
    """Map each top-level `variant:` key to the comment on the next line.

    PyYAML discards comments, but the spec puts a human-readable model name
    (e.g. '# Qwen3-30B-A3B-Instruct w13') right under each bench key — that
    makes the best point label.
    """
    comments: dict[str, str] = {}
    lines = text.splitlines()
    key_re = re.compile(r"^([A-Za-z0-9_\-]+):\s*$")
    for i, line in enumerate(lines):
        m = key_re.match(line)
        if not m:
            continue
        # look at the next non-blank line for a comment
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                continue
            c = nxt.strip()
            if c.startswith("#"):
                comments[m.group(1)] = c.lstrip("#").strip()
            break
    return comments


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("spec", help="Path to the kernel spec YAML")
    p.add_argument("-o", "--output", default="-", help="Output CSV path (default: stdout)")
    p.add_argument(
        "--variant-prefix",
        default="bench",
        help="Only convert variants whose key starts with this (default: bench)",
    )
    args = p.parse_args(argv)

    text = open(args.spec).read()
    spec = yaml.safe_load(text)
    comments = extract_comments(text)

    rows: list[dict] = []
    for key, value in spec.items():
        if not key.startswith(args.variant_prefix):
            continue
        # Each variant is a list of one config dict.
        if not isinstance(value, list) or not value:
            continue
        cfg = value[0]
        dims = cfg.get("dims")
        formula = cfg.get("flop")
        if not dims or not formula:
            print(f"warning: {key} has no dims/flop — skipping", file=sys.stderr)
            continue

        flop = eval_formula(formula, dims)
        nbytes = moe_bytes(dims)
        ai = flop / nbytes
        quant = dims.get("QUANT", 0)
        _, _, qlabel = QUANT_MODEL.get(quant, QUANT_MODEL[0])

        model = comments.get(key, key)
        # compact shape tag for the annotation
        shape = f"M{dims['M']} N{dims['N']} K{dims['K']} E{dims['E']} t{dims['TOPK']}"
        rows.append(
            {
                "series": qlabel,
                "label": f"{model} [{shape}]",
                "arithmetic_intensity": f"{ai:.2f}",
                "flop": f"{flop:.0f}",
                "bytes": f"{nbytes:.0f}",
            }
        )

    if not rows:
        raise SystemExit(f"error: no '{args.variant_prefix}*' variants found in {args.spec}")

    fieldnames = ["series", "label", "arithmetic_intensity", "flop", "bytes"]
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if out is not sys.stdout:
        out.close()
        print(f"wrote {args.output}  ({len(rows)} variants)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
