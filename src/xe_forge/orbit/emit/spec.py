"""
Spec and Model emission (plan §8, §PR 9).

Orbit hands work to Xe-Forge through the file contract Xe-Forge already consumes: a
kernel `.py`, a YAML spec, and a `*_pytorch.py` reference. Nothing new is invented
here — the point of §8 is that the contract exists, so extraction only has to fill a
template whose shape is already defined.

Two details of the real repository shape this:

* The PyTorch reference **is** resolved by name substitution (`<stem>_pytorch.py`), so
  emitting `kernel_pytorch.py` next to `kernel.py` is load-bearing, not decorative.
* The spec is passed explicitly via `--spec`, so the filename is free; the in-tree
  convention is a sibling `<KernelName>.yaml`. We emit `spec.yaml` and pass it
  explicitly, which works with the loader as it stands today.

The observed shape distribution maps onto Xe-Forge's existing variant mechanism, which
already supports arbitrary `bench-gpu-N` families. The one addition the plan asks for
is `weight:` on each variant (§9.1) — and because Xe-Forge's spec loader currently
drops unknown keys silently, an emitted weight neither works nor errors until that
lands. We emit it anyway, and say so in the file, so the spec is correct ahead of the
loader rather than needing a second pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xe_forge.orbit.models import (
    CapturedInvocation,
    KernelBundle,
    KernelRecord,
    ShapeObservation,
)

# Shapes below this share of observed calls are folded away rather than emitted as
# their own variant: a long tail of one-off shapes produces a spec nobody can tune
# against, and the weighted objective would spend its budget on noise.
MIN_VARIANT_WEIGHT = 0.02
MAX_VARIANTS = 8


def weighted_variants(
    shapes: list[ShapeObservation],
    max_variants: int = MAX_VARIANTS,
    min_weight: float = MIN_VARIANT_WEIGHT,
) -> list[dict[str, Any]]:
    """Turn an observed shape distribution into weighted benchmark variants.

    Weights are normalized over the variants actually emitted, so they sum to 1 even
    after the tail is dropped. The dropped mass is reported by `emit_spec` rather than
    silently vanishing — a candidate tuned on 60% of the distribution while the report
    implies 100% is the kind of quiet dishonesty §11.10 warns about.
    """
    if not shapes:
        return []

    total = sum(s.count for s in shapes) or 1
    ranked = sorted(shapes, key=lambda s: s.count, reverse=True)

    kept = [s for s in ranked[:max_variants] if (s.count / total) >= min_weight]
    if not kept:
        kept = ranked[:1]

    kept_total = sum(s.count for s in kept) or 1
    variants: list[dict[str, Any]] = []
    for shape in kept:
        variants.append(
            {
                "dims": dict(shape.dims),
                "dtypes": dict(shape.dtypes),
                "weight": round(shape.count / kept_total, 4),
                "observed_calls": shape.count,
            }
        )
    return variants


def coverage(shapes: list[ShapeObservation], variants: list[dict[str, Any]]) -> float:
    """Fraction of observed calls the emitted variants actually represent."""
    if not shapes:
        return 0.0
    total = sum(s.count for s in shapes) or 1
    covered = sum(int(v["observed_calls"]) for v in variants)
    return covered / total


def build_spec(
    kernel: KernelRecord,
    inputs: CapturedInvocation | None = None,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build the spec document for one kernel.

    Per-variant `rtol`/`atol` already exist in Xe-Forge's `VariantSpec` and are
    resolved spec-over-config, so tightened tolerances (§9.3) work today — Orbit only
    has to emit them.
    """
    variants = weighted_variants(kernel.shapes)
    spec: dict[str, Any] = {}

    if inputs is not None and inputs.shape_map:
        spec["inputs"] = {
            name: {
                "shape": shape,
                "dtype": inputs.dtype_map.get(name, "bfloat16"),
            }
            for name, shape in inputs.shape_map.items()
        }
    elif variants:
        spec["inputs"] = {"x": {"shape": list(variants[0]["dims"].values()), "dtype": "bfloat16"}}

    params = sorted(spec.get("inputs", {}))
    rtol, atol = tolerance or (1e-3, 1e-6)

    if not variants:
        return spec

    for index, variant in enumerate(variants):
        key = "bench-gpu" if index == 0 else f"bench-gpu-{index}"
        entry: dict[str, Any] = {
            "params": params,
            "dims": variant["dims"],
            # `weight:` is the §9.1 addition: the observed share of this shape in the
            # trace. Xe-Forge's spec loader parses it, and `--objective weighted`
            # scores a candidate across the whole family with it.
            "weight": variant["weight"],
            "rtol": rtol,
            "atol": atol,
        }
        dtypes = variant.get("dtypes") or {}
        if dtypes:
            entry["dtype"] = next(iter(dtypes.values()))
        spec[key] = [entry]

    return spec


def emit_spec(
    kernel: KernelRecord,
    bundle: KernelBundle,
    output_dir: Path,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Write `spec.yaml` plus a candidate README into the candidate directory.

    Returns a summary including the shape coverage the spec achieves, so a caller can
    report how much of the observed distribution the emitted variants represent.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    spec = build_spec(kernel, bundle.inputs, tolerance)
    variants = weighted_variants(kernel.shapes)
    covered = coverage(kernel.shapes, variants)

    header = (
        f"# Generated by xe-orbit from {kernel.id} ({kernel.runtime_name}).\n"
        f"# Extraction level: {bundle.extraction_level.value}\n"
        f"# Shape coverage: {covered * 100:.1f}% of observed calls across "
        f"{len(variants)} variant(s).\n"
        "#\n"
        "# 'weight:' expresses the observed shape distribution (plan §9.1). Run with\n"
        "# --objective weighted to score a candidate across the whole family, with a\n"
        "# hard no-regression constraint on every variant; --variant still selects a\n"
        "# single configuration.\n"
    )
    spec_path = target / "spec.yaml"
    spec_path.write_text(
        header + yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    return {
        "spec_path": str(spec_path),
        "variants": len(variants),
        "coverage": covered,
        "dropped_shapes": max(0, len(kernel.shapes) - len(variants)),
    }


def emit_candidate(
    kernel: KernelRecord,
    bundle: KernelBundle,
    output_dir: Path,
    tolerance: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Write the full candidate directory Xe-Forge consumes (§8).

    candidates/<kernel-id>/
        kernel.py            extracted kernel (or in-situ harness) + Model wrapper
        kernel_pytorch.py    reference implementation from the aten op
        spec.yaml            inputs, weighted bench variants
        bundle/              the extraction closure
        inputs/              captured real tensors
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary = emit_spec(kernel, bundle, target, tolerance)

    # The kernel entry point: for E1/E2 the extracted source, for E3 the harness.
    if bundle.primary_source and Path(bundle.primary_source).is_file():
        kernel_py = target / "kernel.py"
        if Path(bundle.primary_source).suffix == ".py":
            kernel_py.write_text(
                Path(bundle.primary_source).read_text(encoding="utf-8"), encoding="utf-8"
            )
            summary["kernel_path"] = str(kernel_py)

    # The reference is resolved by name substitution, so the filename matters.
    reference = target / "kernel_pytorch.py"
    if not reference.exists():
        reference.write_text(_reference_stub(kernel), encoding="utf-8")
    summary["reference_path"] = str(reference)

    return summary


def _reference_stub(kernel: KernelRecord) -> str:
    op = kernel.framework_op or kernel.runtime_name
    return f'''"""
PyTorch reference for {kernel.id} ({op}).

Xe-Forge resolves this file by name substitution on the kernel filename, so it must
sit beside `kernel.py` as `kernel_pytorch.py`.

This is a stub: the eager-mode equivalent of the op has to be supplied before the
correctness gate means anything. Orbit deliberately does not guess it — a plausible
but wrong reference produces a candidate that passes correctness and is wrong in the
model, which is the failure the whole correctness ladder exists to prevent (§19).
"""

import torch


class Model(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Supply the eager-mode reference for {op!r} before running the "
            "correctness gate."
        )
'''
