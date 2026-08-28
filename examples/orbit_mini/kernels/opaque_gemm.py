"""The opaque library call — the E4 / NO_ACTION path (plan §12.5, §15.2).

DELIBERATE TRAP, of a different kind. Everything else in this package has
extractable source. This does not, and the pipeline has to say so instead of
inventing something.

``torch.matmul`` on a float tensor dispatches into a vendor BLAS — oneDNN or
oneMKL on XPU, MKL or OpenBLAS on CPU, cuBLAS on CUDA. There is no source to
lift, so §12.5 puts it at **E4**:

    "No source extraction is possible. [...] **E4**, with actions restricted to
     BACKEND_CHANGE, LAYOUT_CHANGE, LIBRARY_CONFIG and REGION_FUSION."

Two failure modes are being tested here at once:

1. An extractor that reports E1/E2 for this kernel is lying about what it can
   do, and everything downstream inherits the lie.
2. A ranking stage that emits ``NO_ACTION`` for a kernel holding real GPU time,
   without first proposing one of the four permitted actions, has given up too
   early.

The op is genuinely on the hot path — it is the QKV, output and down
projections of both decoder layers — so it will show up in the catalog with a
meaningful share of the time, not as a rounding error that is easy to skip.
"""

from __future__ import annotations

import torch

#: What §12.5 permits for an E4 kernel. Anything outside this set proposed
#: against this call site is a bug in the action planner.
PERMITTED_ACTIONS: tuple[str, ...] = (
    "backend_change",
    "layout_change",
    "library_config",
    "region_fusion",
)

#: Extraction level this call site must be classified as.
EXTRACTION_LEVEL: str = "E4"

#: Call sites recorded for the catalog, so the rig can assert the classification
#: without a profiler on CPU-only CI.
CALL_SITES: list[dict] = []


def linear_opaque(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    site: str = "unnamed",
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` straight through the vendor BLAS.

    Deliberately *not* a Triton kernel and deliberately not decomposed. The
    whole value of this call site is that it has no source.
    """
    out = torch.matmul(x, weight.transpose(-2, -1))
    if bias is not None:
        out = out + bias
    CALL_SITES.append(
        {
            "site": site,
            "shape": (tuple(x.shape), tuple(weight.shape)),
            "level": EXTRACTION_LEVEL,
            "permitted_actions": PERMITTED_ACTIONS,
            "provider": "vendor_blas",
        }
    )
    return out


def reset_call_sites() -> None:
    """Clear the recorded call sites between runs."""
    CALL_SITES.clear()
