"""The fusable region: gemm -> rmsnorm -> swiglu (plan §12.11, §15.2).

§15.2 asks for "one region of three fusable kernels, for the Xe-Fuse path". This
is it, and it is a real producer-consumer chain in the executed graph rather
than three ops picked because they looked adjacent. In
:class:`~examples.orbit_mini.model.DecoderLayer` the sequence runs, every layer:

    o_proj GEMM  ->  post-attention RMSNorm  ->  SwiGLU (gate/up proj + SiLU)

with two intermediate tensors that a fused replacement would never materialise:
the projected attention output, and the normed hidden state.

Why this is the hard case, per §12.11:

* The three kernels have **three different providers** — vendor BLAS (E4),
  Triton or SYCL (E1/E2), Triton (E1). A region bundle has to hold kernel
  bundles at different extraction levels and still produce one comparable unit.
* The region is defined by *how the framework strings the kernels together*, so
  §12.11 says region bundles are almost always E3 in the first instance: the
  most faithful driver is the framework itself. :func:`run_region` is that
  driver — the fused candidate is compared against this exact sequence, as a
  unit, not against three separate microbenchmarks.
* Fusing across the E4 GEMM is not possible, so a fusion proposal that claims to
  absorb it is wrong. The region deliberately includes a boundary the optimizer
  must respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .opaque_gemm import linear_opaque
from .rmsnorm import rms_norm
from .swiglu import swiglu_projection


@dataclass(frozen=True)
class RegionMember:
    """One kernel in the region, with the provenance the catalog would record."""

    name: str
    provider: str
    extraction_level: str
    fusable: bool


@dataclass(frozen=True)
class RegionSpec:
    """Static description of the region, readable without running anything."""

    region_id: str
    members: tuple[RegionMember, ...]
    edges: tuple[tuple[str, str], ...]
    intermediates: tuple[str, ...]
    driver: str
    notes: str = ""
    extraction_level: str = "E3"
    metadata: dict = field(default_factory=dict)


#: The region as the catalog should see it.
MLP_REGION = RegionSpec(
    region_id="orbit_mini.decoder.mlp_entry",
    members=(
        RegionMember("o_proj_gemm", "vendor_blas", "E4", fusable=False),
        RegionMember("post_attention_rmsnorm", "triton_or_sycl", "E1", fusable=True),
        RegionMember("swiglu_projection", "triton", "E1", fusable=True),
    ),
    edges=(
        ("o_proj_gemm", "post_attention_rmsnorm"),
        ("post_attention_rmsnorm", "swiglu_projection"),
    ),
    intermediates=("attn_projected", "hidden_normed"),
    driver="examples.orbit_mini.kernels.region:run_region",
    notes=(
        "The GEMM is E4 and cannot be fused into. A fusion candidate that claims "
        "to absorb it is proposing something it cannot deliver; the region exists "
        "partly to make that claim testable."
    ),
)


def run_region(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the whole region and hand back its intermediates.

    §12.11: "a driver that runs the whole region so a fused replacement can be
    compared against the unfused sequence as a unit". Returning the
    intermediates is what lets the rig check that a fused candidate really did
    eliminate them, rather than computing them and throwing them away.
    """
    attn_projected = linear_opaque(hidden, o_proj_weight, site="region.o_proj")
    merged = residual + attn_projected
    hidden_normed = rms_norm(merged, norm_weight)
    activated = swiglu_projection(hidden_normed, gate_weight, up_weight)
    return activated, {
        "attn_projected": attn_projected,
        "hidden_normed": hidden_normed,
        "residual_out": merged,
    }
