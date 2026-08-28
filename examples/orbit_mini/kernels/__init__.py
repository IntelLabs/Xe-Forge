"""Kernels for ``orbit_mini`` (plan §15.2).

Everything under here is arranged to be *hard to extract correctly*. The map:

===========================  ==================================================
``triton_compat.py``         Triton availability shim; keeps the source
                             Triton-shaped on machines with no Triton.
``helpers_a.py``             Device helpers, module 1 of 3.
``helpers_b.py``             Device helpers, module 2 of 3 — reached only
                             through the ``device_ops`` alias re-export.
``helpers_c.py``             Device helpers, module 3 of 3 — reached only
                             transitively, from ``helpers_b``.
``device_ops.py``            Pure re-export surface. The alias hop.
``tuned.py`` + ``.json``     The non-code data dependency (§12.8).
``rmsnorm.py``               The hot kernel: autotune, heuristics closing over
                             module state, split closure, data dep.
``swiglu.py``                Consumer end of the fusable region.
``rope.py``                  Second kernel sharing ``helpers_c``.
``opaque_gemm.py``           The E4 / ``NO_ACTION`` call site.
``sycl_op.py``               The hand-written SYCL dispatcher op (P1 rung).
``region.py``                gemm -> rmsnorm -> swiglu, for the Xe-Fuse path.
===========================  ==================================================

None of it imports from ``xe_forge``; the workload has to stand alone, or it is
testing the pipeline against itself.
"""

from __future__ import annotations

from .opaque_gemm import linear_opaque
from .region import MLP_REGION, run_region
from .rmsnorm import rms_norm
from .rope import apply_rope, build_tables
from .swiglu import swiglu_projection
from .triton_compat import HAS_TRITON
from .tuned import TUNED_CONFIG_PATH, lookup

__all__ = [
    "HAS_TRITON",
    "MLP_REGION",
    "TUNED_CONFIG_PATH",
    "apply_rope",
    "build_tables",
    "linear_opaque",
    "lookup",
    "rms_norm",
    "run_region",
    "swiglu_projection",
]
