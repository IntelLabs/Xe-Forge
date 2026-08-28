"""Re-export surface for the device helpers (plan §15.2, §12.6 step 2).

DELIBERATE TRAP. This module contains no code of its own. It exists purely to
put an alias/re-export hop between the hot kernel and the modules its device
helpers actually live in, which is the pattern §12.6 calls out:

    "every call that lands on a `@triton.jit` function is a device helper and is
     added to the work list — transitively, across modules, including helpers
     that are re-exported or imported under an alias."

Real kernels do this constantly (vLLM's attention and fused-MoE helpers are
reached through exactly this kind of package-level façade). An extractor that
resolves free names against the *importing* module's globals and stops will
resolve ``weighted_scale`` to a name in this file and find no definition; it has
to keep following the binding to :mod:`helpers_b`, and from there to
:mod:`helpers_c`.

Every name here is bound under a name different from the one it was defined
with, so matching by symbol name alone fails too.
"""

from __future__ import annotations

from .helpers_a import _rms_scale as rms_scale
from .helpers_a import _sum_of_squares as sum_of_squares
from .helpers_b import _swiglu_activate as swiglu_gate
from .helpers_b import _weighted_scale as weighted_scale
from .helpers_c import _rope_rotate_pair as rope_rotate_pair

__all__ = [
    "rms_scale",
    "rope_rotate_pair",
    "sum_of_squares",
    "swiglu_gate",
    "weighted_scale",
]
