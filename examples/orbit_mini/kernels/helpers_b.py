"""Device helpers, module 2 of 3 (plan §15.2, §12.6).

DELIBERATE TRAP. The hot kernel never names this module. It imports
``weighted_scale`` from :mod:`device_ops`, which is a pure re-export module that
binds ``helpers_b._weighted_scale`` under a *different* name. So resolving the
hot kernel's free names requires following an alias across a re-export chain
(§12.6 step 2, "including helpers that are re-exported or imported under an
alias"), and resolving *this* module's calls requires one more hop into
:mod:`helpers_c`.

Import-graph note: this module imports from :mod:`helpers_c` under an alias too,
so a textual ``import`` scrape of the hot kernel's source finds neither file.
"""

from .helpers_c import _clamp_to_finite as _saturate
from .triton_compat import jit, tl

#: Gate scale applied before the sigmoid. Module-level constant consumed by a
#: device helper, same class of dependency as `helpers_c.CLAMP_LIMIT` — and a
#: `tl.constexpr` instance for the same reason: real Triton refuses bare globals
#: inside @jit functions, and the compat shim degrades constexpr to a plain
#: number, so the CPU fallback is unchanged and the bundle-must-carry-it trap
#: stays armed.
GATE_SCALE = tl.constexpr(1.0)


@jit
def _weighted_scale(x, w, scale):
    """Apply the RMS scale and the learned per-channel weight.

    Reached from the hot kernel only through the ``device_ops.weighted_scale``
    alias. Calls into :mod:`helpers_c` through a *second* alias (``_saturate``),
    which is the third module in the chain.
    """
    normed = _saturate(x.to(tl.float32) * scale)
    return normed * w.to(tl.float32)


@jit
def _swiglu_activate(gate, up):
    """SwiGLU: ``silu(gate) * up``, with the gate saturated first.

    Belongs to the SwiGLU kernel's closure rather than the hot kernel's, but it
    lives in the same module — so a bundle that copies whole modules instead of
    resolved symbols drags in a helper it does not use, and a bundle that copies
    single functions drops the shared :mod:`helpers_c` dependency. Both mistakes
    are visible from here.
    """
    g = _saturate(gate.to(tl.float32)) * GATE_SCALE
    return (g * tl.sigmoid(g)) * up.to(tl.float32)
