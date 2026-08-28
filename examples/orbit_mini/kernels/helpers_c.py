"""Device helpers, module 3 of 3 (plan §15.2, §12.6).

DELIBERATE TRAP. Nothing in this module is imported by the hot kernel directly.
It is reached only *transitively*, through :mod:`helpers_b`, which is itself
reached through the alias re-export in :mod:`device_ops`. Three hops, two of
them invisible to anything that only looks at the kernel module's own imports:

    rmsnorm.py -> device_ops.weighted_scale (alias)
               -> helpers_b._weighted_scale
               -> helpers_c._clamp_to_finite   <- here

A closure walk that stops at depth one, or that resolves imports textually
instead of following ``@triton.jit`` call targets, will miss this file. The
resulting bundle imports cleanly on the dev machine — because the source
package is on ``sys.path`` — and fails the §12.12 isolated-import check, which
is the point.

This module is also where RoPE's rotation math lives, so the closure of a
*second* kernel (``kernels/rope.py``) overlaps this one. Bundles for the two
kernels must both contain it; deduplicating it away from either is a bug.
"""

from .triton_compat import jit, tl

#: Saturation limit for the SwiGLU gate. A module-level constant referenced from
#: inside a device helper: it must travel with the bundle (§12.6 step 3) or the
#: extracted kernel silently changes numerics.
#:
#: A `tl.constexpr` *instance*, not a bare float, because real Triton refuses bare
#: globals inside @jit functions ("Cannot access global variable ... from within
#: @jit'ed function") — found by running this workload on an actual XPU, where the
#: fixture had only ever exercised the no-triton fallback. Under the compat shim
#: `tl.constexpr` degrades to a plain number, so the CPU path is unchanged, and the
#: extraction trap is intact: the constant still lives at module level in a third
#: module and still has to be collected.
CLAMP_LIMIT = tl.constexpr(30.0)


@jit
def _clamp_to_finite(x):
    """Clamp to ``[-CLAMP_LIMIT, +CLAMP_LIMIT]``.

    Deepest node in the hot kernel's device-helper closure. Reads the
    module-level :data:`CLAMP_LIMIT`, so both the function *and* the constant
    have to be collected.
    """
    lo = -CLAMP_LIMIT
    hi = CLAMP_LIMIT
    return tl.minimum(tl.maximum(x, lo), hi)


@jit
def _rope_rotate_pair(x_even, x_odd, cos, sin):
    """Rotate one ``(even, odd)`` coordinate pair by ``(cos, sin)``.

    Shared with ``kernels/rope.py``; see the module docstring on why the overlap
    matters for per-kernel bundle closure.
    """
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return out_even, out_odd
