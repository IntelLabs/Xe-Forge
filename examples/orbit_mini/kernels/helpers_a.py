"""Device helpers, module 1 of 3 (plan §15.2, §12.6).

DELIBERATE TRAP. The hot kernel (``kernels/rmsnorm.py``) does not keep its
device helpers next to itself. They are spread over three modules, and the
closure walk in §12.6 step 2 has to follow every hop:

    rmsnorm.py
      ├─ helpers_a._rms_scale            <- this module, imported directly
      └─ device_ops.weighted_scale       <- an alias, re-exported from helpers_b
           └─ helpers_b._weighted_scale
                └─ helpers_c._clamp_to_finite

An extractor that copies only the file the kernel lives in produces a bundle
that fails the isolated-import check in §12.12 step 1, which is exactly the
failure this workload exists to provoke.

This module also owns a module-level constant used as a ``constexpr`` default
(:data:`RMS_EPS_DEFAULT`), covering §12.6 step 3.

No ``from __future__ import annotations`` here on purpose: Triton reads real
annotation objects off the signature, and PEP 563 stringisation breaks its
``tl.constexpr`` detection.
"""

from .triton_compat import jit, tl

#: Fallback epsilon. The tuned-config JSON (§12.8) normally supplies this, but a
#: module-level default is what a `constexpr` default looks like to the closure
#: walk, so it stays here as well.
RMS_EPS_DEFAULT: float = 1e-6

#: Number of accumulator lanes the reduction helper assumes. Used as a constexpr.
RMS_ACC_LANES: int = 4


@jit
def _rms_scale(sum_sq, n_cols, eps):
    """Reciprocal-RMS scale factor: ``1 / sqrt(mean(x^2) + eps)``.

    Device helper. Reached by a direct import from the hot kernel — the easy
    hop, present so the hard hops next door are not the only ones.
    """
    mean_sq = sum_sq / n_cols
    return 1.0 / tl.sqrt(mean_sq + eps)


@jit
def _sum_of_squares(x, axis: tl.constexpr):
    """Row-wise sum of squares in fp32, whatever the input dtype."""
    xf = x.to(tl.float32)
    return tl.sum(xf * xf, axis=axis)
