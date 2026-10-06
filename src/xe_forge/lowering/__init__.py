"""Experimental AI lowering of source kernels to lower-level targets.

Orthogonal to source-level optimization (:mod:`xe_forge.pipeline`): a kernel can
be lowered directly, or after Xe-Forge has optimized it as source. The only
target today is Intel vISA (:mod:`xe_forge.lowering.visa`), on Xe2.
"""
