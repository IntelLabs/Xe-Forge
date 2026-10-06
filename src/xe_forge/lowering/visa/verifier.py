"""Correctness checks for hand-lowered kernels.

Low-level code fails in ways source-level edits do not: a wrong mask reads past
the end of a buffer, a wrong stride writes into a neighbour's memory, a lane
mapping off by one is right on 128 elements and wrong on 129. So an attempt
must pass every check below before it is called correct:

- **several shapes**: the spec's own dims, plus sizes either side of the block
  size, odd sizes and a size of one, for each runtime dimension in turn;
- **several seeds**: values are random, never a pattern that hides an error;
- **guard regions**: every buffer the kernel can write sits between sentinel
  bytes, and a changed sentinel is an out-of-bounds write;
- **tolerances per dtype**: exact for copies and integers, close to the
  dtype's resolution for floating point, looser where transcendentals are
  involved.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch

from xe_forge.lowering.visa.diagnostics import CorrectnessResult

GUARD_BYTES = 4096
GUARD_PATTERN = 0xA5

# (rtol, atol) per output dtype. The float rows sit a little above the dtype's
# resolution: the reference is itself a compiled kernel with its own rounding.
_TOLERANCE = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
    torch.float64: (1e-9, 1e-12),
}
_TRANSCENDENTAL_TOLERANCE = {
    torch.float32: (1e-4, 1e-5),
}


def tolerance_for(
    dtype: torch.dtype,
    *,
    transcendental: bool = False,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[float, float]:
    """Tolerance for comparing outputs of ``dtype``; explicit values win."""
    if not dtype.is_floating_point:
        base = (0.0, 0.0)
    elif transcendental and dtype in _TRANSCENDENTAL_TOLERANCE:
        base = _TRANSCENDENTAL_TOLERANCE[dtype]
    else:
        base = _TOLERANCE.get(dtype, (1e-2, 1e-2))
    return (rtol if rtol is not None else base[0], atol if atol is not None else base[1])


@dataclass(frozen=True)
class VerificationCase:
    dims: dict[str, int]
    seed: int
    label: str


def boundary_sizes(original: int, block: int) -> list[int]:
    """Sizes that exercise the edges of a blocked loop over one dimension."""
    sizes = [original, 1, 7, block - 1, block, block + 1, 3 * block + 5]
    out = []
    for s in sizes:
        if s >= 1 and s not in out:
            out.append(s)
    return out


def verification_cases(
    dims: dict[str, int],
    block: int,
    *,
    seeds: tuple[int, ...] = (0, 1),
    extra: list[dict[str, int]] | None = None,
    sweep: bool = True,
    max_elements: int = 1 << 24,
) -> list[VerificationCase]:
    """The spec's dims, then each dimension swept over :func:`boundary_sizes`.

    ``sweep=False`` is for kernels that only accept particular sizes (an
    unmasked kernel needs multiples of its block); then only the spec's dims and
    ``extra`` are used.
    """
    cases: list[VerificationCase] = []
    seen: set[tuple] = set()

    def add(d: dict[str, int], label: str):
        key = tuple(sorted(d.items()))
        if key in seen or math.prod(d.values() or [1]) > max_elements:
            return
        seen.add(key)
        for s in seeds:
            cases.append(VerificationCase(dict(d), s, label))

    add(dims, "spec")
    # Only integer dims are sizes; a float in `dims` is a constructor scalar.
    for name, value in (dims.items() if sweep else ()):
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        for size in boundary_sizes(value, block):
            add({**dims, name: size}, f"{name}={size}")
    for i, d in enumerate(extra or []):
        add({**dims, **d}, f"extra[{i}]")
    return cases


def block_size(constexprs: dict[str, Any], default: int = 128) -> int:
    """The largest power-of-two integer constexpr: the tile a program covers."""
    blocks = [v for v in constexprs.values() if isinstance(v, int) and not isinstance(v, bool) and v >= 2 and v & (v - 1) == 0]
    return max(blocks) if blocks else default


def spec_with_dims(spec, variant: str, dims: dict[str, int]):
    """A copy of ``spec`` whose first ``variant`` entry uses ``dims``."""
    spec = copy.deepcopy(spec)
    entry = spec._variants(variant)[0]
    entry.dims = {**entry.dims, **dims}
    return spec


# -- guard regions ------------------------------------------------------------


@dataclass
class GuardedTensor:
    original: torch.Tensor
    view: torch.Tensor
    buffer: torch.Tensor  # uint8, guard + span + guard
    span_bytes: int

    def guards_intact(self) -> bool:
        head = self.buffer[:GUARD_BYTES]
        tail = self.buffer[GUARD_BYTES + self.span_bytes :]
        return bool((head == GUARD_PATTERN).all() and (tail == GUARD_PATTERN).all())

    def write_back(self) -> None:
        self.original.copy_(self.view)


def guard(t: torch.Tensor) -> GuardedTensor:
    """Place a copy of ``t`` (same shape and strides) between sentinel bytes."""
    if any(s < 0 for s in t.stride()):
        raise ValueError("negative strides are not supported")
    span_elems = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()) if n > 0) if t.numel() else 0
    span_bytes = span_elems * t.element_size()
    # Keep the interior aligned like a fresh allocation.
    buffer = torch.full((GUARD_BYTES * 2 + span_bytes,), GUARD_PATTERN, dtype=torch.uint8, device=t.device)
    interior = buffer[GUARD_BYTES : GUARD_BYTES + span_bytes].view(t.dtype)
    view = interior.as_strided(t.shape, t.stride())
    view.copy_(t)
    return GuardedTensor(t, view, buffer, span_bytes)


# -- comparison ---------------------------------------------------------------


def compare(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_failures: int = 8,
) -> CorrectnessResult:
    """Element-wise comparison with the statistics the model needs to localize an error."""
    if expected.shape != actual.shape:
        return CorrectnessResult(False, error=f"shape mismatch: expected {list(expected.shape)}, got {list(actual.shape)}")
    if expected.dtype != actual.dtype:
        return CorrectnessResult(False, error=f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    total = expected.numel()
    if total == 0:
        return CorrectnessResult(True, total_elements=0)
    e = expected.detach().to(torch.float64).flatten()
    a = actual.detach().to(torch.float64).flatten()
    nonfinite_new = (~torch.isfinite(a)) & torch.isfinite(e)
    diff = (a - e).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    both_nan = torch.isnan(a) & torch.isnan(e)
    same_inf = torch.isinf(a) & torch.isinf(e) & (torch.sign(a) == torch.sign(e))
    ok = (diff <= atol + rtol * e.abs()) | both_nan | same_inf
    failing = (~ok).nonzero().flatten()
    rel = diff / e.abs().clamp_min(1e-30)
    result = CorrectnessResult(
        success=failing.numel() == 0,
        max_abs_error=float(diff[~(both_nan | same_inf)].max()) if total else 0.0,
        max_rel_error=float(rel[~(both_nan | same_inf)].max()) if total else 0.0,
        failing_elements=int(failing.numel()),
        total_elements=total,
        nan_or_inf=bool(nonfinite_new.any()),
    )
    if failing.numel():
        shape = list(expected.shape)
        for flat in failing[:max_failures].tolist():
            result.first_failures.append(
                {
                    "index": list(_unravel(flat, shape)),
                    "expected": float(e[flat]),
                    "got": float(a[flat]),
                }
            )
    return result


def _unravel(flat: int, shape: list[int]) -> tuple[int, ...]:
    idx = []
    for n in reversed(shape):
        idx.append(flat % n if n else 0)
        flat //= n if n else 1
    return tuple(reversed(idx))


def merge(results: list[tuple[str, CorrectnessResult]]) -> CorrectnessResult:
    """One verdict over many cases; the first failing case is the one described."""
    if not results:
        return CorrectnessResult(False, error="no verification case ran")
    failing = [(label, r) for label, r in results if not r.success]
    worst_abs = max((r.max_abs_error or 0.0) for _, r in results)
    worst_rel = max((r.max_rel_error or 0.0) for _, r in results)
    if not failing:
        return CorrectnessResult(True, shapes_checked=len(results), max_abs_error=worst_abs, max_rel_error=worst_rel)
    label, first = failing[0]
    merged = copy.deepcopy(first)
    merged.shapes_checked = len(results)
    merged.max_abs_error = worst_abs
    merged.max_rel_error = worst_rel
    merged.failing_case = {"case": label, "failing_cases": len(failing), "of": len(results)}
    return merged
