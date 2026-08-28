"""Rotary position embedding (plan §15.2).

RoPE is here because it is the part of a Qwen-shaped decoder that most reliably
produces a *separate* small kernel under ``torch.compile`` rather than being
folded into attention, so the kernel catalog has something in it that is neither
the hot kernel nor an opaque BLAS call.

Its device helper is ``device_ops.rope_rotate_pair``, an alias for
``helpers_c._rope_rotate_pair`` — so :mod:`helpers_c` sits in the closure of two
different kernels (this one and, transitively, RMSNorm). Getting per-kernel
bundles right when helper modules overlap is the multi-file case §12.6 is about.

The cos/sin table is computed once and cached on the module, which makes it a
host-side dependency of the kernel: §12.6 step 5 says the launch wrapper's own
work — grid computation, stride derivation, table construction — is part of the
kernel's real behaviour and has to be extracted with it.
"""

import torch

from .device_ops import rope_rotate_pair as _rope_rotate_alias
from .triton_compat import HAS_TRITON, jit, tl

#: Default RoPE base. Module-level constant read by the table builder.
ROPE_THETA: float = 10000.0

_TABLE_CACHE: dict = {}


@jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    seq_len,
    HALF_DIM: tl.constexpr,
):
    """Apply RoPE to one ``(row, head_dim)`` slice, rotate-half layout.

    One program per flattened ``(batch, head, position)`` row of a contiguous
    ``(batch, heads, seq, head_dim)`` tensor, so the position is recovered as
    ``row % seq_len``. Deriving the table index from the grid index rather than
    receiving it is host-side behaviour folded into the kernel — the sort of
    thing §12.6 step 5 says has to be extracted along with the kernel body.
    """
    row = tl.program_id(0)
    pos = row % seq_len
    offs = tl.arange(0, HALF_DIM)
    row_base = row * (2 * HALF_DIM)
    tbl_base = pos * (2 * HALF_DIM)

    x_even = tl.load(x_ptr + row_base + offs)
    x_odd = tl.load(x_ptr + row_base + HALF_DIM + offs)
    cos = tl.load(cos_ptr + tbl_base + offs)
    sin = tl.load(sin_ptr + tbl_base + offs)

    out_even, out_odd = _rope_rotate_alias(x_even, x_odd, cos, sin)
    tl.store(out_ptr + row_base + offs, out_even)
    tl.store(out_ptr + row_base + HALF_DIM + offs, out_odd)


def build_tables(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (and cache) the cos/sin tables.

    Host-side work that belongs to the kernel (§12.6 step 5). A bundle that
    extracts ``_rope_kernel`` and leaves this behind has extracted half of a
    kernel and will be fed a table it did not build.
    """
    key = (seq_len, head_dim, str(device), str(dtype))
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    half = head_dim // 2
    exponent = torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim
    inv_freq = 1.0 / (ROPE_THETA**exponent)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    tables = (emb.cos().to(dtype), emb.sin().to(dtype))
    _TABLE_CACHE[key] = tables
    return tables


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` and ``k`` in place-equivalent fashion.

    Shapes: ``(batch, heads, seq, head_dim)`` for q/k, ``(seq, head_dim)`` for
    the tables. The Triton path is guarded; CPU takes the torch path, which is
    the same rotate-half arithmetic as :func:`helpers_c._rope_rotate_pair`.
    """
    if HAS_TRITON and q.device.type in ("cuda", "xpu"):
        return (
            _launch_rope(q, cos, sin),
            _launch_rope(k, cos, sin),
        )
    return _apply_rope_torch(q, k, cos, sin)


def _launch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Triton launch. Reached only on a GPU with Triton installed."""
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    xc = x.contiguous()
    out = torch.empty_like(xc)
    rows = xc.numel() // head_dim
    _rope_kernel[(rows,)](xc, cos.contiguous(), sin.contiguous(), out, seq_len, head_dim // 2)
    return out.view_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_even, x_odd = x[..., :half], x[..., half:]
    return torch.cat((-x_odd, x_even), dim=-1)


def _apply_rope_torch(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch rotate-half RoPE, matched to the device helper."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
