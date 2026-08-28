"""A two-layer Qwen-shaped decoder block at toy dimensions (plan §15.2).

Structure, not size, is the point. The shapes are small enough to run in
seconds on a CPU, but the *taxonomy* a profiler sees is the one a real decoder
produces: grouped-query attention with QKV bias, pre-norm RMSNorm, rotary
embeddings, a SwiGLU MLP, and projections that land in a vendor BLAS.

    hidden = 128    heads = 4    kv_heads = 2 (GQA)    head_dim = 32
    ffn = 256       seq = 64     batch = 2

Two layers rather than one, because a single layer hides the thing that matters
for §14 shape aggregation: the same kernel is launched twice per forward with
identical shapes, so "how often did this specialization run" has a non-trivial
answer.

Every kernel call site routes through ``examples.orbit_mini.kernels``, which is
where the adversarial structure lives. This module is deliberately boring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.opaque_gemm import linear_opaque
from .kernels.rmsnorm import rms_norm
from .kernels.rope import apply_rope, build_tables
from .kernels.swiglu import swiglu_projection


@dataclass(frozen=True)
class OrbitMiniConfig:
    """Toy dimensions. Qwen-shaped, ~1/50th scale."""

    hidden_size: int = 128
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 32
    ffn_size: int = 256
    num_layers: int = 2
    seq_len: int = 64
    batch_size: int = 2
    rms_eps: float = 1e-6
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_heads * head_dim must equal hidden_size")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be a multiple of num_kv_heads")

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


class RMSNorm(nn.Module):
    """Pre-norm RMSNorm. Delegates to the hot kernel's launch wrapper.

    The wrapper is what reads ``tuned_configs.json`` and picks the autotune
    config, so the data dependency and the config pin are exercised on every
    forward — twice per layer, four times per step.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No `.contiguous()`: the first call of the first layer receives the
        # deliberately transposed input from `get_example_inputs()` and the
        # strides have to survive to the launch record.
        return rms_norm(x, self.weight, eps=self.eps)


class Attention(nn.Module):
    """Grouped-query attention with QKV bias, Qwen-style.

    Q/K/V and the output projection all go through :func:`linear_opaque`, which
    is a plain ``torch.matmul`` — the E4 call sites of §12.5. The attention
    scores matmul is opaque too. So a decent share of this layer's time has no
    extractable source at all, which is what forces the pipeline to produce a
    ``NO_ACTION`` or a restricted-action recommendation instead of a rewrite.
    """

    def __init__(self, config: OrbitMiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        h, kv = config.hidden_size, config.kv_size

        self.q_weight = nn.Parameter(torch.empty(h, h))
        self.k_weight = nn.Parameter(torch.empty(kv, h))
        self.v_weight = nn.Parameter(torch.empty(kv, h))
        self.o_weight = nn.Parameter(torch.empty(h, h))
        # Qwen carries bias on QKV and none on the output projection.
        self.q_bias = nn.Parameter(torch.zeros(h))
        self.k_bias = nn.Parameter(torch.zeros(kv))
        self.v_bias = nn.Parameter(torch.zeros(kv))

        self.scale = config.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch, seq, _ = x.shape

        q = linear_opaque(x, self.q_weight, self.q_bias, site=f"L{self.layer_idx}.q_proj")
        k = linear_opaque(x, self.k_weight, self.k_bias, site=f"L{self.layer_idx}.k_proj")
        v = linear_opaque(x, self.v_weight, self.v_bias, site=f"L{self.layer_idx}.v_proj")

        q = q.view(batch, seq, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch, seq, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch, seq, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        cos, sin = build_tables(seq, cfg.head_dim, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # GQA: repeat the KV heads to match the query heads.
        if cfg.gqa_groups > 1:
            k = k.repeat_interleave(cfg.gqa_groups, dim=1)
            v = v.repeat_interleave(cfg.gqa_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.full((seq, seq), float("-inf"), device=x.device, dtype=scores.dtype)
        scores = scores + torch.triu(causal, diagonal=1)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).reshape(batch, seq, cfg.hidden_size)
        return out


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP. Gate/up projection and activation are one call site."""

    def __init__(self, config: OrbitMiniConfig) -> None:
        super().__init__()
        h, f = config.hidden_size, config.ffn_size
        self.gate_weight = nn.Parameter(torch.empty(f, h))
        self.up_weight = nn.Parameter(torch.empty(f, h))
        self.down_weight = nn.Parameter(torch.empty(h, f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = swiglu_projection(x, self.gate_weight, self.up_weight)
        return linear_opaque(activated, self.down_weight, site="mlp.down_proj")


class DecoderLayer(nn.Module):
    """One pre-norm decoder layer.

    The forward path here *is* the fusable region of ``kernels/region.py``: the
    o_proj GEMM feeds the post-attention RMSNorm, which feeds the SwiGLU. The
    region declaration is not a description of something that might happen; it
    is a description of these three lines.
    """

    def __init__(self, config: OrbitMiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.attention = Attention(config, layer_idx)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.input_norm(x)
        hidden = self.attention(hidden)

        # --- fusable region begins: gemm -> rmsnorm -> swiglu ---------------
        hidden = linear_opaque(hidden, self.attention.o_weight, site=f"L{self.layer_idx}.o_proj")
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_norm(hidden)
        hidden = self.mlp(hidden)
        # --- fusable region ends -------------------------------------------

        return residual + hidden


class OrbitMiniModel(nn.Module):
    """The whole workload: N decoder layers plus a final norm."""

    def __init__(self, config: OrbitMiniConfig | None = None) -> None:
        super().__init__()
        self.config = config or OrbitMiniConfig()
        self.layers = nn.ModuleList(
            DecoderLayer(self.config, i) for i in range(self.config.num_layers)
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.rms_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        """Deterministic init. Reproducibility is a §17 requirement, not a nicety."""
        generator = torch.Generator(device="cpu").manual_seed(20260101)
        for param in self.parameters():
            if param.dim() >= 2:
                flat = torch.empty(param.shape, dtype=torch.float32)
                flat.uniform_(-0.05, 0.05, generator=generator)
                with torch.no_grad():
                    param.copy_(flat)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
