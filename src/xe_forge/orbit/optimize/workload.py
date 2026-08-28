"""
Optimizing the workload's path, not one kernel (plan §13.7, §7.6, §14).

The kernel-scoped loop in `loop.py` answers "can this kernel be made faster". That is
the wrong unit for an inference workload, and picking it produces a predictable kind of
failure: you optimize whatever happens to be extractable and report the result as if it
were a workload improvement.

Measured on Qwen2.5-0.5B under vLLM, the path is:

    GEMM (qkv/o/gate/up/down/lm_head)  93.17%   6305 calls   oneDNN, opaque
    RoPE                                2.56%   3144 calls   SYCL
    SwiGLU activation                   1.60%   1560 calls   vLLM SYCL
    RMSNorm                             1.19%   3185 calls   vLLM SYCL
    sampling                            0.61%    260 calls   Triton
    KV cache write                      0.52%   1560 calls   vLLM SYCL

Every rewritable kernel there sums to under 7%. So the honest action space for *this*
model is dominated by things that are not kernel rewrites at all — batch and serving
configuration, the GEMM backend, layout, fusing adjacent stages — and a loop that can
only patch kernel source cannot reach 93% of the runtime.

That is why the unit here is the workload. An action may be a config change, an
environment or backend change, or a source patch; they are trialled the same way and
measured the same way, by re-running the framework's own benchmark end to end. Hyperloom
draws the same boundary: its EXPLORE phase sweeps server arguments and environment
variants alongside source patches, and every KEEP is revalidated against the full stack
rather than against the kernel that changed.

The architecture is part of the context on purpose. "Optimize this GEMM" invites generic
advice; "this is a 24-layer Qwen2 with hidden 896, GQA 14:2, SwiGLU over 4864, tied
embeddings into a 151936 vocab, decoding at batch 32" invites reasoning about which
projections are skinny and why the model is memory-bound at decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionKind(StrEnum):
    """What sort of change an action makes. Determines how it is applied and reverted."""

    # A serving-configuration argument: batch size, max sequence length, scheduling.
    CONFIG = "config"
    # An environment variable: backend selection, library tuning knobs.
    ENVIRONMENT = "environment"
    # An edit to framework source. The only kind that needs the in-place patcher.
    SOURCE = "source"


@dataclass
class WorkloadAction:
    """One change to the workload, in the terms needed to apply and undo it."""

    kind: ActionKind
    title: str
    rationale: str
    # For CONFIG: engine arguments. For ENVIRONMENT: variables. For SOURCE: unused.
    settings: dict[str, object] = field(default_factory=dict)
    target_file: str = ""
    new_source: bytes | None = None

    @property
    def reversible_without_patching(self) -> bool:
        """Config and environment changes revert by not passing them again.

        Worth distinguishing because it decides how much can go wrong: a config trial
        that crashes leaves nothing behind, while a source trial that crashes leaves a
        modified framework and needs the journal in §13.6 to recover.
        """
        return self.kind in (ActionKind.CONFIG, ActionKind.ENVIRONMENT)


@dataclass
class PathStage:
    """One stage of the model's forward path, with what it actually costs."""

    name: str
    gpu_share: float
    calls: int
    provider: str

    @property
    def rewritable(self) -> bool:
        return self.provider not in ("onednn", "onemkl", "runtime", "unknown")

    def describe(self) -> str:
        note = "" if self.rewritable else "  (opaque — no source to rewrite)"
        return (
            f"{self.gpu_share * 100:6.2f}%  {self.calls:>6} calls  "
            f"{self.provider:<9} {self.name}{note}"
        )


@dataclass
class ModelArchitecture:
    """The shape of the model, so proposals can reason about the path rather than guess."""

    name: str = ""
    layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    attention_heads: int = 0
    kv_heads: int = 0
    vocab_size: int = 0
    activation: str = ""
    tied_embeddings: bool = False

    @property
    def grouped_query(self) -> bool:
        return bool(self.kv_heads) and self.kv_heads < self.attention_heads

    def describe(self) -> str:
        lines = [f"{self.name}: {self.layers} layers, hidden {self.hidden_size}"]
        if self.attention_heads:
            ratio = (
                f"GQA {self.attention_heads}:{self.kv_heads}"
                if self.grouped_query
                else f"MHA {self.attention_heads} heads"
            )
            lines.append(f"  attention: {ratio}, head dim {self.head_dim}")
        if self.intermediate_size:
            lines.append(f"  MLP: {self.activation or 'unknown'} over {self.intermediate_size}")
        if self.vocab_size:
            tied = " (tied to embeddings)" if self.tied_embeddings else ""
            lines.append(f"  vocab: {self.vocab_size}{tied}")
        return "\n".join(lines)

    @property
    def head_dim(self) -> int:
        if not self.attention_heads:
            return 0
        return self.hidden_size // self.attention_heads

    def projection_shapes(self, batch: int) -> list[tuple[str, int, int, int]]:
        """The per-layer GEMM shapes at a given decode batch, as (name, M, K, N).

        These are what the 93% actually is. At decode M is the batch, so every one of
        them is skinny — which is the whole reason batching moved throughput 5x and a
        kernel rewrite could not have.
        """
        if not (self.hidden_size and self.attention_heads):
            return []
        h, d = self.hidden_size, self.head_dim
        qkv_out = (self.attention_heads + 2 * self.kv_heads) * d
        shapes = [
            ("qkv_proj", batch, h, qkv_out),
            ("o_proj", batch, h, h),
        ]
        if self.intermediate_size:
            shapes.append(("gate_up_proj", batch, h, 2 * self.intermediate_size))
            shapes.append(("down_proj", batch, self.intermediate_size, h))
        if self.vocab_size:
            shapes.append(("lm_head", batch, h, self.vocab_size))
        return shapes


@dataclass
class WorkloadProfile:
    """Everything a proposer needs to reason about this workload's path."""

    architecture: ModelArchitecture
    stages: list[PathStage] = field(default_factory=list)
    batch: int = 0
    framework: str = ""
    harness: str = ""
    baseline_tok_s: float | None = None
    minimum_detectable_effect: float = 0.0

    @property
    def rewritable_share(self) -> float:
        return sum(s.gpu_share for s in self.stages if s.rewritable)

    @property
    def opaque_share(self) -> float:
        return sum(s.gpu_share for s in self.stages if not s.rewritable)

    def describe(self) -> str:
        """Render the profile as proposer context.

        Deliberately leads with what cannot be rewritten. A proposer given only the
        kernel list will reach for a kernel; told that 93% of the time is behind a
        closed library, it reaches for the things that can actually move that 93%.
        """
        lines = [self.architecture.describe(), ""]
        if self.batch:
            lines.append(f"serving at batch {self.batch} on {self.framework or 'this framework'}")
        if self.baseline_tok_s:
            lines.append(f"baseline throughput: {self.baseline_tok_s:.1f} tok/s")
            if self.harness:
                lines.append(f"  measured by: {self.harness}")
        lines.append("")
        lines.append("WHERE THE GPU TIME GOES:")
        for stage in sorted(self.stages, key=lambda s: -s.gpu_share):
            lines.append("  " + stage.describe())
        lines.append("")
        lines.append(
            f"{self.opaque_share * 100:.1f}% of GPU time is in kernels with no editable "
            f"source. Rewriting every remaining kernel perfectly would improve end-to-end "
            f"time by at most {self.rewritable_share * 100:.1f}%."
        )
        if self.minimum_detectable_effect:
            lines.append(
                f"This workload can resolve a difference of {self.minimum_detectable_effect:.2f}% "
                f"or larger; anything smaller cannot be demonstrated (§17)."
            )
        if self.architecture.hidden_size and self.batch:
            lines.append("")
            lines.append(f"PER-LAYER GEMM SHAPES AT BATCH {self.batch} (M x K x N):")
            for name, m, k, n in self.architecture.projection_shapes(self.batch):
                lines.append(f"  {name:<14} {m:>5} x {k:>6} x {n:>6}")
        return "\n".join(lines)


def action_space(profile: WorkloadProfile) -> list[str]:
    """The actions worth proposing for this profile, as prompt guidance.

    Derived from the measured split rather than listed generically: a workload whose
    time is mostly opaque gets told to reach for configuration and fusion, and one with
    real rewritable share gets told a kernel rewrite is on the table. Offering the full
    menu every time is how a proposer ends up suggesting a rewrite for a closed library.
    """
    actions = [
        "CONFIG: serving arguments — batch size, scheduling, sequence limits. On a "
        "decode-bound model these change the GEMM shapes themselves, which is the only "
        "way to reach an opaque library kernel.",
        "ENVIRONMENT: backend selection and library tuning variables.",
    ]
    if profile.opaque_share > 0.5:
        actions.append(
            "REGION FUSION: adjacent stages that could be one kernel, removing an "
            "intermediate round trip to memory."
        )
        actions.append(
            "LAYOUT: a different memory layout for the tensors feeding the opaque "
            "kernels, which can change which implementation the library selects."
        )
    if profile.rewritable_share > 0.02:
        actions.append(
            f"SOURCE: rewriting a kernel — bounded by a {profile.rewritable_share * 100:.1f}% "
            f"ceiling across every rewritable kernel combined."
        )
    return actions
