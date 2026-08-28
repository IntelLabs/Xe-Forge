"""Generic fused-MLP patch for vLLM decoder layers: matches the Llama-lineage
decoder-layer idiom by exact text and injects a guarded fused branch; non-matching
decoders are handed to the agent. Design rationale: docs/DESIGN.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from xe_forge.orbit.patch.inplace import InPlacePatcher

# The Llama-lineage decoder-layer idiom; byte-exact, must appear exactly once.
ANCHOR = """        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual"""

HELPERS = '''

# ---- orbit_fused (injected by Xe-Orbit, journalled; revert via xe-orbit fuse-apply --revert) ----
import os as _orbit_os

_ORBIT_FUSED_LOADED = False


def _orbit_fused_ready(m: int) -> bool:
    """The fused r0/r1 path: opt-in, small-M only, extension present."""
    global _ORBIT_FUSED_LOADED
    if _orbit_os.environ.get("ORBIT_FUSED_MLP") != "1" or m > 32:
        return False
    if not _ORBIT_FUSED_LOADED:
        lib = _orbit_os.environ.get("ORBIT_FUSED_LIB", "")
        if not lib:
            return False
        torch.ops.load_library(lib)
        _ORBIT_FUSED_LOADED = True
        print("ORBIT FUSED MLP ACTIVE", file=__import__("sys").stderr)
    return True


def _orbit_packed_weight(layer) -> torch.Tensor:
    """gamma-folded, gate/up-interleaved [K, 2I] weight, cached per layer."""
    cached = getattr(layer, "_orbit_packed", None)
    if cached is not None:
        return cached
    w = layer.mlp.gate_up_proj.weight.data  # [2I, K]
    gamma = layer.post_attention_layernorm.weight.data  # [K]
    two_i = w.shape[0]
    folded = (w.float() * gamma.float()[None, :]).to(w.dtype)
    packed = torch.empty(w.shape[1], two_i, device=w.device, dtype=w.dtype)
    packed[:, 0::2] = folded[: two_i // 2].t()
    packed[:, 1::2] = folded[two_i // 2 :].t()
    layer._orbit_packed = packed.contiguous()
    return layer._orbit_packed
'''

FUSED = """        if _orbit_fused_ready(hidden_states.shape[0]):
            # Fused r0/r1 chain: residual add + rms scale, then GEMM+SwiGLU
            # with gamma folded into the packed weight.
            eps = self.post_attention_layernorm.variance_epsilon
            scale = torch.ops.orbit_fused.add_rms_scale(hidden_states, residual, eps)
            packed = _orbit_packed_weight(self)
            d = torch.ops.orbit_fused.gate_up_swiglu(residual.contiguous(), packed, scale)
            hidden_states, _ = self.mlp.down_proj(d[:, 0::2])
            return hidden_states, residual
""" + ANCHOR


@dataclass
class FusedMlpPlan:
    """What the deterministic pass concluded, before anything is touched."""

    model_id: str
    architecture: str | None = None
    model_file: Path | None = None
    ok: bool = False
    already_patched: bool = False
    needs_agent: str | None = None  # why exact matching stopped, if it did

    def format(self) -> str:
        lines = [f"model:        {self.model_id}"]
        if self.architecture:
            lines.append(f"architecture: {self.architecture}")
        if self.model_file:
            lines.append(f"model file:   {self.model_file}")
        if self.already_patched:
            lines.append("state:        already patched (revert first, or run arms as-is)")
        elif self.ok:
            lines.append("state:        anchor matched exactly once — deterministic patch applies")
        elif self.needs_agent:
            lines.append(f"state:        needs agent — {self.needs_agent}")
        return "\n".join(lines)


def find_architecture(model_id: str, hub_root: Path | None = None) -> str | None:
    """architectures[0] from the HF cache's config.json, or None if not cached."""
    hub = hub_root or Path.home() / ".cache/huggingface/hub"
    cache = hub / ("models--" + model_id.replace("/", "--"))
    for config in sorted(cache.glob("snapshots/*/config.json")):
        try:
            archs = json.loads(config.read_text(encoding="utf-8")).get("architectures") or []
        except (OSError, json.JSONDecodeError):
            continue
        if archs:
            return str(archs[0])
    return None


def find_model_file(architecture: str, vllm_root: Path) -> Path | None:
    """The vLLM source file defining `class <architecture>(`, by exact scan."""
    models_dir = vllm_root / "model_executor" / "models"
    needle = f"class {architecture}("
    for path in sorted(models_dir.glob("*.py")):
        try:
            if needle in path.read_text(encoding="utf-8"):
                return path
        except OSError:
            continue
    return None


def plan(model_id: str, vllm_root: Path, hub_root: Path | None = None) -> FusedMlpPlan:
    """Resolve model id -> architecture -> file -> anchor. No side effects."""
    out = FusedMlpPlan(model_id=model_id)
    out.architecture = find_architecture(model_id, hub_root)
    if out.architecture is None:
        out.needs_agent = (
            "no config.json in the HF cache names an architecture; the model has "
            "not been downloaded, or is not a standard HF checkout"
        )
        return out
    out.model_file = find_model_file(out.architecture, vllm_root)
    if out.model_file is None:
        out.needs_agent = f"no file under {vllm_root}/model_executor/models defines {out.architecture}"
        return out
    text = out.model_file.read_text(encoding="utf-8")
    if "_orbit_fused_ready" in text:
        out.already_patched = True
        out.ok = True
        return out
    count = text.count(ANCHOR)
    if count != 1:
        out.needs_agent = (
            f"the decoder-layer idiom matched {count} times in {out.model_file.name} "
            "(need exactly 1); this decoder deviates from the Llama lineage and the "
            "patch must be authored against its actual forward"
        )
        return out
    out.ok = True
    return out


def apply(plan_result: FusedMlpPlan, journal_dir: Path) -> None:
    """Apply the guarded patch through the journalled patcher."""
    if not plan_result.ok or plan_result.model_file is None:
        raise ValueError("apply() called on a plan that did not resolve; check plan().ok first")
    if plan_result.already_patched:
        return
    target = plan_result.model_file
    patcher = InPlacePatcher(journal_dir=journal_dir, sandbox_roots=[target.parent])
    original = target.read_text(encoding="utf-8")
    patched = original.replace(ANCHOR, FUSED) + HELPERS
    patcher.apply(
        target,
        patched.encode(),
        kernel_id=f"fused_mlp:{plan_result.architecture}",
        reason="fused r0/r1 chain (generic decoder-layer idiom)",
    )


def revert(journal_dir: Path) -> dict:
    """Digest-verified revert of everything this journal applied."""
    return InPlacePatcher(journal_dir=journal_dir, sandbox_roots=[]).revert_all()
