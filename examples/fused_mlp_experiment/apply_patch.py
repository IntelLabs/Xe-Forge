"""Apply (or revert) the fused-MLP patch to vllm-src through the journalled patcher.

Usage: apply_patch.py apply|revert|status
The patch is a guarded branch: env ORBIT_FUSED_MLP=1 + M <= 32 + the extension
loadable — otherwise the original path runs byte-for-byte. §13.2 rules: journalled,
atomic, revertible; the journal lives in the scratch run dir.
"""
import sys
from pathlib import Path

from xe_forge.orbit.patch.inplace import InPlacePatcher

TARGET = Path.home() / ".cache/orbit-dev/vllm-src/vllm/model_executor/models/qwen2.py"
JOURNAL = Path(__file__).parent / "patch_journal"

HELPERS = '''

# ---- orbit_fused (injected by Xe-Orbit, journalled; revert via patcher) ----
import os as _orbit_os

_ORBIT_FUSED_LOADED = False


def _orbit_fused_ready(m: int) -> bool:
    """The fused r0/r1 path: opt-in, small-M only, extension present (plan §14.4)."""
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

ANCHOR = """        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual"""

FUSED = """        if _orbit_fused_ready(hidden_states.shape[0]):
            # Fused r0/r1 chain (plan §13.4): residual add + rms scale in one tiny
            # kernel, then Xe-Fuse's k2 GEMM+SwiGLU with gamma folded into the
            # packed weight. Numerically as close to fp64 truth as the unfused
            # path (measured); token-level equivalence is gated by the e2e check.
            eps = self.post_attention_layernorm.variance_epsilon
            scale = torch.ops.orbit_fused.add_rms_scale(hidden_states, residual, eps)
            packed = _orbit_packed_weight(self)
            d = torch.ops.orbit_fused.gate_up_swiglu(residual.contiguous(), packed, scale)
            hidden_states, _ = self.mlp.down_proj(d[:, 0::2])
            return hidden_states, residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual"""


def main() -> int:
    patcher = InPlacePatcher(journal_dir=JOURNAL, sandbox_roots=[TARGET.parent])
    mode = sys.argv[1]
    if mode == "revert":
        outcome = patcher.revert_all()
        print("revert:", outcome)
        return 0
    if mode == "status":
        print(patcher.format())
        return 0

    original = TARGET.read_text(encoding="utf-8")
    assert original.count(ANCHOR) == 1, f"anchor found {original.count(ANCHOR)} times"
    assert "_orbit_fused_ready" not in original, "already patched"
    patched = original.replace(ANCHOR, FUSED) + HELPERS
    patcher.apply(TARGET, patched.encode(), kernel_id="qwen2_fused_mlp", reason="fused r0/r1 chain")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
