"""The generic fused-MLP patcher: deterministic resolution, honest residue (§13.4).

The claim under test is the point-and-start contract: a model id resolves to an
architecture, the architecture to the vLLM file defining it, and the decoder-layer
idiom is matched by exact text — with everything the exact pass cannot decide
reported as an agent handoff, never guessed.
"""

import json

import pytest

from xe_forge.orbit.patch import fused_mlp
from xe_forge.orbit.patch.fused_mlp import ANCHOR

DECODER = (
    "import torch\n"
    "from torch import nn\n\n\n"
    "class TinyDecoderLayer(nn.Module):\n"
    "    def forward(self, positions, hidden_states, residual):\n"
    "        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)\n\n"
    "        # Fully Connected\n"
    + ANCHOR
    + "\n\n\nclass TinyForCausalLM(nn.Module):\n    pass\n"
)


def make_tree(tmp_path, model_id="org/tiny", arch="TinyForCausalLM", body=DECODER):
    hub = tmp_path / "hub"
    snap = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"architectures": [arch]}))
    vllm_root = tmp_path / "vllm"
    models = vllm_root / "model_executor" / "models"
    models.mkdir(parents=True)
    (models / "tiny.py").write_text(body)
    return hub, vllm_root, models / "tiny.py"


class TestPlan:
    def test_resolves_id_to_file_and_anchor(self, tmp_path):
        hub, vllm_root, model_file = make_tree(tmp_path)
        out = fused_mlp.plan("org/tiny", vllm_root, hub_root=hub)
        assert out.ok and not out.already_patched
        assert out.architecture == "TinyForCausalLM"
        assert out.model_file == model_file

    def test_uncached_model_is_agent_handoff(self, tmp_path):
        _, vllm_root, _ = make_tree(tmp_path)
        out = fused_mlp.plan("org/other", vllm_root, hub_root=tmp_path / "hub")
        assert not out.ok and "architecture" in (out.needs_agent or "")

    def test_nonconforming_decoder_is_agent_handoff_not_regex(self, tmp_path):
        # A decoder that deviates from the idiom (e.g. no fused-add layernorm
        # signature) must be handed off, not force-matched.
        hub, vllm_root, _ = make_tree(
            tmp_path,
            body=DECODER.replace("self.post_attention_layernorm(hidden_states, residual)", "self.ln(x)"),
        )
        out = fused_mlp.plan("org/tiny", vllm_root, hub_root=hub)
        assert not out.ok
        assert "matched 0 times" in out.needs_agent

    def test_double_anchor_refused(self, tmp_path):
        hub, vllm_root, _ = make_tree(tmp_path, body=DECODER + "\n" + ANCHOR + "\n")
        out = fused_mlp.plan("org/tiny", vllm_root, hub_root=hub)
        assert not out.ok and "matched 2 times" in out.needs_agent


class TestApplyRevert:
    def test_apply_guards_and_revert_restores_bytes(self, tmp_path):
        hub, vllm_root, model_file = make_tree(tmp_path)
        original = model_file.read_bytes()
        journal = tmp_path / "journal"

        out = fused_mlp.plan("org/tiny", vllm_root, hub_root=hub)
        fused_mlp.apply(out, journal)
        patched = model_file.read_text()
        assert "_orbit_fused_ready" in patched
        assert ANCHOR in patched  # guard off => original path intact, byte-for-byte
        assert patched.index("_orbit_fused_ready(hidden_states.shape[0])") < patched.index(ANCHOR)

        replan = fused_mlp.plan("org/tiny", vllm_root, hub_root=hub)
        assert replan.ok and replan.already_patched  # idempotent, not double-patched

        fused_mlp.revert(journal)
        assert model_file.read_bytes() == original

    def test_apply_on_unresolved_plan_refuses(self, tmp_path):
        with pytest.raises(ValueError):
            fused_mlp.apply(fused_mlp.FusedMlpPlan(model_id="x"), tmp_path / "j")
