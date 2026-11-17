"""Flags and settings reach what they configure: config overrides, the variant,
.env, the launch hint per agent, the dspy engine's profiler, a one-shape baseline check."""

from __future__ import annotations

import argparse
import os

import pytest
import yaml


def test_an_override_reaches_a_section_whose_name_holds_an_underscore():
    from xe_forge.config import ConfigManager

    c = ConfigManager().override(
        device_config_default_num_warps=8, device_config_preferred_tile_m=64,
        xpu_group_size_m=2, llm_temperature=0.3,
    ).get()  # fmt: skip
    assert c.device_config.default_num_warps == 8 and c.device_config.preferred_tile_m == 64
    assert c.device_config.group_size_m == 2 and c.llm.temperature == 0.3


def test_the_specs_default_variant_wins_over_the_dsls_family(tmp_path):
    from xe_forge.core.spec_loader import load_spec

    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump({"default_variant": "dec", "dec": [{"dims": {"N": 1}}]}))
    s = load_spec(spec)
    assert s.resolve_variant(None, "bench-xpu") == "dec"
    assert s.resolve_variant("other", "bench-xpu") == "other"
    spec.write_text(yaml.safe_dump({"bench-xpu": [{"dims": {"N": 1}}]}))
    assert load_spec(spec).resolve_variant(None, "bench-xpu") == "bench-xpu"


def test_dotenv_in_the_working_directory_is_loaded_and_the_shell_wins(tmp_path, monkeypatch):
    from xe_forge import load_env

    (tmp_path / ".env").write_text("XF_TEST_ENV_A=from-file\nXF_TEST_ENV_B=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XF_TEST_ENV_A", raising=False)
    monkeypatch.setenv("XF_TEST_ENV_B", "from-shell")
    load_env()
    assert (
        os.environ["XF_TEST_ENV_A"] == "from-file" and os.environ["XF_TEST_ENV_B"] == "from-shell"
    )
    monkeypatch.delenv("XF_TEST_ENV_A")


@pytest.mark.parametrize("name, says", [("claude", "claude /optimize-kernel k"),
                                        ("codex", 'codex "Follow .claude/commands')])  # fmt: skip
def test_each_agent_names_its_own_launch_command(name, says):
    from xe_forge import agent_cli

    assert agent_cli.get(name).interactive("k").startswith(says)


@pytest.mark.parametrize("unitrace, vtune, kind", [(True, True, "UnitraceXPUProfiler"),
                                                   (False, True, "XPUProfiler"),
                                                   (False, False, None)])  # fmt: skip
def test_the_dspy_engine_profiles_with_what_was_asked(monkeypatch, unitrace, vtune, kind):
    from xe_forge.config import ConfigManager
    from xe_forge.core import profiler
    from xe_forge.engines.dspy_engine import DSPyEngine

    monkeypatch.setattr(profiler.UnitraceXPUProfiler, "available", lambda self: True)
    monkeypatch.setattr(profiler.XPUProfiler, "available", lambda self: True)
    c = ConfigManager().get()
    c.profiler.unitrace_enabled, c.profiler.vtune_enabled, c.trial.enabled = unitrace, vtune, False
    got = DSPyEngine(c).profiler
    assert (type(got).__name__ if got else None) == kind


def test_a_one_shape_run_is_held_to_the_captured_kernel_too(tmp_path, capsys):
    from xe_forge.skills import benchmark

    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump({"default_variant": "dec", "captured_us": {"dec": 10.0},
                                    "dec": [{"dims": {"N": 1}}]}))  # fmt: skip

    def single(base):
        def run():
            print(f"Correctness: PASSED\nPerformance: baseline_us={base}, kernel_us=5.0, "
                  f"speedup={base / 5:.2f}x")  # fmt: skip
            return 0

        return run

    args = argparse.Namespace(spec=str(spec), variant=None, baseline_us=None)
    assert benchmark._weighted(args, single(14.0)) == 0
    assert "PASSED" in capsys.readouterr().out
    assert benchmark._weighted(args, single(40.0)) == 1
    assert "BASELINE_NOT_CAPTURED_KERNEL" in capsys.readouterr().out


def test_the_device_asked_for_wins_over_what_torch_sees(monkeypatch):
    import argparse

    from xe_forge.frontend import cli, profile

    monkeypatch.setenv("XE_FORGE_DEVICE", "cuda:1")
    assert profile.device_type() == "cuda"
    a = argparse.Namespace(device="cpu")
    cli._use_device(a)
    assert a.device == "cpu" and os.environ["XE_FORGE_DEVICE"] == "cpu"
    a = argparse.Namespace(device=None)
    cli._use_device(a)
    assert a.device == "cpu"  # the exported choice carries on


def test_a_given_reference_is_never_the_locators_to_write():
    from xe_forge.claude.generator import _render

    def text(given):
        return _render("kernel-locator.md.j2", kernel_name="k", kernel_repo="/r",
                       profile_path="p.md", baseline_oracle=False, reference_given=given)  # fmt: skip

    assert "check it, never write it" in text(True) and "Do not write, replace" in text(True)
    assert "`test_kernels/k_pytorch.py`. Nothing else" not in text(True)
    assert "A standard reference, if one exists" in text(False)


def test_a_variant_with_several_entries_is_warned_about(tmp_path, caplog):
    from xe_forge.core.spec_loader import load_spec

    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump({"bench-gpu": [{"dims": {"N": 1}}, {"dims": {"N": 2}}]}))
    load_spec(spec)
    assert "only the first is used" in caplog.text


def test_the_tools_take_their_limits_and_binaries_as_flags(monkeypatch):
    import sys

    from xe_forge import skills

    monkeypatch.setenv("UNITRACE_BIN", "/opt/u/unitrace")
    seen = {}
    monkeypatch.setattr("argparse.ArgumentParser.parse_args",
                        lambda self, *a, **k: seen.setdefault(self.prog, self) and sys.exit(0))  # fmt: skip
    with pytest.raises(SystemExit):
        skills.main()
    parser = next(iter(seen.values()))
    sub = {a.dest: a for a in parser._actions}["skill"].choices
    prof = {a.dest: a.default for a in sub["profile"]._actions}
    assert prof["unitrace_bin"] == "/opt/u/unitrace"
    cap = {a.dest: a for a in sub["capture"]._actions}["capture_command"].choices["parse"]
    assert {a.dest: a.default for a in cap._actions}["min_known"] == 90.0
    pchk = {a.dest: a for a in sub["patch"]._actions}["patch_command"].choices["check"]
    assert {a.dest: a.default for a in pchk._actions}["margin"] == 0.1


def _gpu() -> str | None:
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else None


@pytest.mark.skipif(_gpu() is None, reason="the token reference scores on a GPU")
def test_the_agents_own_reference_runner_scores_the_tokens(tmp_path):
    """A model transformers cannot load: the agent's load() is the reference."""
    import json
    import subprocess
    import sys

    from xe_forge.skills import patch

    (tmp_path / "ref.py").write_text(
        "import torch\n"
        "from xe_forge.frontend.profile import Vocab\n"
        "class M(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        torch.manual_seed(0)\n"
        "        self.e = torch.nn.Embedding(64, 16)\n"
        "        self.o = torch.nn.Linear(16, 64)\n"
        "    def forward(self, x):\n"
        "        return self.o(self.e(x))\n"
        "def load(model, dtype, device):\n"
        "    return M().to(device, dtype), Vocab(lambda t: [ord(c) % 64 for c in t], list(range(64)))\n"
    )
    sys.path.insert(0, str(tmp_path))
    import ref  # the same model, to make the tokens the reference itself would pick
    import torch

    m, vocab = ref.load("m", torch.float32, "cpu")
    from xe_forge.frontend import profile

    prompt = profile.token_ids(vocab, 1, 4)[0]
    ids, got = list(prompt), []
    for _ in range(3):
        nxt = int(m(torch.tensor([ids]))[0, -1].argmax())
        got.append(nxt)
        ids.append(nxt)
    (tmp_path / "got.json").write_text(json.dumps([got]))
    out = subprocess.run([sys.executable, "-c", patch._REFERENCE, "m", "1", "4", "3", "0",
                          str(tmp_path / "got.json"), "", "float32", "", "2",
                          str(tmp_path / "ref.py")],
                         capture_output=True, text=True, env={**os.environ, "XE_FORGE_DEVICE": _gpu()})  # fmt: skip
    assert out.returncode == 0, out.stderr[-500:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["transformers"][0].endswith("ref.py")
    assert max(res["gaps"]["0"][0]) < 1e-3  # the reference's own tokens: no gap


@pytest.mark.parametrize("active, experts_read", [(None, 2), (16, 16)])
def test_a_grouped_gemm_is_bounded_without_a_routing_model(active, experts_read):
    """Top-k experts are certain; more only when measured (or stated by the naming agent)."""
    from xe_forge.frontend import analyze
    from xe_forge.frontend.ir import Naming, Workload

    n = k = 1024
    attrs = {"parts": [{"N": n, "K": k, "E": 64, "grouped": True, "rows": "token",
                        "w_bytes": 2, "a_bytes": 2, "o_bytes": 2}]}  # fmt: skip
    if active:
        attrs["active_experts"] = active
    w = Workload(workload_id="g", family="gemm", dims={"M": 4, "N": n, "K": k},
                 dtypes=["bfloat16"], layout=[], attrs=attrs, var_axes={}, calls=1,
                 device_us=10_000.0, share=1.0, ops=[], naming=Naming("kernel"))  # fmt: skip
    calib = {"bandwidth_gbs": 1000.0, "matmul_tflops": {"bfloat16": 100.0}}
    _, eff = analyze.gemm_bound(w, calib, {"topk": 2})
    rows = 4 * 2
    moved = rows * k * 2 + experts_read * k * n * 2 + rows * n * 2
    assert eff == pytest.approx(moved / (1000.0 * 1e3) / 10_000.0)
