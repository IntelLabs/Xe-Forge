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


def test_the_device_index_asked_for_reaches_naming_and_calibration(monkeypatch):
    import torch

    from xe_forge.frontend import calibrate, cli, profile

    monkeypatch.setenv("XE_FORGE_DEVICE", "xpu:1")
    assert profile.device() == "xpu:1" and profile.device_type() == "xpu"
    a = argparse.Namespace(device=None)
    cli._use_device(a)
    assert a.device == "xpu:1"
    seen = []
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda i=0: seen.append(("name", i)) or "g")
    monkeypatch.setattr(torch.xpu, "set_device", lambda i: seen.append(("set", i)))
    assert profile.device_info()["name"] == "g" and seen == [("name", 1)]
    calibrate.measure(read_bytes=16, n=2, iters=1, dtypes=("float32",))
    assert ("set", 1) in seen


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


@pytest.mark.parametrize("dev", [
    pytest.param(_gpu(), marks=pytest.mark.skipif(_gpu() is None, reason="no GPU here")),
    "cpu",  # a run on the host scores on the host
])  # fmt: skip
def test_the_agents_own_reference_runner_scores_the_tokens(tmp_path, dev):
    """A model transformers cannot load: the agent's load() is the reference, on the run's
    device."""
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
                         capture_output=True, text=True, env={**os.environ, "XE_FORGE_DEVICE": dev})  # fmt: skip
    assert out.returncode == 0, out.stderr[-500:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["transformers"][0].endswith("ref.py") and res["device"] == dev
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


@pytest.mark.parametrize("opt_ms, says, slower", [(1.0008, "NO MEASURABLE CHANGE", False),
                                                   (0.9990, "NO MEASURABLE CHANGE", False),
                                                   (1.10, "PERFORMANCE REGRESSION", True),
                                                   (0.80, "SUCCESS", False)])  # fmt: skip
def test_a_difference_inside_the_noise_band_is_no_regression(monkeypatch, opt_ms, says, slower):
    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.models import ExecutionResult

    ex = KernelBenchExecutor(device="cpu", min_effect=0.02)
    times = iter([1.0, opt_ms])
    monkeypatch.setattr(ex, "_check_correctness", lambda **k: True)
    monkeypatch.setattr(ex, "execute", lambda *a, **k: ExecutionResult(
        success=True, execution_time_ms=next(times), output_correct=True))  # fmt: skip
    got = ex.compare_kernels("orig", "opt", kernel_name="k")
    assert says in got.feedback_message and got.is_slower is slower
    assert (got.verdict == "INDISTINGUISHABLE") is (says == "NO MEASURABLE CHANGE")


@pytest.mark.parametrize("original_ms, slower, inside", [(0.98, True, False), (1.02, False, False),
                                                       (0.99, False, True)])  # fmt: skip
def test_the_noise_band_edges_are_changes(monkeypatch, original_ms, slower, inside):
    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.core.sycl_executor import SyclExecutor
    from xe_forge.models import ExecutionResult

    ex = KernelBenchExecutor(device="cpu", min_effect=0.02)
    times = iter([original_ms, 1.0])
    monkeypatch.setattr(ex, "_check_correctness", lambda **k: True)
    monkeypatch.setattr(ex, "execute", lambda *a, **k: ExecutionResult(
        success=True, execution_time_ms=next(times), output_correct=True))  # fmt: skip
    got = ex.compare_kernels("orig", "opt", kernel_name="k")
    assert got.is_slower is slower and (got.verdict == "INDISTINGUISHABLE") is inside

    monkeypatch.setenv("BENCHMARK_MIN_EFFECT", "0.02")
    sy = SyclExecutor.__new__(SyclExecutor)
    sy._backend = sy._build_dir = sy._cached_input_dir = None
    times = iter([original_ms, 1.0])
    monkeypatch.setattr(sy, "execute", lambda **k: ExecutionResult(
        success=True, execution_time_ms=next(times)))  # fmt: skip
    got = sy.compare_kernels("orig", "opt", input_dir="/nonexistent")
    assert got.is_slower is slower and (got.verdict == "INDISTINGUISHABLE") is inside


@pytest.mark.parametrize("band", ["0", "-0.01", "nan", "inf"])
def test_a_noise_band_that_is_not_finite_and_positive_is_refused(monkeypatch, band):
    from xe_forge.config import noise_band
    from xe_forge.core.executor import KernelBenchExecutor

    monkeypatch.setenv("BENCHMARK_MIN_EFFECT", band)
    with pytest.raises(ValueError, match="finite and above 0"):
        noise_band()
    with pytest.raises(ValueError, match="finite and above 0"):
        KernelBenchExecutor(device="cpu", min_effect=float(band))


@pytest.mark.parametrize("band, says", [("0", "finite and above 0"), ("abc", "must be a number")])
def test_a_bad_noise_band_is_refused_before_anything_is_timed(monkeypatch, capsys, band, says):
    import argparse

    from xe_forge.core.sycl_executor import SyclExecutor
    from xe_forge.frontend import cli
    from xe_forge.skills import benchmark

    monkeypatch.setenv("BENCHMARK_MIN_EFFECT", band)
    assert cli.main(["run", "launcher.sh"]) == 2
    assert says in capsys.readouterr().err
    monkeypatch.setattr(benchmark, "_trial_target", lambda a: pytest.fail("a trial was set up"))
    with pytest.raises(SystemExit) as exit_info:
        benchmark.run(argparse.Namespace(device=None))
    out = capsys.readouterr().out
    assert exit_info.value.code == 2 and "VERDICT: BAD_CONFIGURATION" in out and says in out
    with pytest.raises(ValueError, match=says):
        SyclExecutor(device_target="any")


def test_the_profile_tool_and_the_dspy_cli_refuse_a_bad_noise_band(monkeypatch, capsys):
    import sys

    from xe_forge import cli, skills

    monkeypatch.setenv("BENCHMARK_MIN_EFFECT", "0")
    monkeypatch.setattr(sys, "argv", ["xe-forge-skill", "profile", "kernel.py"])
    assert skills.main() == 2
    monkeypatch.setattr(sys, "argv", ["xe-forge"])
    monkeypatch.setattr(cli, "_parse_args", lambda: (None, argparse.Namespace()))
    assert cli.main() == 2
    assert capsys.readouterr().err.count("finite and above 0") == 2


def test_the_builtin_skill_prints_no_ratio_inside_the_noise_band(tmp_path, monkeypatch, capsys):
    from xe_forge.core.executor import KernelBenchExecutor
    from xe_forge.models import ExecutionResult
    from xe_forge.skills import benchmark

    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump({"weights": {"dec": 0.5, "pre": 0.5},
                                    "dec": [{"dims": {"N": 1}}], "pre": [{"dims": {"N": 2}}]}))  # fmt: skip
    for name in ("base.py", "opt.py"):
        (tmp_path / name).write_text("x = 1\n")
    times = iter([1.0, 1.01, 1.0, 0.5])  # dec inside the band, pre 2x faster
    monkeypatch.setattr(KernelBenchExecutor, "_check_correctness", lambda self, **k: True)
    monkeypatch.setattr(KernelBenchExecutor, "execute", lambda self, *a, **k: ExecutionResult(
        success=True, execution_time_ms=next(times), output_correct=True))  # fmt: skip
    args = argparse.Namespace(spec=str(spec), variant="dec", baseline_us=None, device="cpu",
                              baseline=str(tmp_path / "base.py"), dsl=None,
                              optimized=str(tmp_path / "opt.py"), reference=None)  # fmt: skip
    assert benchmark._run_builtin(args) == 0
    text = capsys.readouterr().out
    out = benchmark.parse_outcome(text)
    assert "speedup=none" in text and out["verdict"] == "INDISTINGUISHABLE"
    assert "speedup" not in out and out["correctness"] == "pass"
    times = iter([1.01])
    args.baseline_us = "1000"
    assert benchmark._run_builtin(args) == 0
    assert benchmark.parse_outcome(capsys.readouterr().out)["verdict"] == "INDISTINGUISHABLE"
    times, args.baseline_us = iter([1.0, 1.01, 1.0, 0.5]), None
    args.variant = None
    assert benchmark._weighted(args, lambda: benchmark._run_builtin(args)) == 0
    out = benchmark.parse_outcome(capsys.readouterr().out)
    assert out.get("verdict") != "SLOWER_ON"
    assert out["speedup"] == pytest.approx(1 / (0.5 + 0.5 / 2), abs=0.01)


@pytest.mark.skipif(_gpu() is not None, reason="needs a machine without a GPU")
def test_the_reference_refuses_the_host_when_a_gpu_run_asked_for_one(tmp_path):
    import subprocess
    import sys

    from xe_forge.skills import patch

    (tmp_path / "got.json").write_text("[[1]]")
    env = {k: v for k, v in os.environ.items() if k != "XE_FORGE_DEVICE"}
    out = subprocess.run([sys.executable, "-c", patch._REFERENCE, "m", "1", "4", "1", "0",
                          str(tmp_path / "got.json"), "", "float32", "", "1"],
                         capture_output=True, text=True, env=env)  # fmt: skip
    assert out.returncode != 0 and "sees no GPU" in out.stderr
