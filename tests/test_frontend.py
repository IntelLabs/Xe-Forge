"""Frontend unit tests: no device, no network; the agent runs against a stub ``claude``."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from xe_forge.dsl_registry import get_stages_for_dsl
from xe_forge.frontend import analyze, formula, manifest
from xe_forge.frontend.capture import _counters
from xe_forge.models import DSL, OptimizationStage

# ── OpenCL ──────────────────────────────────────────────────────────────────


def test_opencl_kernel_is_a_python_file():
    # The OpenCL C source is a string in a Python file that KernelBench imports.
    assert DSL.OPENCL.kernel_ext == ".py"
    assert DSL.OPENCL.code_language == "python"


def test_opencl_has_no_triton_stages():
    stages = set(get_stages_for_dsl("opencl"))
    assert OptimizationStage.ANALYSIS in stages
    for triton_only in (
        OptimizationStage.AUTOTUNING,
        OptimizationStage.TENSOR_DESCRIPTORS,
        OptimizationStage.PERSISTENT_KERNEL,
    ):
        assert triton_only not in stages


# ── counters: unknown is not zero ───────────────────────────────────────────


def _record(kernel, time_ns, read, write, util):
    return {
        "kernel": kernel,
        "time_ns": time_ns,
        "memory": {"read": {"bytes": read}, "write": {"bytes": write}},
        "counters": {"XVE_ACTIVE_UTILIZATION[%]": util},
    }


def test_counters_keep_unknown_as_none():
    out = _counters({"records": [_record("k", 1000, None, None, None)]})
    assert out["k"] == {"gb_per_s": None, "util_pct": None, "us": 1.0}


def test_counters_average_only_measured_records():
    out = _counters(
        {"records": [_record("k", 1000, 600, 400, 80.0), _record("k", 1000, None, 5, None)]}
    )
    # Only the first record measured both; its 1000 B / 1000 ns and 80 % are not halved.
    assert out["k"]["gb_per_s"] == 1.0
    assert out["k"]["util_pct"] == 80.0
    assert out["k"]["us"] == 2.0


def _workload(kernels):
    return SimpleNamespace(framework_meta={"kernels": kernels}, device_us=sum(kernels.values()))


def test_sol_ignores_kernels_with_unknown_counters():
    calib = {"bandwidth_gbs": 100.0}
    w = _workload({"k": 10.0})
    unknown = {"counters": {"k": {"gb_per_s": None, "util_pct": 90.0, "us": 10.0}}}
    assert analyze._sol_parts(w, calib, unknown) is None  # unmeasured, not 0% of peak
    known = {"counters": {"k": {"gb_per_s": 95.0, "util_pct": 10.0, "us": 10.0}}}
    sol, mem, comp = analyze._sol_parts(w, calib, known)
    assert sol == pytest.approx(0.95) and mem == pytest.approx(0.95)
    assert comp == pytest.approx(0.10)


def test_sol_needs_a_bandwidth_peak():
    w = _workload({"k": 10.0})
    ctx = {"counters": {"k": {"gb_per_s": 95.0, "util_pct": 10.0, "us": 10.0}}}
    assert analyze._sol_parts(w, {}, ctx) is None


# ── names into paths ────────────────────────────────────────────────────────


def test_sidecar_keeps_a_dotted_kernel_name(tmp_path):
    a = manifest.sidecar(tmp_path / "rms_norm.fwd__bench-t1", ".status.json")
    b = manifest.sidecar(tmp_path / "rms_norm.fwd__bench-t512", ".status.json")
    assert a.name == "rms_norm.fwd__bench-t1.status.json"
    assert a != b


def test_repo_path_is_one_filename_part():
    assert "/" not in manifest._flat("vllm/csrc")
    assert manifest._flat("vllm-xpu-kernels") == "vllm-xpu-kernels"


# ── formulas from agents are bounded ────────────────────────────────────────


@pytest.mark.parametrize("expr", ["10**10**10", "2**(T*H)", "(-2)**0.5", "1/0"])
def test_formula_rejects_unbounded_or_invalid(expr):
    start = time.monotonic()
    assert formula.evaluate(expr, {"T": 4096, "H": 4096}) is None
    assert time.monotonic() - start < 1.0


def test_formula_evaluates_costs():
    assert formula.evaluate("2*M*N*K", {"M": 2, "N": 3, "K": 4}) == 48.0
    assert formula.evaluate("M//2 + 1", {"M": 7}) == 4.0


# ── the agent runner, against a stub claude ─────────────────────────────────

STUB = """\
#!{python}
# First call: non-ASCII output, then error_max_turns. Resume: a detached sleeper, then the answer.
import json, os, subprocess, sys
def out(o): print(json.dumps(o, ensure_ascii=False, separators=(',', ':')))  # compact, as claude
argv = sys.argv[1:]
if "--resume" not in argv:
    out({{"type": "assistant", "text": "build \\u2014 caf\\u00e9 \\u2014 " * 200}})
    out({{"type": "result", "subtype": "error_max_turns", "session_id": "s-1"}})
    sys.exit(1)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
open("{pidfile}", "w").write(str(child.pid))
open("answer.json", "w").write(json.dumps({{"ok": True}}))
out({{"type": "result", "subtype": "success", "session_id": "s-1"}})
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it is dead for our purpose.
    status = Path(f"/proc/{pid}/status")
    return not (status.is_file() and "\nState:\tZ" in status.read_text())


def test_agent_resumes_after_non_ascii_and_kills_detached_work(tmp_path, monkeypatch):
    from xe_forge.frontend import agent

    bin_dir, pidfile = tmp_path / "bin", tmp_path / "child.pid"
    bin_dir.mkdir()
    stub = bin_dir / "claude"
    stub.write_text(STUB.format(python=sys.executable, pidfile=pidfile))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(agent, "RESUMES", 1)

    got = agent.run("frontend-resume.md.j2", {}, tmp_path / "work", "answer.json", "python", 5)

    assert got == {"ok": True}
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 15
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(pid), "the session's detached work outlived the step"


# ── stack check timeout log ─────────────────────────────────────────────────


def test_stack_check_timeout_log_is_text(tmp_path, monkeypatch):
    from xe_forge import config, external
    from xe_forge.skills import upstream

    monkeypatch.setattr(upstream, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        config, "get_config", lambda: SimpleNamespace(external=SimpleNamespace(stack_check="x"))
    )
    monkeypatch.setattr(external, "render_command", lambda *a, **k: ["true"])

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("x", 1, output=b"line one\nline two\n", stderr=b"err\n")

    monkeypatch.setattr(upstream.subprocess, "run", timeout)
    ok, why = upstream._stack_check([tmp_path], "k", "upstream")
    assert ok is False and "timed out" in why
    assert (tmp_path / "stack_check.log").read_text() == "line one\nline two\nerr\n"


# ── OpenVINO: the trace and unitrace name a kernel differently ──────────────


def test_openvino_counter_family_joins_both_names():
    from xe_forge.frontend.adapters.openvino import counter_family as f

    pairs = [
        ("rms_gpu_bfyx_opt__f16", "rms_gpu_bfyx_opt_1933771266323936851_0_0__sa"),
        ("ocl::rope::opt___f16", "rope_opt__4308312662623125788__sa"),
        ("concatenation_gpu_simple_ref__f16", "concatenation_gpu_simple_ref_48809118_1_0__sa"),
    ]
    for trace, unitrace in pairs:
        assert f(trace) == f(unitrace)
    # Not predictable from the plugin's name: left unmatched rather than guessed.
    assert f("ocl::sdpa::opt__f16") != f("sdpa_micro__prefill_14496116136220614062__sa")


def test_rekey_merges_instances_onto_trace_names():
    from xe_forge.frontend.adapters.openvino import counter_family
    from xe_forge.frontend.capture import _rekey

    counters = {
        "concatenation_gpu_simple_ref_1_0_0__sa": {"gb_per_s": 100.0, "util_pct": 10.0, "us": 3.0},
        "concatenation_gpu_simple_ref_2_1_0__sa": {"gb_per_s": 200.0, "util_pct": None, "us": 1.0},
        "gemm_kernel": {"gb_per_s": 50.0, "util_pct": 50.0, "us": 9.0},
    }
    kernels = {"concatenation_gpu_simple_ref__f16": 1.0, "undef": 5.0}
    out = _rekey(counters, kernels, counter_family)
    assert set(out) == {"concatenation_gpu_simple_ref__f16"}
    got = out["concatenation_gpu_simple_ref__f16"]
    assert got["gb_per_s"] == 125.0  # (100*3 + 200*1) / 4
    assert got["util_pct"] == 10.0  # only the instance that measured it
    assert got["us"] == 4.0


def test_optimize_records_the_repo_revision(tmp_path):
    from xe_forge.frontend.cli import _git_revision

    repo = tmp_path / "kernels"
    repo.mkdir()

    def git(*a):
        return subprocess.run(
            ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("remote", "add", "origin", "https://example.invalid/kernels.git")
    (repo / "k.cl").write_text("// v1\n")
    git("add", "k.cl")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "v1")

    rev = _git_revision(repo)
    assert rev["commit"] == git("rev-parse", "HEAD")
    assert rev["url"] == "https://example.invalid/kernels.git"
    assert rev["dirty"] is False
    (repo / "k.cl").write_text("// v2\n")
    assert _git_revision(repo)["dirty"] is True
    assert _git_revision(tmp_path / "missing")["commit"] is None


def test_optimize_session_finds_this_interpreter_first():
    from xe_forge.frontend.cli import _session_env

    first = _session_env()["PATH"].split(os.pathsep)[0]
    assert first == str(Path(sys.executable).parent)
