"""Frontend unit tests: no device, no network; the agent runs against a stub ``claude``."""

from __future__ import annotations

import json
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
from xe_forge.models import DSL, OptimizationStage
from xe_forge.skills.capture import _counters, _usable

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


OPENCL_KERNEL = """
from xe_forge import opencl

SRC = "__kernel void k(__global float* x) { x[get_global_id(0)] += 1.0f; }"


class Model:
    def forward(self, x):
        return opencl.run(SRC, "k", x)
"""


GLUON_KERNEL = """
import triton
import triton.language as tl
from triton.experimental import gluon


@gluon.jit
def k(x):
    pass


class Model:
    def forward(self, x):
        return k[(1,)](x, num_warps=4)
"""


@pytest.mark.parametrize("dsl, code, says, final", [
    ("opencl", OPENCL_KERNEL, None, None),
    ("opencl", OPENCL_KERNEL.replace("class Model", "class Other"), "MISSING: class Model",
     "MISSING: class Model"),
    ("triton", OPENCL_KERNEL, "MISSING: import triton", "Not valid Triton"),
    ("triton", "import triton\nimport triton.language as tl\n\nclass Model:\n    pass\n",
     "MISSING: @triton.jit decorator", None),
    ("gluon", GLUON_KERNEL, None, None),
    ("gluon", GLUON_KERNEL.replace("import triton\nimport triton.language as tl\n",
                                   "from triton.experimental.gluon import language as gl\n"),
     None, None),
    ("triton", GLUON_KERNEL.replace("@gluon.jit", "@triton.jit").replace(
        "import triton.language as tl\n", ""), "MISSING: import triton.language", None),
    ("gluon", GLUON_KERNEL.replace("num_warps=4", "num_warps=3"),
     "INVALID num_warps=3: Must be power of 2.", None),
])  # fmt: skip
def test_the_dspy_optimizer_holds_only_triton_kernels_to_triton(dsl, code, says, final):
    from xe_forge.agents.optimizer_agent import OptimizerAgent
    from xe_forge.agents.utils import SUCCESS_MESSAGE

    measured = []

    def compare_kernels(**kw):
        measured.append(kw["optimized_code"])
        return SimpleNamespace(optimized_correct=True, is_slower=False, speedup=1.5,
                               original_time_us=10.0, optimized_time_us=6.0,
                               original_tflops=None, optimized_tflops=None)  # fmt: skip

    agent = OptimizerAgent(executor=SimpleNamespace(compare_kernels=compare_kernels), dsl=dsl)
    tool, _ = agent._create_verify_tool(code, "k", [(4,)], None)
    assert tool.func(code) == (says or SUCCESS_MESSAGE)
    assert len(measured) == (says is None)  # the executor runs what passes the checks
    ok, speedup, _, _, err = agent._final_verify(code, code, "k", [(4,)], None, None)
    assert err == final and ok is (final is None)
    assert speedup == (1.5 if final is None else None)


@pytest.mark.parametrize("device", ["xpu", "cuda", "cpu"])
def test_dspy_prompts_offer_a_non_triton_dsl_only_its_own_stages(device):
    import re

    from xe_forge.agents.analyzer_agent import _build_issue_categories
    from xe_forge.prompts import PromptLibrary

    lib = PromptLibrary("opencl", device)
    for template in ("optimization_signature", "optimization_react_signature",
                     "algorithmic_signature", "coordinator_signature"):  # fmt: skip
        text = lib.render_for_signature(template)
        assert "OpenCL C" in text and "triton" not in text.lower(), template
        assert "TENSOR_DESCRIPTORS" not in text and "PERSISTENT_KERNEL" not in text, template
    cats = _build_issue_categories(DSL.OPENCL)
    assert "TENSOR DESCRIPTORS" not in cats and "AUTOTUNING" not in cats
    assert not re.search(r"\btl\.|triton|grf_mode|num_warps|BLOCK_M", cats, re.I)
    assert "TENSOR DESCRIPTORS" in _build_issue_categories(DSL.TRITON)
    assert "tl.load" in _build_issue_categories(DSL.TRITON)
    assert "tl.make_tensor_descriptor" in PromptLibrary("triton", device).render_for_signature(
        "optimization_signature")  # fmt: skip


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


def test_records_without_a_time_are_no_counters():
    """Records with samples and launches but no time carry no counter."""
    empty = [
        {"kernel": "k", "time_ns": None, "samples": 196, "launches": 196, "counters": {},
         "memory": {"read": {"bytes": None}, "write": {"bytes": None}}}
    ]  # fmt: skip
    assert _usable(empty) == []  # so the no-records branch runs and says so
    assert _counters({"records": empty}) == {}  # and nothing would have survived anyway
    assert len(_usable([*empty, _record("k", 1000, 600, 400, 80.0)])) == 1


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


@pytest.mark.parametrize(
    "expr",
    ["10**10**10", "2**(T*H)", "(-2)**0.5", "1/0", "min((0-1)**0.5, 1)", "max((-T)**0.5, 1)"],
)
def test_formula_rejects_unbounded_or_invalid(expr):
    start = time.monotonic()
    assert formula.evaluate(expr, {"T": 4096, "H": 4096}) is None
    assert time.monotonic() - start < 1.0


def test_formula_evaluates_costs():
    assert formula.evaluate("2*M*N*K", {"M": 2, "N": 3, "K": 4}) == 48.0
    assert formula.evaluate("M//2 + 1", {"M": 7}) == 4.0


def test_formula_caps_a_read_by_a_window():
    """An attention's KV read: the context, capped by a window that is -1 for none."""
    kv = "B*(K if W < 0 else min(K, W + 1))"
    assert formula.names(kv) == {"B", "K", "W"}  # min is not a dim
    assert formula.evaluate(kv, {"B": 8, "K": 4343, "W": -1}) == 8 * 4343
    assert formula.evaluate(kv, {"B": 8, "K": 4343, "W": 255}) == 8 * 256
    assert formula.evaluate("max(A, B) - 2*B", {"A": 1, "B": 2}) is None  # a cost is >= 0
    assert formula.names("print(A)") is None and formula.names("min(*A)") is None


def _windowed(window: int, us: float):
    from xe_forge.frontend.ir import CapturedOp, TensorArg

    return CapturedOp(id=f"op-{window}", framework_op="fa::fwd", args=[TensorArg([8, 128], "bfloat16"),
                      None], scalars=[None, str(window)], kernel_symbols={"fmha": us}, calls=1,
                      device_us=us)  # fmt: skip


def test_every_runtime_size_at_a_token_count_reaches_the_spec(tmp_path):
    """Decode steps share a token count but not a context length: each is a variant of its
    own, and the inputs check reads each one's own calls and time."""
    from xe_forge.frontend import canonical, ir, manifest, naming
    from xe_forge.frontend.analyze import Entry
    from xe_forge.skills.inputs import shares

    def op(tokens, ctx, us):
        return ir.CapturedOp(id=f"op-{tokens}-{ctx}", framework_op="fa::fwd",
                             args=[ir.TensorArg([tokens, 128], "bfloat16"), None],
                             scalars=[None, str(ctx)], kernel_symbols={"fmha": us}, calls=1,
                             device_us=us,
                             framework_meta={"named": {"route": "kernel", "name": "fmha",
                                                       "family": "attention",
                                                       "tolerance": {"rtol": 0.02, "atol": 0.01}}})  # fmt: skip

    ops = [op(512, 512, 90.0), op(4, 513, 10.0), op(4, 514, 11.0), op(4, 515, 12.0)]
    (w,) = canonical.workloads(ops, 123.0, naming.name_op)
    spec, record = manifest.build_spec(Entry(name="fmha", repo="r", dsl="sycl", workloads=[w],
                                             decisions=[], target="kernel"))  # fmt: skip
    dims = sorted((v[0]["dims"]["ARG0_D0"], v[0]["dims"]["ARG1"]) for k, v in spec.items()
                  if k.startswith("bench"))  # fmt: skip
    assert dims == [(4, 513), (4, 514), (4, 515), (512, 512)]
    cap = tmp_path / "capture.json"
    ir.dump(ir.CaptureRun(run={}, ops=ops, workloads=[w]), cap)
    got = shares({"dataset": str(cap), "variants": record})
    assert sorted(v["device_us"] for v in got.values()) == [10.0, 11.0, 12.0, 90.0]
    assert spec["bench-t512"][0]["rtol"] == 0.02 and spec["bench-t512"][0]["atol"] == 0.01
    # The tolerance is the naming agent's judgment: without one there is no spec, no default.
    w.naming.tolerance = None
    assert manifest.build_spec(Entry(name="fmha", repo="r", dsl="sycl", workloads=[w],
                                     decisions=[], target="kernel")) is None  # fmt: skip
    listing = {"fa::fwd": {"args": [], "dims": {}}}
    (tmp_path / "r").mkdir()
    bad = {"route": "kernel", "name": "fmha", "repo": "r", "dsl": "sycl",
           "tolerance": {"rtol": 0, "atol": 0.01}}  # fmt: skip
    got, problems = naming.keep_valid({"ops": {"fa::fwd": bad}, "model": {}}, listing, tmp_path)
    assert "tolerance" not in got["ops"]["fa::fwd"] and "that tolerance dropped" in problems[0]


def _row_capture(row, ops_us):
    """A capture at ``row``: prefill at batch x input tokens, decode at batch, ``ops_us`` each."""
    from xe_forge.frontend import ir

    tokens = (row["batch"] * row["input_len"], row["batch"])
    ops = [ir.CapturedOp(id=f"op{t}", framework_op="aten::gelu",
                         args=[ir.TensorArg([t, 128], "bfloat16")], kernel_symbols={"gelu": us},
                         calls=2, device_us=us) for t, us in zip(tokens, ops_us, strict=True)]  # fmt: skip
    rt = {k: row[k] for k in ("batch", "input_len", "output_len")}
    return ir.CaptureRun(run={"runtime": rt, "total_device_us": sum(ops_us),
                              "attributed_device_us": sum(ops_us)}, ops=ops)  # fmt: skip


def test_rows_merge_into_one_workload_spanning_every_rows_shapes():
    """Each row's ops scaled to its weight of the anchor's time, time per call kept: the
    same op at another row's token counts is more shapes of one workload."""
    from xe_forge.frontend import rows

    anchor = {"batch": 8, "input_len": 64, "output_len": 4, "weight": 0.75}
    other = {"batch": 1, "input_len": 64, "output_len": 4, "weight": 0.25}
    merged = rows.merge([_row_capture(anchor, [90.0, 10.0]), _row_capture(other, [30.0, 10.0])],
                        [anchor, other])  # fmt: skip
    (w,) = merged.workloads
    hist = next(iter(w.var_axes.values()))
    assert {t: h["device_us"] for t, h in hist.items()} == {"512": 67.5, "8": 7.5, "64": 18.75,
                                                             "1": 6.25}  # fmt: skip
    assert merged.run["total_device_us"] == 100.0 and w.device_us == 100.0
    op = next(o for o in merged.ops if o.id == "r02:op64")
    assert op.device_us / op.calls == 15.0  # time per call is the row's own
    assert op.framework_meta["row"] == 2 and merged.run["runtime"]["rows"][1] == other


def test_rows_merge_each_kernels_counters_by_its_scaled_time():
    from xe_forge.frontend import rows

    anchor = {"batch": 8, "input_len": 64, "output_len": 4, "weight": 0.5}
    other = {"batch": 1, "input_len": 64, "output_len": 4, "weight": 0.5}
    caps = [_row_capture(anchor, [90.0, 10.0]), _row_capture(other, [40.0, 10.0])]
    caps[0].run["counters"] = {"gelu": {"us": 100.0, "gb_per_s": 400.0, "util_pct": 10.0}}
    caps[1].run["counters"] = {"gelu": {"us": 50.0, "gb_per_s": 100.0, "util_pct": None}}
    got = rows.merge(caps, [anchor, other]).run["counters"]["gelu"]
    # the second row counts as much time as the first (weight 0.5 of 100 us each)
    assert got == {"us": 100.0, "gb_per_s": 250.0, "util_pct": 10.0}


def test_rows_merge_scales_each_kernels_unitrace_time_like_its_counters():
    from xe_forge.frontend import rows

    anchor = {"batch": 8, "input_len": 64, "output_len": 4, "weight": 0.1}
    other = {"batch": 1, "input_len": 64, "output_len": 4, "weight": 0.9}
    caps = [_row_capture(anchor, [90.0, 10.0]), _row_capture(other, [40.0, 10.0])]
    for op in caps[1].ops:  # another workload: gelu runs in the anchor row only
        op.framework_op, op.kernel_symbols = "aten::silu", {"silu": op.device_us}
    caps[0].run["counters"] = {"gelu": {"us": 100.0, "gb_per_s": 400.0, "util_pct": 10.0}}
    caps[0].run["unitrace_kernels"] = {"gelu": {"us": 100.0, "calls": 4},
                                       "both": {"us": 100.0, "calls": 4}}  # fmt: skip
    caps[1].run["unitrace_kernels"] = {"silu": {"us": 50.0, "calls": 4},
                                       "both": {"us": 50.0, "calls": 4}}  # fmt: skip
    merged = rows.merge(caps, [anchor, other])
    ran = merged.run["unitrace_kernels"]
    assert ran["gelu"]["us"] == pytest.approx(10.0) and ran["silu"]["us"] == pytest.approx(90.0)
    assert ran["both"]["us"] == pytest.approx(100.0)  # 0.1 x 100 + 0.9 x 100/50 x 50
    w = next(w for w in merged.workloads if "gelu" in w.framework_meta["kernels"])
    calib = {"bandwidth_gbs": 1000.0}
    assert analyze._sol_parts(w, calib, analyze.context(merged)) is not None  # fully sampled


def test_rows_file_puts_the_largest_row_first_and_refuses_a_bad_one(tmp_path):
    from xe_forge.frontend import rows

    f = tmp_path / "rows.yaml"
    f.write_text("rows:\n - {batch: 1, input_len: 4096, output_len: 8, weight: 0.2}\n"
                 " - {batch: 32, input_len: 512, output_len: 8, weight: 0.8}\n")  # fmt: skip
    assert [r["batch"] for r in rows.load(f)] == [32, 1]
    for bad in ("rows: []", "rows:\n - {batch: 1, input_len: 8, output_len: 8, weight: 0.5}",
                "rows:\n - {batch: 1, input_len: 8, weight: 1}",
                "rows:\n - {batch: 0, input_len: 8, output_len: 8, weight: 1}",
                "rows: [a: {b"):  # fmt: skip
        f.write_text(bad)
        with pytest.raises(ValueError):
            rows.load(f)


@pytest.mark.parametrize("ignore_env, kept, kept_batch", [(False, False, 1), (False, True, 1),
                                                         (False, True, 2), (True, False, 1)])  # fmt: skip
def test_other_rows_rerun_the_launcher_at_their_size_with_no_agent(
    tmp_path, monkeypatch, ignore_env, kept, kept_batch
):
    from xe_forge.frontend import cli, ir
    from xe_forge.skills.capture import _sha256

    anchor = {"batch": 8, "input_len": 64, "output_len": 4, "weight": 0.5}
    other = {"batch": 1, "input_len": 64, "output_len": 4, "weight": 0.5}
    (tmp_path / "launcher").mkdir()
    (tmp_path / "launcher" / "run.sh").write_text("#!/bin/sh\n")
    (tmp_path / "names.json").write_text("{}")
    ir.dump(_row_capture(anchor, [90.0, 10.0]), tmp_path / "capture.json")
    raw = tmp_path / "rows" / "02" / "raw"
    if kept:  # a row this launcher already captured at its size is read again, not rerun
        raw.mkdir(parents=True)
        sha = _sha256(tmp_path / "launcher" / "run.sh")
        rt = {"batch": kept_batch, "input_len": 64, "output_len": 4}
        (raw / "run.json").write_text(json.dumps({"launcher_sha256": sha, "runtime": rt}))
    calls = []

    def tool(py, *argv, problems=None, **env):
        calls.append(argv[1])
        if argv[1] == "run":
            assert env["XF_BATCH"] == "1" and env["FRAMEWORK_PYTHON"] == "/fw/python"
        else:
            assert "--names" in argv and py == "/fw/python"
            ir.dump(_row_capture(anchor if ignore_env else other, [30.0, 10.0]), argv[4])
        return 0

    monkeypatch.setattr(cli, "_tool", tool)
    got, problem = cli.capture_rows(SimpleNamespace(rows=[anchor, other]), tmp_path, "/fw/python")
    assert calls == (["parse"] if kept and kept_batch == 1 else ["run", "parse"])
    assert (got is None) == ignore_env == ("XF_BATCH=1" in (problem or ""))
    if got is not None:
        assert len(ir.load(got).ops) == 4


@pytest.mark.parametrize("config_args, n", [(None, 1), (["ARG1"], 2), (["arg1"], 2)])
def test_config_args_make_each_value_a_workload(config_args, n):
    """A window is a scalar like a size, so only the naming agent's config_args split on it."""
    from xe_forge.frontend import canonical, naming

    ops = [_windowed(-1, 130.0), _windowed(255, 14.0)]
    for op in ops:
        op.framework_meta["named"] = {"route": "kernel", "name": "fmha", "family": "attention",
                                      **({"config_args": config_args} if config_args else {})}  # fmt: skip
    got = canonical.workloads(ops, 144.0, naming.name_op)
    assert len(got) == n
    if n == 2:
        assert sorted(w.dims["ARG1"] for w in got) == [-1, 255]


def test_naming_refuses_config_args_that_are_not_scalar_dims(tmp_path):
    from xe_forge.frontend import naming

    listing = {"op": {"args": [{"name": "q", "shape": [8, 128]}, {"name": "w", "value": "255"}],
                      "dims": {"ARG0_D0": 8, "ARG0_D1": 128, "ARG1": 255}}}  # fmt: skip
    entry = {"route": "aten", "config_args": ["ARG1"]}
    assert naming.validate({"ops": {"op": entry}}, listing, tmp_path) == []
    for bad in (["ARG0_D0"], "ARG1", ["nope"]):
        problems = naming.validate({"ops": {"op": dict(entry, config_args=bad)}}, listing, tmp_path)
        assert any("config_args" in p for p in problems), bad


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


CODEX_STUB = """\
#!{python}
# `codex exec --json`: first call stops with no answer; `exec ... resume <id>` finishes.
import json, subprocess, sys
def out(o): print(json.dumps(o, separators=(',', ':')))
argv = sys.argv[1:]
assert argv[0] == "exec" and "--json" in argv, argv
out({{"type": "thread.started", "thread_id": "t-1"}})
if "resume" not in argv:
    out({{"type": "item.completed", "item": {{"type": "agent_message", "text": "caf\\u00e9 " * 200}}}})
    out({{"type": "turn.completed", "usage": {{}}}})
    sys.exit(0)
assert argv[argv.index("resume") + 1] == "t-1", argv
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
open("{pidfile}", "w").write(str(child.pid))
open("answer.json", "w").write(json.dumps({{"ok": True}}))
out({{"type": "turn.completed", "usage": {{}}}})
"""

STUBS = {"claude": STUB, "codex": CODEX_STUB}


def _stub_cli(tmp_path, monkeypatch, backend: str, text: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / backend
    stub.write_text(text)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XE_FORGE_AGENT", backend)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it is dead for our purpose.
    status = Path(f"/proc/{pid}/status")
    return not (status.is_file() and "\nState:\tZ" in status.read_text())


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_agent_resumes_after_non_ascii_and_kills_detached_work(tmp_path, monkeypatch, backend):
    from xe_forge.frontend import agent

    pidfile = tmp_path / "child.pid"
    _stub_cli(tmp_path, monkeypatch, backend,
              STUBS[backend].format(python=sys.executable, pidfile=pidfile))  # fmt: skip
    monkeypatch.setattr(agent, "RESUMES", 1)

    got = agent.run("frontend-resume.md.j2", {}, tmp_path / "work", "answer.json", "python", 5)

    assert got == {"ok": True}
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 15
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(pid), "the session's detached work outlived the step"


ONESHOT_STUB = """\
#!{python}
# Records its argv and cwd, checks the staged input, answers the way its CLI does.
import json, os, sys
argv = sys.argv[1:]
json.dump({{"argv": argv, "cwd": os.getcwd()}}, open("{record}", "w"))
assert open("inputs/config.json").read() == '{{"x": 1}}'
if {reject}:
    print('{reject_line}'); sys.exit(1)
answer = {{"rows": [{{"batch": 1}}]}}
if "{backend}" == "codex":
    open(argv[argv.index("-o") + 1], "w").write(json.dumps(answer))
    print(json.dumps({{"type": "turn.completed"}}))
else:
    print(json.dumps({{"type": "result", "structured_output": answer}}))
"""

REJECT = {"claude": '{"type":"system","subtype":"api_retry","error_status":401}',
          "codex": '{"type":"turn.failed","error":{"message":"401 Unauthorized"}}'}  # fmt: skip


@pytest.mark.parametrize("backend", ["claude", "codex"])
@pytest.mark.parametrize("reject", [False, True])
def test_oneshot_stages_inputs_and_reads_the_answer(tmp_path, monkeypatch, backend, reject):
    from xe_forge import agent_cli as agents
    from xe_forge.frontend import agent

    record = tmp_path / "argv.json"
    _stub_cli(tmp_path, monkeypatch, backend, ONESHOT_STUB.format(
        python=sys.executable, record=record, backend=backend, reject=reject,
        reject_line=REJECT[backend]))  # fmt: skip
    schema = {"type": "object", "properties": {"rows": {"type": "array"}}}
    if reject:
        with pytest.raises(agents.AgentError, match="rejected"):
            agent.oneshot("suggest rows", {"config.json": '{"x": 1}'}, schema)
        return
    assert agent.oneshot("suggest rows", {"config.json": '{"x": 1}'}, schema) == {
        "rows": [{"batch": 1}]}  # fmt: skip
    argv = json.loads(record.read_text())["argv"]
    assert argv[-1] == "suggest rows" if backend == "codex" else argv[1] == "suggest rows"


def test_oneshot_timeout_kills_the_sessions_detached_work(tmp_path, monkeypatch):
    from xe_forge.frontend import agent

    pidfile = tmp_path / "child.pid"
    _stub_cli(tmp_path, monkeypatch, "claude",
              f"#!/bin/sh\nsleep 60 &\necho $! > {pidfile}\nsleep 60\n")  # fmt: skip
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        agent.oneshot("p", {}, {"type": "object"}, timeout=1)
    assert time.monotonic() - start < 5
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 15
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(pid), "the session's detached work outlived the timeout"


def test_tool_profiles_map_to_each_cli(tmp_path, monkeypatch):
    """Read-only is confined and shell-free on claude; on codex reads go through a
    read-only sandboxed shell with no network. Nothing beyond the profile is granted."""
    from xe_forge import agent_cli as agents

    for name in ("claude", "codex"):
        _stub_cli(tmp_path, monkeypatch, name, "#!/bin/sh\n")
    schema, answer = tmp_path / "s.json", tmp_path / "a.json"
    schema.write_text("{}")
    claude = agents.get("claude").oneshot("p", schema, answer, agents.READ_ONLY)
    assert "--restricted" in claude and "dontAsk" in claude
    assert claude[claude.index("--tools") + 1] == "Read,Glob,Grep"
    codex = agents.get("codex").oneshot("p", schema, answer, agents.READ_ONLY)
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    assert "sandbox_workspace_write.network_access=false" in codex
    assert "web_search=disabled" in codex
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex
    none = agents.get("claude").oneshot("p", schema, answer, {"read": False})
    assert none[none.index("--tools") + 1] == "" and "--allowedTools" not in none
    assert "features.shell_tool=false" in agents.get("codex").oneshot(
        "p", schema, answer, {"read": False})  # fmt: skip
    with pytest.raises(agents.AgentError, match="unknown agent CLI"):
        agents.get("gemini")


def test_codex_reads_the_generated_workspace(tmp_path):
    from xe_forge import agent_cli as agents

    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    (tmp_path / ".claude" / "commands").mkdir()
    (tmp_path / "CLAUDE.md").write_text("# Workflow\n")
    (tmp_path / ".claude" / "commands" / "optimize-kernel.md").write_text("Optimize `$ARGUMENTS`.")
    (tmp_path / ".claude" / "agents" / "tool-runner.md").write_text(
        '---\nname: tool-runner\ndescription: "Run tools"\ntools: Bash\n---\nRun "it".\n'
    )
    codex = agents.get("codex")
    codex.prepare(tmp_path)
    text = (tmp_path / "AGENTS.md").read_text()
    assert text.endswith("# Workflow\n") and ".claude/agents/<name>.md` (tool-runner)" in text
    toml = (tmp_path / ".codex" / "agents" / "tool-runner.toml").read_text()
    assert 'name = "tool-runner"' in toml and 'developer_instructions = "Run \\"it\\"."' in toml
    assert codex.task(tmp_path, "rmsnorm") == "Optimize `rmsnorm`."
    assert agents.get("claude").task(tmp_path, "rmsnorm") == "/optimize-kernel rmsnorm"
    assert agents.get("codex").auth_failed(REJECT["codex"])
    assert agents.get("claude").auth_failed(REJECT["claude"])
    assert not agents.get("claude").auth_failed('{"type":"assistant","text":"401"}')


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


# ── capture tools: the instruments and checks around an agent's launcher ────

UNITRACE_STUB = """#!/bin/bash
# Stub unitrace: run the launcher, then write a device-timing table where -o pointed.
while [[ $1 == -* ]]; do [[ $1 == -o ]] && { out=$2; shift; }; [[ $1 == --group ]] && shift; shift; done
"$@" || exit $?
cat > "${out%.txt}.123.txt" <<'T'
== Device timing ==
Kernel, Calls, Time (ns), Time (%), Average (ns), Min (ns), Max (ns)
"rms_opt_123_0_0__sa[SIMD16 {1; 1; 1} {16; 1; 1}]", 4, 8000, 40, 2000, 2000, 2000
"gemm_kernel", 2, 12000, 60, 6000, 6000, 6000
T
# --chrome-kernel-logging: one gpu_op per launch, into the working directory.
r='rms_opt_123_0_0__sa[SIMD16 {1; 1; 1} {16; 1; 1}]'
ev() { printf '{"ph": "X", "cat": "gpu_op", "name": "%s", "ts": %s, "dur": %s},' "$1" "$2" "$3"; }
{ printf '{"traceEvents": ['; ev gemm_kernel 1 6.0; ev "$r" 10 2.0; ev "$r" 20 2.0
  ev gemm_kernel 30 6.0; ev "$r" 40 2.0; ev "$r" 50 2.0
  printf '{"ph": "M", "name": "process_name"}]}'; } > python.123.json
"""

LAUNCHER = """#!/bin/bash
cat > "$RAW_DIR/meta.json" <<'J'
{"framework": "toyfw", "framework_version": "1", "model": {"name": "toy"},
 "runtime": {"batch": 1, "precision": "float16", "quantization": "float16"}, "device": {"type": "xpu"}}
J
cat > "$RAW_DIR/ops.json" <<'J'
{"rms": {"framework_op": "toy::RMS", "kernels": ["^rms_opt"], "calls": 4,
         "args": [{"name": "x", "shape": [4, 896], "dtype": "float16"}]},
 "fc": {"framework_op": "toy::FC", "kernels": ["^gemm_kernel$"%s],
        "args": [{"name": "x", "shape": [4, 896], "dtype": "float16"}]}}
J
"""


def _tool(*argv):
    r = subprocess.run([sys.executable, "-m", "xe_forge.skills", *argv], capture_output=True,
                       text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))  # fmt: skip
    return r.returncode, r.stdout + r.stderr


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_capture_tools_on_an_agent_op_map(tmp_path):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER % "")
    raw = tmp_path / "raw"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))
    assert code == 0, out
    assert json.loads((raw / "unitrace_kernels.json").read_text())["rms_opt_123_0_0__sa"] == {
        "us": 8.0, "calls": 4}  # fmt: skip
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 0, out
    cap = json.loads((tmp_path / "capture.json").read_text())
    assert cap["run"]["attributed_pct"] == 100.0
    assert cap["run"]["launcher_sha256"]
    assert {o["framework_op"]: o["device_us"] for o in cap["ops"]} == {"toy::FC": 12.0,
                                                                       "toy::RMS": 8.0}  # fmt: skip


# Two identical runs of the workload, and a stray launch before them -- the lazy allocation
# or one-off init every real launcher makes. The timeline is then an odd length, which is all
# the old check tested for.
UNITRACE_STUB_STRAY = UNITRACE_STUB.replace(
    "{ printf '{\"traceEvents\": ['; ev gemm_kernel 1 6.0;",
    "{ printf '{\"traceEvents\": ['; ev one_off_init 0 1.0; ev gemm_kernel 1 6.0;",
)

# A launcher that ran the workload twice, honours XE_FORGE_TRACE_STEPS, and reports which
# pass it is in: $RAW_DIR differs between the two, so each pass leaves its own note.
LAUNCHER_TWICE = """#!/bin/bash
echo "${XE_FORGE_TRACE_STEPS:-unset}" > "$RAW_DIR/steps_seen"
# Beside the launcher, so it survives the instrumented pass's scratch being removed: one
# line per pass, in order, which is the only way to see what the second pass was given.
echo "${XE_FORGE_TRACE_STEPS:-unset}" >> "$(dirname "$0")/steps_log"
cat > "$RAW_DIR/meta.json" <<J
{"framework": "toyfw", "framework_version": "1", "model": {"name": "toy"},
 "runtime": {"batch": 1, "precision": "float16", "quantization": "float16", "steps": ${XE_FORGE_TRACE_STEPS:-50}},
 "device": {"type": "xpu"}, "repeats": 2}
J
cat > "$RAW_DIR/ops.json" <<'J'
{"rms": {"framework_op": "toy::RMS", "kernels": ["^rms_opt"], "calls": 2,
         "args": [{"name": "x", "shape": [4, 896], "dtype": "float16"}]},
 "fc": {"framework_op": "toy::FC", "kernels": ["^gemm_kernel$"],
        "args": [{"name": "x", "shape": [4, 896], "dtype": "float16"}]}}
J
"""


def test_capture_parse_takes_two_identical_runs_with_a_stray_launch(tmp_path):
    """The window is checked by the work in it, not by the timeline's length being even."""
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB_STRAY)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER_TWICE)
    raw = tmp_path / "raw"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))
    assert code == 0, out
    t = json.loads((raw / "unitrace_launches.json").read_text())
    assert t["total"] % t["repeats"], "the point of this test is an odd timeline"
    assert t["match"] == 1.0 and t["stray"] == 1
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 0, out


def test_capture_parse_refuses_a_window_that_is_not_the_run_before_it(tmp_path):
    """A launcher whose two runs differ is still caught: only the second one was warm."""
    half = UNITRACE_STUB.replace('ev gemm_kernel 30 6.0; ev "$r" 40 2.0; ev "$r" 50 2.0',
                                 "ev other_kernel 30 6.0; ev other_kernel 40 2.0")  # fmt: skip
    unitrace = _exe(tmp_path / "unitrace", half)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER_TWICE)
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 1
    assert "not the work the run before it did" in out


def test_capture_run_shortens_only_the_instrumented_pass(tmp_path):
    """--trace-steps reaches the unitrace pass and never the timed one."""
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB_STRAY)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER_TWICE)
    raw = tmp_path / "raw"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace",
                      str(unitrace), "--trace-steps", "2")  # fmt: skip
    assert code == 0, out
    # The timed pass writes $RAW_DIR itself; the instrumented one writes the scratch copy,
    # which `capture run` removes -- so the note that survives is the timed pass's.
    assert (raw / "steps_seen").read_text().strip() == "unset"
    assert (tmp_path / "steps_log").read_text().split() == ["unset", "2"]
    # And the capture describes the timed pass: 50 steps, whatever the other one ran.
    assert json.loads((raw / "run.json").read_text())["runtime"]["steps"] == 50
    assert json.loads((raw / "run.json").read_text())["trace_steps"] == 2
    _, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert "over 2 step(s)" in out, out


@pytest.mark.parametrize("exported", [None, "2"])
def test_capture_run_trace_steps_zero_measures_the_whole_workload_twice(tmp_path, monkeypatch,
                                                                        exported):  # fmt: skip
    """0 is the whole workload under unitrace too, and it holds against an exported value."""
    if exported:
        monkeypatch.setenv("XE_FORGE_TRACE_STEPS", exported)
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB_STRAY)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER_TWICE)
    raw = tmp_path / "raw"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace",
                      str(unitrace), "--trace-steps", "0")  # fmt: skip
    assert code == 0, out
    assert (tmp_path / "steps_log").read_text().split() == ["unset", "unset"]
    assert json.loads((raw / "run.json").read_text())["trace_steps"] is None


def test_capture_run_says_when_the_launcher_ignores_trace_steps(tmp_path):
    """A launcher written before the variable keeps working, and is not claimed to be short."""
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER % "")  # never reads the variable
    raw = tmp_path / "raw"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace",
                      str(unitrace), "--trace-steps", "2")  # fmt: skip
    assert code == 0, out
    assert "does not read XE_FORGE_TRACE_STEPS" in out
    assert json.loads((raw / "run.json").read_text())["trace_steps"] is None
    _, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert "step(s)" not in out, out


def test_capture_run_keeps_an_exported_trace_steps_out_of_the_timed_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("XE_FORGE_TRACE_STEPS", "2")
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB_STRAY)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER_TWICE)
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    assert (raw / "steps_seen").read_text().strip() == "unset"


def test_capture_parse_refuses_invented_or_double_claimed_kernels(tmp_path):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER % ', "^rms_opt", "^no_such_kernel"')
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 1
    assert "matches no kernel unitrace saw" in out and "claimed by both" in out


def test_capture_parse_refuses_a_tampered_raw(tmp_path):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER % "")
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    (raw / "unitrace_kernels.json").write_text("{}")
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 1 and "changed since" in out


def test_bound_tool_lists_workloads_by_share(tmp_path):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", LAUNCHER % "")
    raw, cap = tmp_path / "raw", tmp_path / "capture.json"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    assert _tool("capture", "parse", str(raw), "-o", str(cap))[0] == 0
    code, out = _tool("bound", str(cap), "--json", "--min-share", "0")
    assert code == 0, out
    rows = json.loads(out)
    assert [r["op"] for r in rows] == ["toy::FC", "toy::RMS"]  # costliest first
    assert rows[0]["share"] == 0.6 and rows[0]["eff"] is None  # no calibration, no counters
    # A draft names.json is applied first: what the naming agent's answer makes of the capture.
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"ops": {"toy::RMS": {"route": "kernel", "name": "rms_v2"}}}))
    code, out = _tool("bound", str(cap), "--json", "--min-share", "0", "--names", str(names))
    rows, _, problem = out.partition("bound: PROBLEM")  # the run's weights check on the draft
    assert code == 0 and "rms_v2" in [r["kernel"] for r in json.loads(rows)], out
    assert "no GEMM's weight was captured" in problem  # its FC has no weight argument
    # With the namer's ops.json, the draft is validated as the run will: what it drops is shown.
    listing = tmp_path / "ops.json"
    listing.write_text(json.dumps({"toy::RMS": {"args": [], "dims": {}}}))
    code, out = _tool("bound", str(cap), "--json", "--min-share", "0", "--names", str(names),
                      "--ops", str(listing), "--repos", str(tmp_path))  # fmt: skip
    assert code == 0 and "bound: INVALID 'toy::RMS': repo None" in out, out
    assert "rms_v2" not in out.partition("bound:")[0]  # applied as the run keeps it: unnamed


def test_bound_with_draft_names_bounds_with_their_model_facts(tmp_path, monkeypatch, capsys):
    import argparse

    from xe_forge.frontend import ir
    from xe_forge.skills import bound

    cap = tmp_path / "capture.json"
    ir.dump(ir.CaptureRun(run={"total_device_us": 1.0}, ops=[]), cap)
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"ops": {}, "model": {"topk": 4}}))
    seen = []
    real = analyze.context
    monkeypatch.setattr(analyze, "context", lambda c: seen.append(real(c)) or seen[-1])
    args = argparse.Namespace(capture=str(cap), names=str(names), ops=None, repos=None,
                              calibration=None, min_share=0.0, json=True)  # fmt: skip
    bound.run(args)
    assert seen[0]["topk"] == 4


def test_a_cost_over_unknown_dims_loses_the_cost_not_the_naming(tmp_path):
    """A GEMM named right with a cost over M/N/K (not dims of the op) keeps its gemm."""
    from xe_forge.frontend import naming

    (tmp_path / "repo").mkdir()
    listing = {"op": {"args": [{"name": "x", "shape": [4, 8]}, {"name": "w", "shape": [16, 8]}],
                      "dims": {"X_D0": 4, "X_D1": 8, "W_D0": 16, "W_D1": 8}}}  # fmt: skip
    entry = {"route": "kernel", "name": "k", "repo": "repo", "dsl": "sycl",
             "gemm": {"activation": "x", "weight": "w", "n_axis": 0},
             "cost": {"bytes": "M*K", "flops": "2*M*N*K"},
             "kernels": {"a": {"route": "aten", "cost": {"bytes": "X_D0*X_D1"}},
                         "b": {"route": "aten", "cost": {"bytes": "N"}}}}  # fmt: skip
    got, problems = naming.keep_valid({"ops": {"op": entry}, "model": {}}, listing, tmp_path)
    e = got["ops"]["op"]
    assert e["gemm"] and "cost" not in e and "cost" not in e["kernels"]["b"]
    assert e["kernels"]["a"]["cost"] == {"bytes": "X_D0*X_D1"}  # a valid one is kept
    assert len(problems) == 3 and all("that cost dropped" in p for p in problems)
    bad = dict(entry, route="nowhere")  # any other problem still drops the entry
    got, _ = naming.keep_valid({"ops": {"op": bad}, "model": {}}, listing, tmp_path)
    assert got["ops"] == {}


def test_a_bad_config_arg_loses_the_config_args_not_the_attentions_kv_args(tmp_path):
    """A config_arg by an unknown name must not drop the attention op and its kv_args."""
    from xe_forge.frontend import naming

    (tmp_path / "repo").mkdir()
    listing = {"op": {"args": [{"name": "arg0", "shape": [8, 128]},
                               {"name": "arg1", "shape": [64, 4, 128]},
                               {"name": "arg19", "value": "-1"}],
                      "dims": {"ARG0_D0": 8, "ARG0_D1": 128, "ARG1_D0": 64, "ARG1_D1": 4,
                               "ARG1_D2": 128, "ARG19": -1}}}  # fmt: skip
    entry = {"route": "kernel", "name": "fmha", "repo": "repo", "dsl": "sycl",
             "kv_args": ["arg1"], "config_args": ["arg19"]}  # fmt: skip
    # An argument's name is accepted as its dim's (ARG19), as kv_args and token_arg name them.
    assert naming.validate({"ops": {"op": entry}}, listing, tmp_path) == []
    got, problems = naming.keep_valid(
        {"ops": {"op": dict(entry, config_args=["window"])}, "model": {}}, listing, tmp_path
    )
    assert got["ops"]["op"]["kv_args"] == ["arg1"] and "config_args" not in got["ops"]["op"]
    assert len(problems) == 1 and problems[0].endswith("that config_args dropped")
    for field, bad in (("kv_args", [["arg1"]]), ("token_arg", ["arg0"])):  # not a crash
        got, _ = naming.keep_valid({"ops": {"op": dict(entry, **{field: bad})}, "model": {}},
                                   listing, tmp_path)  # fmt: skip
        assert field not in got["ops"]["op"] and got["ops"]["op"]["name"] == "fmha"


def test_a_kernel_a_library_launches_is_also_tuned_as_its_call(tmp_path):
    from xe_forge.frontend import analyze, naming
    from xe_forge.frontend.ir import CaptureRun, Naming, Workload

    (tmp_path / "lib").mkdir()
    (tmp_path / "app").mkdir()
    listing = {"op": {"args": [{"name": "q", "shape": [1, 16, 160, 72]}], "dims": {}}}
    call = {"name": "attend", "repo": "app", "dsl": "sycl", "via": "model.py:211"}
    entry = {"route": "kernel", "name": "micro_sdpa", "repo": "lib", "dsl": "opencl",
             "call": call}  # fmt: skip
    assert naming.validate({"ops": {"op": entry}}, listing, tmp_path) == []
    got, problems = naming.keep_valid(
        {"ops": {"op": dict(entry, call={**call, "repo": "nowhere"})}, "model": {}}, listing,
        tmp_path)  # fmt: skip
    assert "call" not in got["ops"]["op"] and problems[0].endswith("that call dropped")
    assert naming.validate({"ops": {"op": dict(entry, route="aten")}}, listing, tmp_path)

    w = Workload(workload_id="attn/a", family="attention", dims={"S": 160}, dtypes=["bfloat16"],
                 layout=[], attrs={}, var_axes={}, calls=10, device_us=100.0, share=0.5, ops=[],
                 naming=Naming("kernel", "micro_sdpa", "lib", "opencl", call=call))  # fmt: skip
    cap = CaptureRun(run={"total_device_us": 200.0}, ops=[], workloads=[w])
    out = analyze.analyze(cap, top=3)
    by = {e.name: e for e in out.entries}
    assert by["micro_sdpa"].target == "kernel" and by["micro_sdpa"].repo == "lib"
    assert by["attend"].target == "library" and by["attend"].repo == "app"
    assert by["attend"].workloads[0].workload_id == "attn/a" and w.naming.route == "kernel"


def test_naming_family_is_free_text_and_a_bad_gemm_spec_degrades(tmp_path):
    from xe_forge.frontend import canonical, naming
    from xe_forge.frontend.ir import CapturedOp, TensorArg

    (tmp_path / "repo").mkdir()
    listing = {"op": {"args": [{"name": "x", "shape": [4, 8]}, {"name": "w", "shape": [16, 8]}],
                      "dims": {}}}  # fmt: skip
    entry = {"route": "kernel", "name": "k", "repo": "repo", "dsl": "sycl", "family": "ssm_scan",
             "gemm": {"activation": "x", "weight": "w", "n_axis": "zero", "grouped": "yes"}}  # fmt: skip
    assert naming.validate({"ops": {"op": entry}}, listing, tmp_path) == []
    bad = dict(entry, family="Not A Word")
    assert any(
        "lowercase word" in p for p in naming.validate({"ops": {"op": bad}}, listing, tmp_path)
    )
    op = CapturedOp(id="op-0", framework_op="op", args=[TensorArg([4, 8], "float16"),
                    TensorArg([16, 8], "float16")], framework_meta={"arg_names": ["x", "w"]})  # fmt: skip
    got = canonical._one_gemm(op, entry["gemm"])  # malformed n_axis/grouped: no N, no crash
    assert got == {"M": 4, "K": 8, "grouped": False}


SPLIT_LAUNCHER = """#!/bin/bash
cat > "$RAW_DIR/meta.json" <<'J'
{"framework": "toyfw", "model": {"name": "toy"}, "runtime": {"batch": 1, "precision": "float16", "quantization": "float16"}, "device": {}}
J
cat > "$RAW_DIR/ops.json" <<'J'
{"rms": {"framework_op": "toy::RMS", "kernels": ["^rms_opt"]},
 "fc_q": {"framework_op": "toy::FC", "kernels": ["^gemm_kernel$"], "launches": [[0, 1]],
          "args": [{"name": "x", "shape": [4, 896], "dtype": "float16"}]},
 "fc_o": {"framework_op": "toy::FC", "kernels": ["^gemm_kernel$"], "launches": [%s],
          "args": [{"name": "x", "shape": [4, 128], "dtype": "float16"}]}}
J
"""


def test_timeline_follows_submission_not_skewed_start_times(tmp_path):
    from xe_forge.skills.capture import _timeline

    def ev(name, ts, i=None):
        return {"ph": "X", "cat": "gpu_op", "name": name, "ts": ts, "dur": 1.0,
                **({"args": {"id": str(i)}} if i is not None else {})}  # fmt: skip

    # a oneDNN gemm reported as starting before the kernel submitted just ahead of it
    (tmp_path / "python.1.json").write_text(json.dumps({"traceEvents": [
        ev("norm", 0.0, 1), ev("convert", 20.5, 2), ev("gemm_kernel", 4.3, 3)]}))  # fmt: skip
    (tmp_path / "python.2.json").write_text(json.dumps({"traceEvents": [
        ev("later", 300.0), ev("first_of_2", 100.0)]}))  # fmt: skip
    assert [n for n, _ in _timeline(tmp_path)] == [
        "norm", "convert", "gemm_kernel", "first_of_2", "later"]  # fmt: skip


@pytest.mark.parametrize("w0, w1, bad", [
    (["in0", "weight"], ["in0", "weight"], False),
    (["in0", "const:layers.0.w"], ["in0", "const:layers.1.w"], True),
    (["in0", "weight"], ["in0", "weight", "bias"], False),  # another arity: another signature
])  # fmt: skip
def test_ops_json_nodes_of_one_type_name_their_arguments_by_role(w0, w1, bad):
    from xe_forge.skills.capture import _ops_summary

    def fc(names):
        return {
            "framework_op": "FC",
            "kernels": ["^gemm$"],
            "args": [{"name": n, "shape": [16, 8], "dtype": "float16"} for n in names],
        }

    ops = {"fc0": fc(w0), "fc1": fc(w1), "fc2": fc(w1)}
    _, problems = _ops_summary(ops, [["gemm", 1.0, 1]], False)
    hits = [p for p in problems if "name their arguments differently" in p]
    assert len(hits) == bad, problems  # one per type, not one per node


@pytest.mark.parametrize(
    "fc_o, problem",
    [("[3, 4]", None), ("[0, 1]", "claimed by both"), ("[1, 2]", "matches none of its kernel")],
)
def test_capture_parse_splits_a_shared_kernel_by_launch(tmp_path, fc_o, problem):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", SPLIT_LAUNCHER % fc_o)
    raw, cap = tmp_path / "raw", tmp_path / "capture.json"
    code, out = _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))
    assert code == 0, out
    assert len(json.loads((raw / "unitrace_launches.json").read_text())["launches"]) == 6
    code, out = _tool("capture", "parse", str(raw), "-o", str(cap))
    if problem:
        assert code == 1 and problem in out, out
        return
    assert code == 0, out
    ops = json.loads(cap.read_text())["ops"]
    fcs = sorted((o["args"][0]["shape"][1], o["device_us"], o["calls"]) for o in ops
                 if o["framework_op"] == "toy::FC")  # fmt: skip
    assert fcs == [(128, 6.0, 1), (896, 6.0, 1)]  # one gemm_kernel launch each


def test_counters_bound_only_a_kernel_of_one_workload_that_they_sampled():
    calib = {"bandwidth_gbs": 100.0}
    w1, w2 = _workload({"gemm_kernel": 10.0}), _workload({"gemm_kernel": 10.0})
    w1.workload_id, w2.workload_id = "gemm/a", "gemm/b"
    good = {"gb_per_s": 50.0, "util_pct": 10.0, "us": 20.0}
    # One library kernel for two matmuls: its averaged counters bound neither.
    ctx = {"counters": {"gemm_kernel": good}, "shared": {"gemm_kernel"}}
    assert analyze._sol_parts(w1, calib, ctx) is None
    # Its own kernel, but sampled for 2 of its 10 us: a few launches, not the kernel.
    ctx = {"counters": {"gemm_kernel": dict(good, us=2.0)}, "shared": set()}
    assert analyze._sol_parts(w1, calib, ctx) is None
    ctx = {"counters": {"gemm_kernel": good}, "shared": set()}
    assert analyze._sol_parts(w1, calib, ctx)[0] == pytest.approx(0.5)


# ── a framework patch: the agent writes it, `patch check` and `run` gate it ─


def _ref_stub(path, bf16, fp32=None, argv=None):
    """The GPU reference: per-token gaps for the rows asked, bf16's or float32's (None: the
    float32 pass fails; a row of lists: its gaps per draw); each call's argv appended to
    ``argv``."""
    table = json.dumps({"bfloat16": [list(g) for g in bf16],
                        "float32": [list(g) for g in fp32] if fp32 else None})  # fmt: skip
    helper = path.with_suffix(".py")
    helper.write_text(f"""import json, sys
table = json.loads({table!r})
dtype, rows = sys.argv[9], sys.argv[10]
dtype = dtype if dtype == "float32" else "bfloat16"  # the first pass, any precision
if {str(argv)!r} != "None":
    open({str(argv)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
if table[dtype] is None:
    sys.exit("float32 does not load")
rows = [int(i) for i in rows.split(",")] if rows else range(len(table[dtype]))
runs = lambda g: g if g and isinstance(g[0], list) else [g]
print(json.dumps({{"gaps": {{str(i): runs(table[dtype][i]) for i in rows}}, "device": "cpu",
                  "transformers": ["5", "t"]}}))
""")
    return _exe(path, f'#!/bin/bash\nshift\nexec {sys.executable} {helper} "$@"\n')


def _patch_dir(tmp_path, tokens, base_from="parent", gaps=((0, 0, 0, 0), (0, 0.2, 0, 0))):
    """A clone with base + one patch commit, generate.sh emitting ``tokens``, and stubs for the
    framework interpreter (its sys.prefix) and the GPU reference (fixed per-token gaps)."""
    repos = tmp_path / "repos"
    repo = repos / "fw"
    repo.mkdir(parents=True)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                               *a], check=True, capture_output=True, text=True).stdout.strip()  # fmt: skip

    git("init", "-q")
    (repo / "model.py").write_text("ARCHS = []\n")
    git("add", "."), git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    (repo / "model.py").write_text("ARCHS = ['Toy']\n")
    git("commit", "-q", "-am", "add Toy")
    work = tmp_path / "work"
    work.mkdir()
    (work / "patch.json").write_text(json.dumps({
        "python": sys.executable, "repo": "fw", "package": "json",
        "base": base if base_from == "parent" else "0" * 40}))  # fmt: skip
    (work / "generate.sh").write_text(f"echo '{json.dumps(tokens)}' > \"$OUT/tokens.json\"\n")
    (work / "PR.md").write_text("Add the Toy architecture.\n")
    fw = _exe(tmp_path / "fw-python", "#!/bin/bash\necho /no/such/prefix\n")
    return work, repos, fw, _ref_stub(tmp_path / "ref-python", gaps)


def _check(work, repos, fw, ref):
    return _tool("patch", "check", str(work), "--repos", str(repos), "--framework-python",
                 str(fw), "--reference-python", str(ref), "--model", "toy", "--batch", "2",
                 "--input-len", "4", "--min-match", "4")  # fmt: skip


def test_patch_check_passes_a_patch_that_generates_the_reference(tmp_path):
    work, repos, fw, ref = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    code, out = _check(work, repos, fw, ref)
    assert code == 0, out
    rec = json.loads((work / "check.json").read_text())
    assert rec["pass"] and rec["worst"] == [0, 0.2] and rec["files"] == ["model.py"]
    assert "ARCHS = ['Toy']" in (work / "fw.patch").read_text()  # git's diff, written by the tool
    assert "Subject: [PATCH] add Toy" in (work / "fw.mbox").read_text()  # the commit, as sent
    pr = (work / "PR.md").read_text()
    assert (
        pr.startswith("Add the Toy architecture.")
        and "| worst gap per sequence (nat) | [0, 0.2] |" in pr
    )
    _check(work, repos, fw, ref)
    assert (work / "PR.md").read_text() == pr  # a re-check replaces its numbers, never adds more


@pytest.mark.parametrize("spoil, problem", [
    (lambda work, repo: (work / "PR.md").unlink(), "no PR.md"),
    (lambda work, repo: (repo / "model.py").write_text("ARCHS = ['Toy', 'Uncommitted']\n"),
     "uncommitted changes"),
    (lambda work, repo: subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
         "--amend", "-m", "add Toy\n\nSigned-off-by: someone <s@s>"], check=True),
     "Signed-off-by"),
])  # fmt: skip
def test_patch_check_fails_a_patch_not_ready_for_its_maintainers(tmp_path, spoil, problem):
    work, repos, fw, ref = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    spoil(work, repos / "fw")
    code, out = _check(work, repos, fw, ref)
    assert code == 1 and problem in out, out


def test_patch_check_uses_the_agents_reference_interpreter(tmp_path):
    """patch.json's "reference_python" (the agent's environment) runs the fixed scoring."""
    work, repos, fw, ref = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    meta = json.loads((work / "patch.json").read_text())
    (work / "patch.json").write_text(json.dumps({**meta, "reference_python": str(ref)}))
    code, out = _check(work, repos, fw, tmp_path / "no-such-python")
    assert code == 0, out


@pytest.mark.parametrize("tokens, base, gaps, problem", [
    ([[1, 2, 3, 4], [5, 6, 9, 9]], "parent", [[0, 0, 0, 0], [0, 0, 4.1, 0]], "nat below"),
    ([[1, 2, 3, 4], [5, 6]], "parent", [[0, 0, 0, 0], [0, 0]], "nat below"),  # too short
    ([[1, 2, 3, 4], [5, 6, 7, 8]], "unknown", [[0] * 4] * 2, "not an ancestor"),
])  # fmt: skip
def test_patch_check_fails_a_wrong_patch(tmp_path, tokens, base, gaps, problem):
    work, repos, fw, ref = _patch_dir(tmp_path, tokens, base, gaps)
    code, out = _check(work, repos, fw, ref)
    assert code == 1 and problem in out, out


@pytest.mark.parametrize("fp32, code, worst, judge", [
    ([[0] * 4, [0, 0, 0.0, 0]], 0, [0, 0.0], ["bfloat16", "float32"]),  # bf16 wrong there: fp32 decides
    ([[0] * 4, [0, 0, 1.2, 0]], 1, [0, 1.2], ["bfloat16", "float32"]),  # fp32 agrees: still a FAIL
    ([[0] * 4, [0.5, 0, 0, 0]], 1, [0, 0.5], ["bfloat16", "float32"]),  # fp32 judges every position
    (None, 1, [0, 1.4375], ["bfloat16", "bfloat16"]),  # the float32 pass fails: bf16's verdict stands
])  # fmt: skip
def test_patch_check_judges_a_sequence_bf16_fails_again_in_float32(tmp_path, fp32, code, worst,
                                                                   judge):  # fmt: skip
    work, repos, fw, _ = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    argv = tmp_path / "ref-argv"
    ref = _ref_stub(tmp_path / "ref-python", [[0] * 4, [0, 0, 1.4375, 0]], fp32, argv)
    got, out = _check(work, repos, fw, ref)
    assert got == code, out
    rec = json.loads((work / "check.json").read_text())
    assert rec["worst"] == worst and rec["judge"] == judge
    assert rec["first_gaps"][1] == [0, 0, 1.4375, 0]
    calls = [json.loads(line) for line in argv.read_text().splitlines()]
    assert [(c[8], c[9], c[10]) for c in calls] == [("bfloat16", "", "1"), ("float32", "1", "8")]


@pytest.mark.parametrize("draws, code, problem", [
    # token 2 is 0.5 off on one draw, 1.4 on another: within the reference's own spread
    ([[0] * 7 + [1.4], [0] * 7 + [0.5]], 0, ""),
    # consistent across draws: a wrong token, however noisy the rest
    ([[0] * 7 + [1.2], [0] * 7 + [1.3]], 1, "nat below"),
    # two of eight tokens pass only within the spread: too unstable to judge
    ([[0] * 6 + [1.4, 1.4], [0] * 6 + [0.5, 0.5]], 1, "too unstable"),
])  # fmt: skip
def test_patch_check_excuses_only_what_the_reference_disagrees_with_itself_on(
    tmp_path, draws, code, problem
):
    work, repos, fw, _ = _patch_dir(tmp_path, [list(range(8)), list(range(8))])
    ref = _ref_stub(tmp_path / "ref-python", [[0] * 8, [0] * 7 + [2.0]], [[0] * 8, draws])
    got, out = _tool("patch", "check", str(work), "--repos", str(repos), "--framework-python",
                     str(fw), "--reference-python", str(ref), "--model", "toy", "--batch", "2",
                     "--input-len", "4", "--min-match", "8")  # fmt: skip
    assert got == code and problem in out, out


def test_run_establishes_a_patch_then_captures_on_it(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    answers = iter([{"unsupported": "model", "what": "Toy", "reason": "registry.py:1"},
                    {"notes": "patched"}, {"launcher": "launcher/run.sh"}])  # fmt: skip
    seen = []

    def agent(step, template, ctx, cwd, answer, python, turns):
        seen.append((step, python))
        if step == "patch":
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / "patch.json").write_text('{"python": "/patched/python"}')
        return next(answers)

    monkeypatch.setattr(cli, "_agent", agent)
    checks = []
    monkeypatch.setattr(cli, "_tool", lambda py, *argv, **env: seen.append((argv[0], py))
                        or (argv[0] == "patch" and checks.append(argv)) or 0)  # fmt: skip
    a = SimpleNamespace(model="m", framework="fw", workload="text", batch=1, input_len=4,
                        output_len=2, reference_python="/ref/python")  # fmt: skip
    code, python = cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")
    assert (code, python) == (0, "/patched/python")
    assert seen == [("capture", "/fw/python"), ("patch", "/fw/python"), ("patch", "/fw/python"),
                    ("capture", "/patched/python"), ("capture", "/patched/python")]  # fmt: skip
    assert json.loads((tmp_path / "stack.json").read_text())["python"] == "/patched/python"
    assert checks[0][-2:] == ("--reference-python", "/ref/python")


def test_run_stops_on_a_precision_the_device_has_no_kernel_for(tmp_path, monkeypatch):
    """A checkpoint format only emulated here is reported, never patched."""
    from xe_forge.frontend import cli

    steps = []
    monkeypatch.setattr(cli, "_agent", lambda step, *rest: steps.append(step) or
                        {"unsupported": "precision", "what": "fp4", "reason": "platform.py:1"})  # fmt: skip
    monkeypatch.setattr(cli, "_tool", lambda *a, **k: pytest.fail("nothing runs"))
    a = SimpleNamespace(
        model="m", framework="fw", workload="text", batch=1, input_len=4, output_len=2
    )
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos") == (3, "/fw/python")
    assert steps == ["capture"]


@pytest.mark.parametrize("refusals,recapture,code,captures", [
    (0, 0, 0, []), (1, 0, 0, [["no weight"]]), (2, 0, 1, [["no weight"]]),
    (1, 3, 3, [["no weight"]]), (1, -9, 1, [["no weight"]])])  # fmt: skip
def test_a_capture_the_named_read_refuses_goes_back_once_to_the_capture_agent(
    tmp_path, monkeypatch, refusals, recapture, code, captures
):
    from xe_forge.frontend import cli, ir

    left, seen = [refusals], []
    (tmp_path / "capture.json").write_text("{}")
    (tmp_path / "session.log").write_text("first capture")
    cap = SimpleNamespace(ops=[])
    monkeypatch.setattr(ir, "load", lambda p: cap)
    monkeypatch.setattr(cli, "_stamp", lambda c: "same")  # a fixed launcher: the same stamp

    def name(*args, **kw):
        seen.append("name")
        (tmp_path / "names.json").write_text(json.dumps({"capture": "same", "ops": {}}))

    def tool(py, *argv, problems=None, **env):
        if problems is None or not left[0]:
            return 0
        left[0] -= 1
        problems.append("no weight")
        return 1

    def capture(a, py, run_dir, turns, repos, refused=None):
        seen.append(refused)
        return recapture, "/patched/python"

    monkeypatch.setattr(cli, "name", name)
    monkeypatch.setattr(cli, "_tool", tool)
    monkeypatch.setattr(cli, "capture", capture)
    a = SimpleNamespace(name_coverage=0.99, calibration=None)
    got, python = cli.name_and_read(a, cap, tmp_path, "/fw/python", 5, tmp_path)
    assert got == code and python == ("/patched/python" if captures else "/fw/python")
    assert [s for s in seen if s != "name"] == captures
    # the new capture is named again; a failed one leaves no names.json for the next run
    assert seen.count("name") == 1 + (len(captures) if recapture == 0 else 0)
    assert (tmp_path / "names.json").is_file() == (recapture == 0)
    assert (tmp_path / "session-refused.log").is_file() == bool(captures)


def test_prompt_file_gives_each_row_its_own_slice_and_repeats_too_little_text(tmp_path):
    from xe_forge.frontend import profile

    vocab = profile.Vocab(lambda text: [ord(c) for c in text], range(10))
    f = tmp_path / "prompts.txt"
    f.write_text("abcdefgh")
    rows = profile.token_ids(vocab, 2, 4, f)
    assert rows == [[97, 98, 99, 100], [101, 102, 103, 104]] == profile.token_ids(vocab, 2, 4, f)
    assert profile.token_ids(vocab, 3, 3, f)[2] == [103, 104, 97]  # 8 ids for 3 rows of 3


def test_patch_check_scores_on_the_runs_prompt_file(tmp_path):
    work, repos, fw, ref = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    argv = tmp_path / "ref-argv"
    ref = _ref_stub(tmp_path / "ref-python", [[0] * 4] * 2, argv=argv)
    code, msg = _tool("patch", "check", str(work), "--repos", str(repos), "--framework-python",
                      str(fw), "--reference-python", str(ref), "--model", "toy", "--batch", "2",
                      "--input-len", "4", "--min-match", "4", "--prompt-file", "/run/p.txt")  # fmt: skip
    assert code == 0, msg
    assert json.loads(argv.read_text())[7] == "/run/p.txt"  # the scoring's prompt file
    assert json.loads((work / "check.json").read_text())["prompt_file"] == "/run/p.txt"


@pytest.mark.parametrize("given, scored", [(None, "bfloat16"), ("float16", "float16")])
def test_patch_check_scores_first_in_the_runs_declared_precision(tmp_path, given, scored):
    work, repos, fw, ref = _patch_dir(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]])
    argv = tmp_path / "ref-argv"
    ref = _ref_stub(tmp_path / "ref-python", [[0] * 4] * 2, argv=argv)
    code, msg = _tool("patch", "check", str(work), "--repos", str(repos), "--framework-python",
                      str(fw), "--reference-python", str(ref), "--model", "toy", "--batch", "2",
                      "--input-len", "4", "--min-match", "4",
                      *(["--precision", given] if given else []))  # fmt: skip
    assert code == 0, msg
    assert json.loads(argv.read_text())[8] == scored


@pytest.mark.parametrize("precision, passed", [("auto", None), ("float16", "float16"),
                                               ("int8", None)])  # fmt: skip
def test_run_gives_a_declared_float_precision_to_the_patch_check(
    tmp_path, monkeypatch, precision, passed
):
    from xe_forge.frontend import cli

    answers = iter([{"unsupported": "model", "what": "Toy", "reason": "r"}, {"notes": "patched"},
                    {"launcher": "launcher/run.sh"}])  # fmt: skip

    def agent(step, template, ctx, cwd, answer, python, turns):
        if step == "patch":
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / "patch.json").write_text('{"python": "/patched/python"}')
        return next(answers)

    monkeypatch.setattr(cli, "_agent", agent)
    checks = []
    monkeypatch.setattr(cli, "_tool", lambda py, *argv, **env:
                        (argv[0] == "patch" and checks.append(argv)) or 0)  # fmt: skip
    a = SimpleNamespace(model="m", framework="fw", workload="text", batch=1, input_len=4, output_len=2,
                        precision=precision)  # fmt: skip
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")[0] == 0
    got = checks[0][checks[0].index("--precision") + 1] if "--precision" in checks[0] else None
    assert got == passed


def test_patch_check_gives_generate_its_natural_text_prompt_and_shape(tmp_path):
    """Without a run prompt file the check scores on its own prose, never synthetic ids, and
    tells generate.sh the prompt, shape and the tokens it will score."""
    from xe_forge.skills.patch import PROMPT

    work, repos, fw, _ = _patch_dir(tmp_path, [])
    argv = tmp_path / "ref-argv"
    ref = _ref_stub(tmp_path / "ref-python", [[0] * 4] * 2, argv=argv)
    (work / "generate.sh").write_text(  # echoes back what it was given, as its tokens
        'echo "[[$BATCH, $INPUT_LEN, $OUTPUT_LEN, 0], [0, 0, 0, 0]]" > "$OUT/tokens.json"\n'
        'echo "$PROMPT_FILE" > seen-prompt\n')  # fmt: skip
    code, msg = _check(work, repos, fw, ref)
    assert code == 0, msg
    assert json.loads((work / "check.json").read_text())["tokens"][0] == [2, 4, 4, 0]
    seen = (work / "seen-prompt").read_text().strip()
    assert seen == str(PROMPT) == json.loads(argv.read_text())[7]
    assert len(PROMPT.read_text().split()) > 800  # a 512-token row is prose, not a repeat


def test_capture_parse_refuses_inputs_that_changed_since_the_run(tmp_path):
    from xe_forge.skills.capture import _sha256

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("some text")
    inputs = json.dumps({"prompts": {"file": str(prompts), "sha256": _sha256(prompts)}})
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", (LAUNCHER % "").replace(
        '"runtime": {"batch": 1', f'"runtime": {{"batch": 1, "inputs": {inputs}'))  # fmt: skip
    raw, cap = tmp_path / "raw", tmp_path / "capture.json"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    assert _tool("capture", "parse", str(raw), "-o", str(cap))[0] == 0
    prompts.write_text("other text")
    code, out = _tool("capture", "parse", str(raw), "-o", str(cap))
    assert code == 1 and "input 'prompts'" in out, out


def test_run_gives_the_prompt_file_to_the_patch_check(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    answers = iter([{"unsupported": "model", "what": "Toy", "reason": "r"}, {"notes": "patched"},
                    {"launcher": "launcher/run.sh"}])  # fmt: skip
    ctxs = []

    def agent(step, template, ctx, cwd, answer, python, turns):
        ctxs.append(ctx)
        if step == "patch":
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / "patch.json").write_text('{"python": "/patched/python"}')
        return next(answers)

    monkeypatch.setattr(cli, "_agent", agent)
    checks = []
    monkeypatch.setattr(cli, "_tool", lambda py, *argv, **env:
                        (argv[0] == "patch" and checks.append(argv)) or 0)  # fmt: skip
    inputs = {"prompts": {"file": "/run/inputs/prompts.txt", "sha256": "ab"}}
    a = SimpleNamespace(model="m", framework="fw", workload="text", batch=1, input_len=4, output_len=2,
                        inputs=inputs)  # fmt: skip
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")[0] == 0
    assert all(c["inputs"] == inputs for c in ctxs)
    assert checks[0][-2:] == ("--prompt-file", "/run/inputs/prompts.txt")


def test_naming_listing_merges_identical_shapes_so_rare_ones_show():
    """Many identical decode calls must not crowd a prefill shape out of the agent's view."""
    from xe_forge.frontend import cli

    def op(rows, us):
        return SimpleNamespace(calls=1, device_us=us, kernel_symbols={"k": us}, framework_meta={},
                               args=[SimpleNamespace(shape=[rows, 896], dtype="float32")],
                               scalars=[], arg_names=["x"])  # fmt: skip

    shapes = cli._shapes([op(4, 10.0) for _ in range(20)] + [op(2048, 19.0)])
    assert [(s["calls"], s["us_per_call"]) for s in shapes] == [(20, 10.0), (1, 19.0)]


def test_capture_parse_refuses_a_dtype_name_the_bounds_cannot_size(tmp_path):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(
        tmp_path / "run.sh", (LAUNCHER % "").replace('"dtype": "float16"', '"dtype": "f16"', 1)
    )
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 1 and "dtype 'f16'" in out, out


def test_precision_and_weights_parse_to_torch_dtypes():
    from xe_forge.frontend.dtypes import canonical, compute_dtype, parse_quantization

    assert canonical("bf16") == "bfloat16" and canonical("float8_e4m3fn") == "float8_e4m3fn"
    with pytest.raises(ValueError):
        canonical("f16")
    assert parse_quantization("auto") == (None, None)
    assert parse_quantization("int8") == (None, "int8")
    assert parse_quantization("awq:int4") == ("awq", "int4")
    assert parse_quantization("Q4_K_M") == (
        "Q4_K_M",
        None,
    )  # a scheme alone is recorded, not checked
    with pytest.raises(ValueError):
        parse_quantization("awq:i4")
    assert compute_dtype(["int8", "int32"], "float32") == "float32"


def test_a_template_variable_the_caller_did_not_pass_is_an_error():
    """A template variable the caller did not pass raises instead of rendering blank."""
    from jinja2 import UndefinedError

    from xe_forge.claude.generator import render

    assert "captured.json" in render("frontend-resume.md.j2", answer="captured.json")
    with pytest.raises(UndefinedError):
        render("frontend-resume.md.j2")


def test_an_fp8_model_computes_in_bf16_and_its_gemms_in_fp8():
    """Block-scaled fp8 GEMMs (fp8 operands, fp32 scales) count as fp8, not as their scales;
    the model's compute dtype is still the bf16 the rest runs in."""
    from xe_forge.frontend.dtypes import compute_dtype, observed

    def op(us, *dtypes):
        return SimpleNamespace(device_us=us, args=[None if d is None else
                                                   SimpleNamespace(dtype=d) for d in dtypes])  # fmt: skip

    shares = observed([op(76.0, "float8_e4m3fn", "float8_e4m3fn", None, "float32", "float32"),
                       op(24.0, "bfloat16", "float32")])  # fmt: skip
    assert shares == {"float8_e4m3fn": 0.76, "bfloat16": 0.24}
    assert compute_dtype(list(shares)) == "bfloat16"
    # int8 W8A8 GEMMs (int8 operands, fp32 scales) count as int8; an index does not count
    shares = observed([op(90.0, "int8", "int8", "float32", "float32"),
                       op(10.0, "int64", "bfloat16")])  # fmt: skip
    assert shares == {"int8": 0.9, "bfloat16": 0.1}
    assert compute_dtype(list(shares)) == "bfloat16"


@pytest.mark.parametrize("declared, code", [("auto", 1), ("fp16", 0), ("bf16", 1)])
def test_capture_parse_checks_the_declared_precision(tmp_path, declared, code):
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", (LAUNCHER % "").replace(
        '"precision": "float16"', f'"precision": "{declared}"'))  # fmt: skip
    raw, cap = tmp_path / "raw", tmp_path / "capture.json"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    got, out = _tool("capture", "parse", str(raw), "-o", str(cap))
    assert got == code, out
    assert json.loads(cap.read_text())["run"]["precision"]["compute"] == "float16"
    if declared == "bf16":
        assert "declared compute bfloat16, the trace ran 100% in float16" in out
    if declared == "auto":  # the launcher writes the dtype that ran, not the request
        assert "runtime.precision" in out


@pytest.mark.parametrize("weights, w_bytes, ok", [
    ("int8", 1.06, True), ("gptq:int4", 1.06, False),
    ("Q4_K_M", 1.06, False), ("auto", 1.06, False),  # no stored dtype named: nothing to check
    ("fp8:float8_e4m3fn", 2.0, False),  # more bytes than stored, and no op named as widening
    ("int8", 0.7, False),  # fewer bytes than the dtype stores
    ("k-quant rows packed as bytes:uint8", 0.7, True),  # uint8 also holds packed sub-byte values
    ("k-quant rows packed as bytes:uint8", 1.5, False),
    ("int8", None, True),  # before naming no GEMM's weight is known yet
])  # fmt: skip
def test_declared_weights_are_checked_against_bytes_per_weight_element(weights, w_bytes, ok):
    from xe_forge.skills.capture import _precision_problems

    run = {"runtime": {"precision": "bf16", "quantization": weights},
           "precision": {"compute": "bfloat16", "dtypes": {"bfloat16": 1.0}, "w_bytes": w_bytes}}  # fmt: skip
    assert (not _precision_problems(run)) is ok


@pytest.mark.parametrize("claim, w_bytes, ok", [
    (None, 2.0, False),  # GEMMs read bf16 and nobody says why: loaded wide
    ("aten::to", 2.0, True),  # the namer names the cast, which does read the stored fp8
    ("aten::linear", 2.0, False),  # an op that reads no fp8 is not what widened them
    ("aten::nope", 2.0, False),
    (None, 1.0, True),
    ("aten::to", 1.0, True),  # a claim the bytes do not need is still checked
    ("aten::linear", 1.0, False),
])  # fmt: skip
def test_weights_widened_on_the_fly_are_the_namers_claim_the_tool_verifies(claim, w_bytes, ok):
    from xe_forge.frontend.ir import CapturedOp, TensorArg
    from xe_forge.skills.capture import _precision_problems

    def op(name, dtype):
        return CapturedOp(id=name, framework_op=name, args=[TensorArg([16, 8], dtype)],
                          framework_meta={"name_key": name})  # fmt: skip

    run = {"runtime": {"precision": "bf16", "quantization": "fp8:float8_e4m3fn"},
           "model_facts": {"weights_widened_by": claim},
           "precision": {"compute": "bfloat16", "dtypes": {"bfloat16": 1.0}, "w_bytes": w_bytes}}  # fmt: skip
    ops = [op("aten::to", "float8_e4m3fn"), op("aten::linear", "bfloat16")]
    assert (not _precision_problems(run, named=True, ops=ops)) is ok


@pytest.mark.parametrize("weights, ok", [("int8", False), ("Q4_K_M", False), (None, False)])
def test_named_capture_with_declared_weights_no_gemm_read_is_a_problem(weights, ok):
    from xe_forge.skills.capture import _precision_problems

    run = {"runtime": {"precision": "bfloat16", "quantization": weights},
           "precision": {"compute": "bfloat16", "dtypes": {"bfloat16": 1.0}, "w_bytes": None}}  # fmt: skip
    assert (not _precision_problems(run, named=True)) is ok


def test_naming_checks_a_cost_on_shapes_beyond_the_ones_listed():
    from xe_forge.frontend import cli, naming

    def op(rows, us):
        return SimpleNamespace(calls=1, device_us=us, kernel_symbols={"k": us}, framework_meta={},
                               args=[SimpleNamespace(shape=[rows, 896], dtype="float32")],
                               scalars=[], arg_names=["x"])  # fmt: skip

    ops = [op(r, 100.0) for r in range(1, 13)] + [op(4096, 1.0)]  # the 13th, cheapest, refutes
    names = {"ops": {"K": {"cost": {"bytes": "ARG0_D0 * 1000000", "flops": "0"}}}}
    calib = {"bandwidth_gbs": 1000.0, "matmul_tflops": {"float32": 1.0}}
    assert not naming.refuted(names, {"K": {"shapes": cli._shapes(ops)}}, calib, 0.05)
    assert naming.refuted(names, {"K": {"shapes": cli._shapes(ops, top=None)}}, calib, 0.05)


def test_summary_states_the_declared_precision_and_what_ran(tmp_path):
    from xe_forge.frontend import collect

    run = {"runtime": {"precision": "bfloat16", "quantization": "fp8:float8_e4m3fn"},
           "precision": {"compute": "bfloat16", "w_bytes": 2.0}}  # fmt: skip
    (tmp_path / "capture.json").write_text(json.dumps({"run": run}))
    text = collect.collect(tmp_path, {"kernels": []}, []).read_text()
    assert ("Precision: compute bfloat16 (bfloat16 ran); quantization fp8:float8_e4m3fn (2.0 B per "
            "GEMM weight element read).") in text.splitlines()[2]  # fmt: skip
    (tmp_path / "capture.json").unlink()  # the fusion step's run dir has none
    assert "Precision" not in collect.collect(tmp_path, {"kernels": []}, []).read_text()


@pytest.mark.parametrize("speedup, mergeable", [(1.12, True), (None, False)])
def test_collect_offers_only_a_win_for_merge(tmp_path, speedup, mergeable):
    """A trial at parity changes nothing worth merging, even when its port passes."""
    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.frontend import collect

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "base.py").write_text("x = 1\n")
    (ws / "t0.py").write_text("x = 2\n")
    mgr = TrialManager(ws / "trials")
    mgr.init("k", ws / "base.py")
    tid = mgr.save_trial("k", ws / "t0.py", strategy="try")
    mgr.record_result("k", tid, correctness="pass", speedup=speedup, baseline_us=10.0,
                      triton_us=10.0, verdict=None if speedup else "INDISTINGUISHABLE",
                      source="measured")  # fmt: skip
    (ws / "output").mkdir()
    (ws / "output" / "PORT.txt").write_text("kind: upstream\nstack check python: PASS\n")
    row = collect.collect_one(ws, "k", "v", tmp_path / "results")
    assert row["outcome"] == ("win" if speedup else "parity")
    assert row["mergeable"] is mergeable


def test_a_port_failing_its_stack_check_leaves_nothing_submit_ready(tmp_path):
    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.frontend import collect

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "base.py").write_text("x = 1\n")
    (ws / "t0.py").write_text("x = 2\n")
    mgr = TrialManager(ws / "trials")
    mgr.init("k", ws / "base.py")
    tid = mgr.save_trial("k", ws / "t0.py", strategy="try")
    mgr.record_result("k", tid, correctness="pass", speedup=1.5, baseline_us=15.0,
                      triton_us=10.0, source="measured")  # fmt: skip
    (ws / "output").mkdir()
    (ws / "output" / "PORT.txt").write_text("kind: upstream\nstack check python: FAIL tests\n")
    for name in ("upstream.patch", "upstream.mbox", "PR.md"):
        (ws / "output" / name).write_text("x\n")
    row = collect.collect_one(ws, "k", "v", tmp_path / "results")
    out = Path(row["result"]) / "output"
    assert row["patches"] == [] and row["pr"] is None
    assert sorted(p.name for p in out.glob("*.rejected")) == [
        "PR.md.rejected", "upstream.mbox.rejected", "upstream.patch.rejected"]  # fmt: skip


@pytest.mark.parametrize("code_b, says", [(3, "does not parse"), (1, "does not run")])
def test_prove_tells_a_refused_parse_from_a_model_that_does_not_run(tmp_path, monkeypatch,
                                                                   code_b, says):  # fmt: skip
    """Arm B running with a capture the launcher cannot attribute is not a broken patch."""
    from xe_forge.frontend import agent, prove

    res = tmp_path / "results" / "k__v"
    (res / "output").mkdir(parents=True)
    (res / "output" / "upstream.patch").write_text("diff\n")
    (tmp_path / "results" / "summary.json").write_text(json.dumps([
        {"kernel": "k", "outcome": "win", "speedup": 1.5, "result": str(res),
         "patches": ["output/upstream.patch"]}]))  # fmt: skip
    (tmp_path / "launcher").mkdir()
    (tmp_path / "launcher" / "run.sh").write_text("#!/bin/bash\n")
    seen = {}

    def build(template, ctx, cwd, answer, python, turns):
        seen.update(ctx)
        return {"A": sys.executable, "B": str(_exe(tmp_path / "b-python", "#!/bin/bash\n"))}

    monkeypatch.setattr(agent, "run", build)

    def recapture(run_dir, out, python):
        if python != sys.executable:
            return code_b
        out.mkdir(parents=True)
        (out / "capture.json").write_text("{}")
        return 0

    got = prove.prove(tmp_path, recapture, sys.executable, 1, 5)
    assert says in got, got
    assert seen["launcher"] == tmp_path / "launcher" / "run.sh"  # the build agent holds it


def test_a_patch_that_breaks_the_model_is_found_and_the_rest_still_proved(tmp_path, monkeypatch):
    from xe_forge.frontend import agent, compare, prove

    rows, table = [], ["# Results", "", "| kernel | repo | variant | decision | outcome | "
                       "speedup | trial | kind | stack | mergeable | patches |", "|---|"]  # fmt: skip
    for k in ("good", "bad"):
        res = tmp_path / "results" / f"{k}__v"
        (res / "output").mkdir(parents=True)
        (res / "output" / "upstream.patch").write_text("diff\n")
        rows.append({"kernel": k, "variant": "v", "outcome": "win", "speedup": 1.5,
                     "result": str(res), "patches": ["output/upstream.patch"]})  # fmt: skip
        table.append(f"| `{k}` | r | v | optimize | win | 1.500x | t1 | upstream | - | yes | "
                     "output/upstream.patch |")  # fmt: skip
    (tmp_path / "results" / "summary.json").write_text(json.dumps(rows))
    (tmp_path / "results" / "SUMMARY.md").write_text("\n".join(table) + "\n")
    (tmp_path / "launcher").mkdir()
    (tmp_path / "launcher" / "run.sh").write_text("#!/bin/bash\n")
    built = []

    def build(template, ctx, cwd, answer, python, turns):
        kinds = sorted(Path(p).parts[-3].split("__")[0] for p in ctx["patches"])
        built.append(kinds)
        b = _exe(cwd / ("b-" + "-".join(kinds)), "#!/bin/bash\n")
        return {"A": sys.executable, "B": str(b)}

    def recapture(run_dir, out, python):
        if "bad" in Path(python).name:
            return 1
        out.mkdir(parents=True)
        (out / "capture.json").write_text("{}")
        return 0

    monkeypatch.setattr(agent, "run", build)
    t = {"speedup": 1.2, "spread_a": 0.01, "spread_b": 0.01, "us_a": 120, "us_b": 100}
    monkeypatch.setattr(compare, "compare", lambda a, b: [t])
    monkeypatch.setattr(compare, "write", lambda table, d: d / "compare.md")
    got = prove.prove(tmp_path, recapture, sys.executable, 1, 5)
    assert built == [["bad", "good"], ["good"], ["bad"]]  # together, then each alone
    assert "each was proved alone" in got and "bad__v" in got and "x1.2" in got
    after = {
        r["kernel"]: r for r in json.loads((tmp_path / "results" / "summary.json").read_text())
    }
    assert "not proven in the model" in after["bad"]["outcome"] and not after["bad"]["mergeable"]
    assert after["good"]["outcome"] == "win"
    md = (tmp_path / "results" / "SUMMARY.md").read_text()
    assert "| `bad` | r | v | optimize | win (not proven in the model" in md and "| no |" in md
    assert "| `good` | r | v | optimize | win | 1.500x" in md


# ── fuse ────────────────────────────────────────────────────────────────────


def _fusion_capture():
    """A norm (op-0, named rms) and a mul (op-1, aten) that follows it, at 4 and 512 tokens."""
    from xe_forge.frontend.ir import CapturedOp, CaptureRun, Naming, Workload

    def w(wid, op, name, us):
        return Workload(workload_id=wid, family="norm", dims={"T": "var", "H": 64},
                        dtypes=["bfloat16"], layout=[], attrs={}, calls=2, device_us=us,
                        share=us / 400, ops=[op], naming=Naming("kernel", name, "r", "sycl"),
                        var_axes={"T": {"512": {"calls": 1, "device_us": us - 10},
                                        "4": {"calls": 1, "device_us": 10}}})  # fmt: skip

    ops = [
        CapturedOp("op-0", "rms", [], device_us=200.0),
        CapturedOp("op-1", "mul", [], device_us=100.0),
    ]
    ws = [w("norm/a", "op-0", "rms", 200.0), w("mul/b", "op-1", "mul", 100.0)]
    return CaptureRun(run={"model": {"name": "m"}, "total_device_us": 400.0}, ops=ops, workloads=ws)


_CHAIN = '''"""rms (norm.cpp:10) then mul (model.py:20)."""
import torch
class Model(torch.nn.Module):
    def forward(self, x, w):
        return x * w
def get_init_inputs(T, H):
    return []
def get_inputs(T, H, dtype):
    return [torch.ones(T, H, dtype=dtype), torch.ones(H, dtype=dtype)]
'''


def _fusion(tmp_path, **over):
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "refs").mkdir(exist_ok=True)
    (tmp_path / "refs" / "rms_mul_pytorch.py").write_text(over.pop("ref", _CHAIN))
    f = {"name": "rms_mul", "repo": "repo", "dsl": "sycl", "ops": ["op-0", "op-1"],
         "anchor": "op-0", "reference": "refs/rms_mul_pytorch.py", "why": "w"}  # fmt: skip
    return {"fusions": [{**f, **over}]}


def test_fuse_check_ranks_a_chain_by_its_time_after_the_kernels_wins(tmp_path):
    from xe_forge.skills import fuse

    problems, passed = fuse.check(_fusion(tmp_path), _fusion_capture(), {"rms": 4.0}, tmp_path)
    assert problems == []
    assert passed[0]["chain_us"] == 150.0  # 200 / 4 for the optimized norm + 100
    assert passed[0]["share"] == 0.375


def test_fuse_check_over_several_rows_wants_the_chain_in_every_row(tmp_path):
    """A chain named in one row of a merged capture would rank on that row's time alone."""
    import copy

    from xe_forge.skills import fuse

    cap = _fusion_capture()
    for o in cap.ops:
        o.framework_meta["row"] = 1
    for o in copy.deepcopy(cap.ops):  # row 2: the same ops, their own ids
        o.id, o.framework_meta["row"] = f"r02:{o.id}", 2
        cap.ops.append(o)
        next(w for w in cap.workloads if w.ops[0] == o.id[4:]).ops.append(o.id)
    problems, _ = fuse.check(_fusion(tmp_path), cap, {}, tmp_path)
    assert any("rows [1] only" in p for p in problems), problems
    every = _fusion(tmp_path, ops=["op-0", "op-1", "r02:op-0", "r02:op-1"])
    problems, passed = fuse.check(every, cap, {}, tmp_path)
    assert problems == [] and passed[0]["chain_us"] == 600.0


@pytest.mark.parametrize("over, says", [
    ({"ops": ["op-0", "op-9"]}, "op-9 is not a captured op"),
    ({"ops": ["op-0"]}, "two ops or more"),
    ({"anchor": "op-7"}, "the anchor is one of its ops"),
    ({"repo": "nowhere"}, "repository 'nowhere' not found"),
    ({"ref": _CHAIN.replace("def get_inputs(T, H, dtype)", "def get_inputs(T, dtype)").replace(
        "def get_init_inputs(T, H)", "def get_init_inputs(T)")}, "the anchor's dims"),
    ({"ref": _CHAIN.replace("def get_inputs(T, H, dtype)", "def get_inputs(T, H)")}, "the dtype"),
    ({"ref": "import torch\n"}, "defines no Model"),
    ({"ref": _CHAIN.replace("def get_inputs(T, H, dtype)", "def get_inputs(**kwargs)")},
     "the anchor's dims"),
    ({"ops": ["op-0", "op-0"]}, "listed twice"),
])  # fmt: skip
def test_fuse_check_refuses_a_fusion_the_pipeline_cannot_build(tmp_path, over, says):
    from xe_forge.skills import fuse

    problems, passed = fuse.check(_fusion(tmp_path, **over), _fusion_capture(), {}, tmp_path)
    assert passed == [] and any(says in p for p in problems), problems


@pytest.mark.parametrize("answer, ok", [({"fusions": [], "why": "no chain worth it"}, True),
                                        ({"fusions": [], "why": " "}, False),
                                        ({"fusions": None, "why": "x"}, False),
                                        ({"why": "x"}, False)])  # fmt: skip
def test_fuse_check_passes_no_fusion_proposed_with_a_reason(tmp_path, answer, ok):
    from xe_forge.frontend import ir
    from xe_forge.skills import fuse

    problems, passed = fuse.check(answer, _fusion_capture(), {}, tmp_path)
    assert passed == [] and (problems == []) is ok, problems
    (tmp_path / "fusions.json").write_text(json.dumps(answer))
    ir.dump(_fusion_capture(), tmp_path / "capture.json")
    args = SimpleNamespace(fusions=str(tmp_path / "fusions.json"), results=None,
                           capture=str(tmp_path / "capture.json"))  # fmt: skip
    assert fuse.run(args) == (0 if ok else 1)


def _git_repo(path: Path) -> str:
    def git(*a):
        return subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True,
                              text=True).stdout.strip()  # fmt: skip

    path.mkdir()
    git("init", "-q")
    (path / "a.txt").write_text("a\n")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return git("rev-parse", "HEAD")


def _winning_patch(results: Path, name: str, repo: Path, base: str, old: str, new: str) -> Path:
    out = results / name / "output"
    out.mkdir(parents=True)
    patch = out / "upstream.patch"
    patch.write_text(f"--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-{old}\n+{new}\n")
    (out / "PORT.txt").write_text(f"kind: upstream\noutput/upstream.patch:\n  repository: {repo}\n"
                                  f"  base commit: {base}\n")  # fmt: skip
    return patch


def test_fusions_build_on_the_kernels_winning_patches_applied_in_order(tmp_path):
    from xe_forge.frontend import fuse

    base = _git_repo(tmp_path / "origin")
    p1 = _winning_patch(tmp_path / "results", "k1", tmp_path / "origin", base, "a", "b")
    p2 = _winning_patch(tmp_path / "results", "k2", tmp_path / "origin", base, "b", "c")
    stacked = fuse.stack([p1, p2], tmp_path / "stacked")
    clone = Path(stacked[str(tmp_path / "origin")])
    assert (clone / "a.txt").read_text() == "c\n"
    made = json.loads((tmp_path / "stacked" / "STACK.json").read_text())
    assert made[0]["base"] == base and made[0]["patches"] == [str(p1), str(p2)]
    assert (tmp_path / "origin" / "a.txt").read_text() == "a\n"  # the origin is untouched


def test_fusions_stack_repositories_with_spaced_paths_and_shared_basenames(tmp_path):
    from xe_forge.frontend import fuse

    one, two = tmp_path / "my repos" / "one" / "src", tmp_path / "my repos" / "two" / "src"
    one.parent.mkdir(parents=True)
    two.parent.mkdir(parents=True)
    b1, b2 = _git_repo(one), _git_repo(two)
    p1 = _winning_patch(tmp_path / "results", "k1", one, b1, "a", "b")
    p2 = _winning_patch(tmp_path / "results", "k2", two, b2, "a", "c")
    assert fuse.origin(p1) == (str(one), b1)
    stacked = fuse.stack([p1, p2], tmp_path / "stacked")
    assert set(stacked) == {str(one), str(two)} and len(set(stacked.values())) == 2
    assert (Path(stacked[str(one)]) / "a.txt").read_text() == "b\n"
    assert (Path(stacked[str(two)]) / "a.txt").read_text() == "c\n"


def test_prove_applies_fusions_after_the_kernels_and_names_their_stack(tmp_path, monkeypatch):
    from xe_forge.frontend import agent, prove

    rows = {}
    for where in ("results", "fuse/results"):
        res = tmp_path / where / "k__v"
        (res / "output").mkdir(parents=True)
        (res / "output" / "upstream.patch").write_text("diff\n")
        rows[where] = res
        (tmp_path / where / "summary.json").write_text(json.dumps([
            {"kernel": "k", "outcome": "win", "speedup": 1.5, "result": str(res),
             "patches": ["output/upstream.patch"]}]))  # fmt: skip
    (tmp_path / "fuse" / "stacked").mkdir()
    (tmp_path / "fuse" / "stacked" / "STACK.json").write_text("[]")
    seen = {}

    def build(template, ctx, cwd, answer, python, turns):
        seen.update(ctx)
        return {"error": "stop here"}

    monkeypatch.setattr(agent, "run", build)
    prove.prove(tmp_path, None, sys.executable, 1, 5)
    assert seen["patches"] == [rows["results"] / "output/upstream.patch",
                               rows["fuse/results"] / "output/upstream.patch"]  # fmt: skip
    assert seen["stack"] == tmp_path / "fuse" / "stacked" / "STACK.json"


def test_a_fusion_workspace_gets_its_chain_reference_and_integration_repos(tmp_path):
    from xe_forge.frontend.cli import xe_forge_command

    entry = {"name": "f", "dsl": "sycl", "spec": "s.yaml", "reference": "/r/f_pytorch.py",
             "integration_repos": ["/i/vllm"]}  # fmt: skip
    opts = SimpleNamespace(engine="claude", device="xpu", max_turns=5, compiler_flags=None,
                           passthrough=[])  # fmt: skip
    cmd = xe_forge_command(entry, "bench", tmp_path, tmp_path / "ws", opts)
    assert cmd[cmd.index("--reference") + 1] == "/r/f_pytorch.py"
    assert cmd[cmd.index("--integration-repo") + 1] == "/i/vllm"
    assert "--unitrace" in cmd  # every workspace profiles; a passed-through --no-unitrace wins


def test_optimize_refuses_without_unitrace_unless_disabled(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    (tmp_path / "kernels.yaml").write_text(json.dumps({"kernels": []}))
    monkeypatch.setenv("UNITRACE_BIN", str(tmp_path / "missing-unitrace"))
    opts = SimpleNamespace(
        kernel_repos=str(tmp_path), engine="dspy-agent", only=None, passthrough=[]
    )
    with pytest.raises(SystemExit, match="profiling needs unitrace"):
        cli.optimize(tmp_path / "kernels.yaml", tmp_path, opts)
    opts.passthrough = ["--no-unitrace"]
    assert cli.optimize(tmp_path / "kernels.yaml", tmp_path, opts) == []


def test_optimize_keeps_a_workspace_that_finished(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    ws = tmp_path / "workspaces" / "k__bench"
    ws.mkdir(parents=True)
    manifest.sidecar(ws, ".status.json").write_text(json.dumps({"exit_code": 0}))
    (tmp_path / "kernels.yaml").write_text(json.dumps({"kernels": [
        {"name": "k", "repo": "r", "dsl": "sycl", "enabled": True, "spec": "k.yaml",
         "variants": ["bench"]}]}))  # fmt: skip
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: pytest.fail("ran it again"))
    opts = SimpleNamespace(kernel_repos=str(tmp_path), engine="dspy-agent", only=None,
                           passthrough=["--no-unitrace"])  # fmt: skip
    assert cli.optimize(tmp_path / "kernels.yaml", tmp_path, opts) == [("k", "bench")]


def test_a_dtype_without_a_measured_peak_has_no_compute_bound_and_is_noted():
    from xe_forge.frontend.calibrate import peak_of, unpeaked_note

    tflops, ctx = {"bfloat16": 150.0, "int8": 280.0}, {}
    assert peak_of(tflops, "bfloat16", ctx) == 150.0 and ctx == {}
    assert peak_of(tflops, "float32", ctx) is None  # never another dtype's peak
    assert peak_of(tflops, None, ctx) is None and ctx == {"unpeaked": {"float32"}}
    (note,) = unpeaked_note(ctx)
    assert note.startswith("float32 has no measured matmul peak")
    assert "compute bound is unknown" in note and "bandwidth alone" in note


def _stub_measure(seen):
    def measure(dtypes):
        seen.append(list(dtypes))
        return {"matmul_tflops": {"float16": 90.0, "float64": None},
                "unmeasured": {"matmul_tflops.float64": "failed"}}  # fmt: skip

    return measure


def test_calibration_measures_only_the_dtypes_it_lacks():
    from xe_forge.frontend import calibrate

    calib = {"bandwidth_gbs": 500.0, "matmul_tflops": {"float32": 10.0, "bfloat16": 100.0},
             "unmeasured": {}}  # fmt: skip
    seen = []
    # float8_e4m3fn is held exactly in bfloat16, which is measured: it has a peak already.
    need = {"float32", "bfloat16", "float8_e4m3fn", "float16", "float64", "uint8"}
    got = calibrate.complete(calib, need, _stub_measure(seen))
    assert seen == [["float16", "float64"]]  # uint8 has no matmul to measure it with
    assert got["matmul_tflops"] == {"float32": 10.0, "bfloat16": 100.0, "float16": 90.0,
                                    "float64": None, "uint8": None}  # fmt: skip
    assert got["unmeasured"] == {"matmul_tflops.float64": "failed",
                                 "matmul_tflops.uint8": "no matmul in this dtype to measure"}  # fmt: skip
    assert calib["matmul_tflops"] == {"float32": 10.0, "bfloat16": 100.0}  # not changed in place
    seen.clear()
    assert calibrate.complete(got, need, _stub_measure(seen)) == got and seen == []


@pytest.mark.parametrize("rec", [
    None,                                                      # the measurement did not finish
    {"matmul_tflops": {}, "unmeasured": {"all": "no GPU"}},    # no device to measure on
    {"matmul_tflops": {"float16": None}, "unmeasured": {}},    # no reason of its own
])  # fmt: skip
def test_a_failure_without_a_reason_of_its_own_is_tried_again(rec):
    from xe_forge.frontend import calibrate

    calib = {"bandwidth_gbs": 500.0, "matmul_tflops": {"float32": 10.0}}
    assert calibrate.complete(calib, {"float16"}, lambda ds: rec) is calib


def test_a_measured_peak_clears_its_old_failure():
    from xe_forge.frontend import calibrate

    calib = {"matmul_tflops": {"float32": 10.0}, "unmeasured": {"matmul_tflops.float16": "x"}}
    got = calibrate.complete(calib, {"float16"},
                             lambda ds: {"matmul_tflops": {"float16": 90.0}})  # fmt: skip
    assert got["matmul_tflops"]["float16"] == 90.0 and got["unmeasured"] == {}


def test_a_peak_measured_on_another_device_is_not_merged(capsys):
    from xe_forge.frontend import calibrate

    calib = {"device": {"type": "xpu", "name": "a", "torch": "1"},
             "matmul_tflops": {"float32": 10.0}}  # fmt: skip
    other = {"device": {"type": "xpu", "name": "b", "torch": "1"},
             "matmul_tflops": {"float16": 90.0}}  # fmt: skip
    assert calibrate.complete(calib, {"float16"}, lambda ds: other) is calib
    assert "not merged" in capsys.readouterr().err
    same = {**other, "device": {**calib["device"], "torch": "2"}}  # another torch, one device
    assert calibrate.complete(calib, {"float16"}, lambda ds: same)["matmul_tflops"]["float16"]


def test_peak_dtypes_are_every_measurable_dtype_the_capture_ran():
    from xe_forge.frontend.ir import CapturedOp, CaptureRun, TensorArg

    args = [TensorArg([4], d) for d in ("float32", "int8", "bfloat16", "int64")]
    cap = CaptureRun(run={}, ops=[CapturedOp(id="o", framework_op="f", args=[*args, None])])
    assert analyze.peak_dtypes(cap) == {"float32", "int8", "bfloat16"}


def _gemm(dtype, us, route="library"):
    from xe_forge.frontend.ir import Naming, Workload

    return Workload(workload_id=f"g/{dtype}", family="gemm", dims={"M": 4, "N": 1000, "K": 1000},
                    dtypes=[dtype], layout=[], attrs={"w_bytes": 2}, var_axes={}, calls=1,
                    device_us=us, share=0.5, ops=[],
                    naming=Naming(route, name=f"mm_{dtype}", repo="r", dsl="sycl"))  # fmt: skip


def _calib_cap(tmp_path):
    from xe_forge.frontend.ir import CapturedOp, CaptureRun, TensorArg

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"bandwidth_gbs": 500.0, "matmul_tflops": {"float32": 10.0}}))
    op = CapturedOp(id="o", framework_op="f", args=[TensorArg([4, 4], "float16")])
    return path, CaptureRun(run={}, ops=[op], workloads=[_gemm("bfloat16", 1.0)])


def test_run_writes_back_the_peaks_the_capture_needs(tmp_path, monkeypatch):
    import xe_forge.frontend.cli as fcli

    path, cap = _calib_cap(tmp_path)
    asked = []

    def call(cmd, env, timeout):
        asked.append(cmd[cmd.index("--dtypes") + 1 :])
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text(json.dumps({"matmul_tflops": dict.fromkeys(asked[-1], 50.0)}))
        return 0

    monkeypatch.setattr(fcli.subprocess, "call", call)
    assert fcli.calibrate_missing(path, cap, tmp_path / "run") == path
    assert asked == [["bfloat16", "float16"]]
    assert json.loads(path.read_text())["matmul_tflops"] == {
        "float32": 10.0, "bfloat16": 50.0, "float16": 50.0}  # fmt: skip
    assert fcli.calibrate_missing(path, cap, tmp_path / "run") == path
    assert len(asked) == 1  # nothing left to measure


@pytest.mark.parametrize("fail", ["timeout", "no output"])
def test_a_calibration_that_did_not_finish_leaves_the_file_alone(tmp_path, monkeypatch, fail):
    import xe_forge.frontend.cli as fcli

    path, cap = _calib_cap(tmp_path)
    before = path.read_text()

    def call(cmd, env, timeout):
        if fail == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        return -9

    monkeypatch.setattr(fcli.subprocess, "call", call)
    assert fcli.calibrate_missing(path, cap, tmp_path) == path
    assert path.read_text() == before


def test_a_calibration_that_cannot_be_written_back_is_kept_in_the_run(tmp_path, monkeypatch):
    import xe_forge.frontend.cli as fcli

    path, cap = _calib_cap(tmp_path)
    before = path.read_text()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def call(cmd, env, timeout):
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text(json.dumps({"matmul_tflops": {"bfloat16": 50.0, "float16": 50.0}}))
        return 0

    write = fcli._write_atomic

    def refuse(target, text):
        if target == path:
            raise PermissionError("read-only")
        write(target, text)

    monkeypatch.setattr(fcli.subprocess, "call", call)
    monkeypatch.setattr(fcli, "_write_atomic", refuse)
    got = fcli.calibrate_missing(path, cap, run_dir)
    assert got == run_dir / "calibration.json" and path.read_text() == before
    assert json.loads(got.read_text())["matmul_tflops"]["bfloat16"] == 50.0
    assert [p.name for p in run_dir.iterdir()] == ["calibration.json"]  # no temp file left


def test_a_cost_with_flops_and_no_dtype_is_noted_unpeaked():
    from xe_forge.frontend.ir import Naming, Workload

    w = Workload(workload_id="c", family="other", dims={}, dtypes=["int32"], layout=[],
                 attrs={"cost": {"bytes": "1000", "flops": "1000"}}, var_axes={}, calls=1,
                 device_us=1.0, share=1.0, ops=[], naming=Naming("kernel"))  # fmt: skip
    ctx = {}
    assert analyze.cost_bound(w, {"bandwidth_gbs": 1.0, "matmul_tflops": {"float32": 1.0}}, ctx)
    assert ctx["unpeaked"] == {analyze.UNKNOWN_DTYPE}


def test_a_cost_is_not_refuted_by_another_dtypes_peak():
    from xe_forge.frontend import naming

    calib = {"bandwidth_gbs": 1000.0, "matmul_tflops": {"float32": 10.0}}
    names = {"ops": {"K": {"cost": {"bytes": "1000000", "flops": "1000000000"}}}}

    def shapes(dtype):
        return {"K": {"shapes": [{"us_per_call": 60.0, "dims": {}, "dtype": dtype}]}}

    # flops / the float32 peak is 100 us > 60 us measured; bfloat16 has no peak of its own
    assert naming.refuted(names, shapes("bfloat16"), calib, 0.05) == []
    assert naming.refuted(names, shapes("float32"), calib, 0.05)


@pytest.mark.parametrize("route", ["library", "kernel"])
def test_an_unpeaked_gemm_is_not_stopped_at_roofline(route):
    from xe_forge.frontend.ir import CaptureRun

    calib = {"bandwidth_gbs": 1000.0, "matmul_tflops": {"float32": 10.0}}
    # bytes / bandwidth is ~2 us of the 2.2 us measured: 0.9 of the bandwidth's bound
    ws = [_gemm("bfloat16", 2.2, route), _gemm("float32", 4.2, route)]
    for w in ws[1:]:
        w.attrs["w_bytes"] = 4
    result = analyze.analyze(CaptureRun(run={}, ops=[], workloads=ws), top=5, calibration=calib)
    bf16, fp32 = (result.decisions[w.workload_id] for w in ws)
    assert fp32.reason.startswith("at roofline") and not fp32.enabled
    assert bf16.enabled and bf16.eff >= analyze.AT_ROOFLINE and bf16.unpeaked == ["bfloat16"]
    (entry,) = [e for e in result.entries if e.name == "mm_bfloat16"]
    assert entry.enabled and "bfloat16 has no measured matmul peak" in entry.reason
    assert result.options["unpeaked"] == ["bfloat16"]


def test_a_matmul_dtype_held_exactly_in_a_faster_one_takes_its_peak():
    from xe_forge.frontend.calibrate import peaks

    calib = {"bandwidth_gbs": 500, "matmul_tflops": {"bfloat16": 156.2, "float16": 145.5,
                                                     "int8": 285.6, "float8_e4m3fn": 68.1}}  # fmt: skip
    bw, tflops = peaks(calib)
    assert bw == 500 and tflops["float8_e4m3fn"] == 156.2 and tflops["int8"] == 285.6
    assert tflops["float8_e5m2"] == 156.2  # unmeasured, still never slower than bf16
    assert peaks({"matmul_tflops": {"float8_e4m3fn": 68.1}})[1] == {"float8_e4m3fn": 68.1}


def test_rows_replace_batch_and_lengths(tmp_path):
    from xe_forge.frontend import cli

    f = tmp_path / "rows.yaml"
    f.write_text("rows:\n - {batch: 1, input_len: 8, output_len: 8, weight: 1}\n")
    with pytest.raises(SystemExit):
        cli.main(["run", "-o", str(tmp_path), "--framework", "x", "--model", "m",
                  "--framework-python", "py", "--rows", str(f), "--batch", "2"])  # fmt: skip


@pytest.mark.parametrize("extra", [
    ["--prompt", "a cat"],                                   # a prompt is vision's
    ["--workload", "vision", "--kv-cache", "float8_e4m3fn"],  # no KV cache in a diffusion model
    ["--workload", "vision", "--prompt-file", "p.txt"],      # token prompts are text's
    ["--workload", "action"],                                # a policy acts on what it sees
    ["--workload", "action", "--input-image", "v.png", "--negative-prompt", "n"],
    ["--workload", "action", "--input-image", "v.png", "--kv-cache", "int8"],
    ["--state", "s.npy"],                                    # a robot state is action's
    ["--workload", "action", "--input-image", "v.png", "--state", "missing.npy"],
    ["--framework-source", "https://x/y@v1"],                # the setup agent's, not a given venv's
])  # fmt: skip
def test_run_refuses_the_flags_of_the_other_workload(tmp_path, extra):
    from xe_forge.frontend import cli

    with pytest.raises(SystemExit):
        cli.main(["run", "-o", str(tmp_path), "--framework", "x", "--model", "m",
                  "--framework-python", "py", *extra])  # fmt: skip


def test_a_vision_prompt_is_the_captures_identity_and_reaches_its_agents():
    from types import SimpleNamespace

    from xe_forge.claude.generator import render
    from xe_forge.frontend import cli

    a = SimpleNamespace(workload="vision", batch=1, steps=4, height=64, width=64, frames=1,
                        input_image=[], prompt="a cat", negative_prompt=None)  # fmt: skip
    want = cli.workload(a)
    assert want["prompt"] == "a cat" and cli._same_workload(want, dict(want))
    assert not cli._same_workload(want, {**want, "prompt": cli.VISION_PROMPT})
    ctx = dict(model="m", framework="fw", python="py", cache="/c", inputs={}, precision="auto",
               quantization="auto", kv_cache="auto", refused=[], answer="a.json",
               framework_notes=[], workload="vision", runtime_keys=cli.WORKLOADS["vision"],
               repos="/r", what="x", reason="y", **want)  # fmt: skip
    capture = render("frontend-capture.md.j2", **ctx)
    assert 'The prompt is "a cat"' in capture and '"prompt", "negative_prompt"' in capture
    assert "the negative prompt" not in capture  # none given
    assert '"no blur"' in render("frontend-capture.md.j2", **{**ctx, "negative_prompt": "no blur"})
    assert 'with the prompt "a cat"' in render("frontend-patch.md.j2", **ctx)


def _kv_op(dtype, kv_args=("cache",)):
    from xe_forge.frontend.ir import CapturedOp, TensorArg

    return CapturedOp("op-0", "attn", [TensorArg([4, 64], "bfloat16"), TensorArg([9, 64], dtype)],
                      framework_meta={"arg_names": ["q", "cache"],
                                      "named": {"kv_args": list(kv_args)}})  # fmt: skip


@pytest.mark.parametrize("declared, held, kv_args, bad", [
    ("float8_e4m3fn", "uint8", ("cache",), False),         # a byte container of its size
    ("float8_e4m3fn", "float8_e4m3fn", ("cache",), False),
    ("float8_e4m3fn", "bfloat16", ("cache",), True),       # declared fp8, ran bf16
    ("bfloat16", "float16", ("cache",), True),             # same size, another dtype
    ("float8_e5m2", "float8_e4m3fn", ("cache",), True),
    ("float8_e4m3fn", "bfloat16", (), False),              # none named: nothing to check
    ("auto", "bfloat16", ("cache",), False),
    ("float8_e4m3fn", None, ("cache",), False),            # no dtype: skipped
    ("int2", "uint8", ("cache",), False),                  # sub-byte: packed in a byte
    ("uint4", "int8", ("cache",), False),
    ("int2", "int4", ("cache",), True),
    ("bfloat16", "uint8", ("cache",), True),               # a byte cannot hold two
])  # fmt: skip
def test_declared_kv_cache_against_the_named_cache_arguments(declared, held, kv_args, bad):
    from xe_forge.skills.capture import _kv_problems

    got = _kv_problems({"runtime": {"kv_cache": declared}}, [_kv_op(held, kv_args)])
    assert bool(got) == bad, got


def test_sub_byte_dtypes_are_dtypes_at_bits_over_eight():
    from xe_forge.frontend.dtypes import canonical, parse_quantization, storage_bytes, tensor_bytes

    for s in ("", "u"):
        for b in range(1, 8):
            assert storage_bytes(canonical(f"{s}int{b}")) == b / 8
    assert parse_quantization("hqq:int2") == ("hqq", "int2")
    assert tensor_bytes([4096, 4096], "int2") == 4096 * 4096 / 4
    with pytest.raises(ValueError):
        canonical("int9")


@pytest.mark.parametrize("weights, w_bytes, ok", [
    ("hqq:int2", 0.25, True),
    ("hqq:int2", 0.375, True),    # an fp16 scale and zero per 32
    ("hqq:int2", 0.5, False),     # int4's bytes: not int2
    ("hqq:int2", 0.2, False),     # fewer bytes than int2 stores
    ("gptq:int4", 0.53, True),    # 4-bit and wider keep the quarter
    ("gptq:int4", 0.7, False),
    ("int8", 1.25, True), ("int8", 1.3, False),
])  # fmt: skip
def test_sub_byte_weights_allow_their_scales_and_not_a_wider_format(weights, w_bytes, ok):
    from xe_forge.skills.capture import _precision_problems

    run = {"runtime": {"precision": "bf16", "quantization": weights},
           "precision": {"compute": "bfloat16", "dtypes": {"bfloat16": 1.0}, "w_bytes": w_bytes}}  # fmt: skip
    assert (not _precision_problems(run)) is ok, _precision_problems(run)


def test_a_generic_store_writing_other_state_too_holds_the_cache_in_some_call():
    """aten::index_put_ writes the fp8 cache and a bf16 conv state: the call order (time)
    decides nothing; a launcher whose every call holds bf16 still fails."""
    from xe_forge.skills.capture import _kv_problems

    run = {"runtime": {"kv_cache": "float8_e4m3fn"}}
    assert _kv_problems(run, [_kv_op("bfloat16"), _kv_op("uint8")]) == []
    assert _kv_problems(run, [_kv_op("bfloat16"), _kv_op("bfloat16")])


def test_kv_args_must_name_the_ops_arguments(tmp_path):
    from xe_forge.frontend import naming

    (tmp_path / "repo").mkdir()
    listing = {"op": {"args": [{"name": "q"}, {"name": "cache"}], "dims": {}}}
    entry = {"route": "kernel", "name": "k", "repo": "repo", "dsl": "sycl"}
    good = {"ops": {"op": {**entry, "kv_args": ["cache"]}}}
    assert naming.validate(good, listing, tmp_path) == []
    for bad in (["key_cache"], "cache", []):
        problems = naming.validate({"ops": {"op": {**entry, "kv_args": bad}}}, listing, tmp_path)
        assert any("kv_args" in p and "['cache', 'q']" in p for p in problems)  # what it can be


@pytest.mark.parametrize("kv_args, says", [
    ({"store.cache": ["uint8"]}, "KV cache float8_e4m3fn (checked on 1 named cache arguments)."),
    ({}, "(not checked: no op names it)")])  # fmt: skip
def test_summary_states_the_declared_kv_cache_and_whether_it_was_checked(tmp_path, kv_args,
                                                                          says):  # fmt: skip
    from xe_forge.frontend import collect

    run = {"runtime": {"precision": "bfloat16", "quantization": "auto", "kv_cache": "float8_e4m3fn"},
           "precision": {"compute": "bfloat16", "w_bytes": 2.0, "kv_args": kv_args}}  # fmt: skip
    (tmp_path / "capture.json").write_text(json.dumps({"run": run}))
    assert says in collect.collect(tmp_path, {"kernels": []}, []).read_text().splitlines()[2]


def test_capture_prompt_carries_the_kv_cache_and_run_refuses_a_bad_one(tmp_path):
    from xe_forge.claude.generator import render
    from xe_forge.frontend import cli

    text = render("frontend-capture.md.j2", model="m", framework="fw", python="py", batch=1,
                  workload="text", runtime_keys=cli.WORKLOADS["text"], framework_notes=[],
                  input_len=4, output_len=2, cache="/c", inputs={}, precision="auto",
                  quantization="auto", kv_cache="float8_e4m3fn", refused=[], answer="a.json")  # fmt: skip
    assert "the KV cache\nstored as float8_e4m3fn" in text and '"float8_e4m3fn" verbatim' in text
    with pytest.raises(SystemExit):
        cli.main(["run", "-o", str(tmp_path), "--framework", "x", "--model", "m",
                  "--framework-python", "py", "--kv-cache", "fp7"])  # fmt: skip


def _stage_named(rec: dict, name: str) -> dict:
    return next(s for s in rec["stages"] if s["stage"] == name)


def _latent(d: Path, arr) -> Path:
    import numpy as np

    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "latent_step1.npy", arr)
    return d


def test_quality_passes_byte_identical_arms(tmp_path):
    import numpy as np

    from xe_forge.frontend import quality

    x = np.arange(16, dtype=np.float32)
    rec = quality.quality(
        tmp_path, {"A": [], "B": []}, {"A": [_latent(tmp_path / "a", x)], "B": [_latent(tmp_path / "b", x)]}
    )  # fmt: skip
    assert rec["verdict"] == "identical"
    assert json.loads((tmp_path / "quality.json").read_text())["verdict"] == "identical"


@pytest.mark.parametrize(
    "scale, verdict",
    [(1.0, "pass"), (100.0, "fail")],  # B as close to the reference as A, and far worse
)
def test_quality_judges_the_gate_against_a_float32_reference(tmp_path, scale, verdict):
    import numpy as np

    from xe_forge.frontend import quality

    rng = np.random.default_rng(0)
    ref = rng.standard_normal((4, 16)).astype(np.float32)
    (tmp_path / "reference").mkdir()
    np.save(tmp_path / "reference" / "step1.npy", ref)
    a = _latent(tmp_path / "a", ref + rng.standard_normal(ref.shape) * 1e-3)
    b = _latent(tmp_path / "b", ref + rng.standard_normal(ref.shape) * 1e-3 * scale)
    rec = quality.quality(tmp_path, {"A": [], "B": []}, {"A": [a], "B": [b]})
    assert rec["verdict"] == verdict
    assert "float32 reference" in _stage_named(rec, "step1")["why"]


def test_quality_without_a_reference_uses_the_absolute_convention(tmp_path):
    """No reference: A against B, judged by the stage's documented limits, and said to be so."""
    import numpy as np

    from xe_forge.frontend import quality

    base = np.ones((8, 8), dtype=np.float32)
    a = _latent(tmp_path / "a", base)
    near = _latent(tmp_path / "b", base * (1 + 1e-5))
    far = _latent(tmp_path / "c", base * 2)
    assert quality.quality(tmp_path, {"A": []}, {"A": [a], "B": [near]})["verdict"] == "pass"
    rec = quality.quality(tmp_path, {"A": []}, {"A": [a], "B": [far]})
    assert rec["verdict"] == "fail"
    assert "convention" in _stage_named(rec, "step1")["why"]


def test_quality_lets_the_worst_gate_stage_decide(tmp_path):
    """A byte-identical prompt embedding must not cover a bad step-1 latent."""
    import numpy as np

    from xe_forge.frontend import quality

    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        np.save(d / "prompt_embeds.npy", np.ones((2, 8), dtype=np.float32))
    np.save(a / "latent_step1.npy", np.ones((4, 4), dtype=np.float32))
    np.save(b / "latent_step1.npy", np.ones((4, 4), dtype=np.float32) * 3)
    rec = quality.quality(tmp_path, {"A": []}, {"A": [a], "B": [b]})
    assert _stage_named(rec, "prompt")["verdict"] == "identical"
    assert _stage_named(rec, "step1")["verdict"] == "fail"
    assert rec["verdict"] == "fail"


@pytest.mark.parametrize("nan, verdict", [(False, "identical"), (True, "fail")])
def test_a_guard_fails_the_verdict_only_on_a_catastrophe(tmp_path, nan, verdict):
    """The gate holds; a guard further from the reference warns, a NaN in it fails."""
    import numpy as np

    from xe_forge.frontend import quality

    rng = np.random.default_rng(0)
    ref = rng.standard_normal((4, 16)).astype(np.float32)
    (tmp_path / "reference").mkdir()
    np.save(tmp_path / "reference" / "latent.npy", ref)
    a, b = tmp_path / "a", tmp_path / "b"
    for d, noise in ((a, 1e-3), (b, 1e-1)):
        _latent(d, np.ones((4, 4), dtype=np.float32))
        final = ref + rng.standard_normal(ref.shape).astype(np.float32) * noise
        if nan and d == b:
            final[0, 0] = np.nan
        np.save(d / "latent_final.npy", final)
    rec = quality.quality(tmp_path, {"A": []}, {"A": [a], "B": [b]})
    assert _stage_named(rec, "latent")["verdict"] == "fail"
    assert rec["verdict"] == verdict


def test_patch_check_judges_vision_against_the_native_path(tmp_path):
    """No tokens: the reference is generate.sh under XE_FORGE_NATIVE, and the op gain is timed."""
    from xe_forge.skills import patch

    work = tmp_path / "patch-1"
    work.mkdir()
    (work / "generate.sh").write_text(
        "set -e\n"
        'python3 -I -c "\n'
        "import os, numpy as np\n"
        "out = os.environ['OUT']\n"
        "native = os.environ.get('XE_FORGE_NATIVE') == '1'\n"
        "bad = native and os.environ.get('BREAK') == '1'\n"
        "np.save(out + '/prompt_embeds.npy', np.ones((2, 4), dtype=np.float32))\n"
        "np.save(out + '/latent_step1.npy', np.full((4, 4), 3.0 if bad else 1.0, np.float32))\n"
        '"\n'
    )

    def patched_out(d: Path) -> Path:
        d.mkdir()
        assert subprocess.run(["bash", str(work / "generate.sh")], cwd=work,
                              env={**os.environ, "OUT": str(d)}).returncode == 0  # fmt: skip
        return d

    rec: dict = {"patched_s": 1.0}
    assert patch._by_native(work, sys.executable, patched_out(tmp_path / "got"), rec) == []
    assert rec["quality"]["verdict"] == "identical"
    assert "op_gain" in rec and json.loads((work / "quality.json").read_text())

    # Now the native path disagrees: the patch must fail its own gate.
    os.environ["BREAK"] = "1"
    try:
        rec2: dict = {"patched_s": 1.0}
        problems = patch._by_native(work, sys.executable, patched_out(tmp_path / "got2"), rec2)
    finally:
        del os.environ["BREAK"]
    assert problems and "does not match" in problems[0]
    assert _stage_named(rec2["quality"], "step1")["verdict"] == "fail"


def test_every_claude_session_lifts_the_background_ceiling(monkeypatch):
    # Headless claude's own background ceiling must not stop a benchmark before its watchdog
    # can print a verdict; frontend and kernel sessions share the lifted ceiling.
    from xe_forge.engines.claude_engine import claude_env
    from xe_forge.frontend.agent import TIMEOUT

    config = SimpleNamespace(llm=SimpleNamespace(api_base=None, api_key=None, model=None))
    monkeypatch.delenv("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS", raising=False)
    assert claude_env(config)["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] == str(TIMEOUT * 1000)
    monkeypatch.setenv("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS", "5")
    assert claude_env(config)["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] == "5"  # the caller's wins


def test_a_port_back_is_its_commits_with_a_pr_carrying_the_measured_numbers(tmp_path, monkeypatch):
    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.skills import upstream

    repo = tmp_path / "kernels"
    _git_repo(repo)
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    mgr = TrialManager(ws / "trials")
    (ws / "base.cpp").write_text("// baseline\n")
    mgr.init("k", ws / "base.cpp")
    (ws / "t.cpp").write_text("// faster\n")
    tid = mgr.save_trial("k", ws / "t.cpp", strategy="tile 64")
    mgr.record_result(
        "k",
        tid,
        correctness="pass",
        baseline_us=20.0,
        triton_us=10.0,
        speedup=2.0,
        source="measured",
    )
    args = SimpleNamespace(
        kernel_name="k", kind="upstream", notes="", stack="", trials_dir="trials"
    )
    upstream._clone(SimpleNamespace(repo=str(repo), name=None, rev=None))
    clone = ws / "upstream"

    def git(*a):
        subprocess.run(["git", "-C", str(clone), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                       check=True, capture_output=True)  # fmt: skip

    (clone / "a.txt").write_text("b\n")
    assert upstream._patch(args) == 1  # uncommitted: no patch would carry it
    git("commit", "-qam", "Tile the loop by 64")
    assert upstream._patch(args) == 1  # no PR.md yet
    (ws / "output").mkdir(exist_ok=True)
    (ws / "output" / "PR.md").write_text("Tile the loop.\n")
    assert upstream._patch(args) == 0
    assert "+b" in (ws / "output" / "upstream.patch").read_text()
    assert "Subject: [PATCH] Tile the loop by 64" in (ws / "output" / "upstream.mbox").read_text()
    pr = (ws / "output" / "PR.md").read_text()
    assert "| baseline (us) | 20.0 |" in pr and "| speedup | 2.0 |" in pr
    upstream._clone(SimpleNamespace(repo=str(repo), name=None, rev=None))  # reused after a commit
    assert upstream._patch(args) == 0 and (ws / "output" / "PR.md").read_text() == pr


def test_vision_run_reaches_its_first_check(tmp_path):
    """A vision run must get through argument handling without touching an agent.

    The missing input image is the first check past that point and it needs no agent, no GPU
    and no model, which is what makes it usable as the probe.
    """
    out = subprocess.run(
        [sys.executable, "-m", "xe_forge.cli", "run", "--framework", "vllm-omni",
         "--workload", "vision", "--model", "X", "--framework-python", "/bin/false",
         "-o", str(tmp_path / "run"), "--steps", "20", "--height", "1024", "--width", "1024",
         "--frames", "1", "--batch", "1", "--input-image", str(tmp_path / "absent.png")],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert "AttributeError" not in out.stderr, out.stderr[-600:]
    assert "no such input image" in out.stderr
    assert out.returncode == 1


def test_no_prompt_is_the_same_whether_none_or_empty():
    """A capture is not a different workload because it recorded '' where the CLI held None."""
    from xe_forge.frontend.cli import _same_workload

    want = {"framework": "vllm-omni", "steps": 20, "prompt": "a fox", "negative_prompt": None}
    assert _same_workload(want, {**want, "negative_prompt": ""})
    assert _same_workload({**want, "negative_prompt": ""}, want)
    # A real negative prompt still has to match.
    assert not _same_workload(want, {**want, "negative_prompt": "blurry"})
    # And a field that is not a prompt keeps comparing exactly: 0 is not None.
    assert not _same_workload({**want, "steps": None}, {**want, "steps": 0})


def test_an_action_run_is_a_policy_on_what_it_sees_and_its_agents_save_the_actions(tmp_path):
    from xe_forge.claude.generator import render
    from xe_forge.frontend import cli

    a = SimpleNamespace(workload="action", batch=1, steps=None, calls=8, prompt="open it",
                        input_image=[str(tmp_path / "b.png"), str(tmp_path / "a.png")],
                        state=str(tmp_path / "s.npy"), framework="fw", model="m", inputs={},
                        precision="auto", quantization="auto", kv_cache="auto")  # fmt: skip
    want = cli.workload(a)
    assert want["input_images"] == sorted(want["input_images"]) and want["state"].endswith("s.npy")
    assert "steps" not in cli._wanted(a)  # the policy's own: whatever it ran is this run
    assert cli._wanted(SimpleNamespace(**{**vars(a), "steps": 10}))["steps"] == 10
    assert not cli._same_workload(want, {**want, "state": None})
    assert not cli._same_workload({**want, "steps": 10}, {**want, "steps": 4})
    ctx = dict(model="m", framework="fw", python="py", cache="/c", inputs={}, precision="auto",
               quantization="auto", kv_cache="auto", refused=[], answer="a.json",
               framework_notes=[], workload="action", runtime_keys=cli.WORKLOADS["action"],
               repos="/r", what="x", reason="y", **want)  # fmt: skip
    capture = render("frontend-capture.md.j2", **ctx)
    assert "called 8 times" in capture and 'The instruction is "open it"' in capture
    assert "`actions.npy`" in capture and "the KV cache" not in capture
    assert f'"state" "{want["state"]}" -- the path, not what it holds' in capture
    assert "denoising its actions over" not in capture  # no step count given: the policy's own
    patch = render("frontend-patch.md.j2", **{**ctx, "steps": 4})
    assert "call the policy m" in patch and "4 denoising steps" in patch
    assert "`actions.npy`" in patch and "XE_FORGE_NATIVE=1" in patch and "--input-len" not in patch


def test_a_policys_actions_are_compared_when_nothing_else_was_saved(tmp_path):
    import numpy as np

    from xe_forge.frontend import quality

    for arm, v in (("A", 1.0), ("B", 1.0 + 1e-5)):
        (tmp_path / arm).mkdir()
        np.save(tmp_path / arm / "actions.npy", np.full((4, 7), v, dtype=np.float32))
    dirs = {"A": [tmp_path / "A"], "B": [tmp_path / "B"]}
    judged = quality.quality(tmp_path, dirs, dirs)
    stage = next(s for s in judged["stages"] if s["stage"] == "actions")
    assert stage["verdict"] == "pass" and judged["verdict"] == "pass"


def test_a_framework_source_is_a_repository_and_maybe_a_ref():
    from xe_forge.frontend import cli

    assert cli._source("https://github.com/o/r@v1.1.0") == ("https://github.com/o/r", "v1.1.0")
    assert cli._source("https://github.com/o/r") == ("https://github.com/o/r", None)
    assert cli._source("git@github.com:o/r") == ("git@github.com:o/r", None)
    assert cli._source("git@github.com:o/r@release/2") == ("git@github.com:o/r", "release/2")
    assert cli._source(None) == (None, None)


def _setup_args(tmp_path, **kw):
    return SimpleNamespace(framework="fw", model="m", device="xpu", framework_note=["given"],
                           framework_source="https://x/fw@v2", framework_venv=tmp_path / "venv",
                           **kw)  # fmt: skip


def _fake_build(tmp_path, calls, sees=True, python=None, commit="c1"):
    def agent(step, template, context, cwd, answer, py, turns):
        calls.append((step, template, context))
        exe = Path(python or f"{context['venv']}/bin/python")
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text(f"#!/bin/sh\nexit {0 if sees else 1}\n")  # the device probe's verdict
        exe.chmod(0o755)
        return {
            "python": str(exe),
            "source": "https://x/fw",
            "ref": "v2",
            "commit": commit,
            "version": "2.0",
            "env": {"XF_TEST_SETUP_ATTN": "sdpa"},
            "notes": ["two views"],
            "dropped": [{"package": "cuda-only", "why": "CUDA"}],
        }

    return agent


def test_a_framework_the_setup_agent_built_is_reused_with_its_env_and_notes(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    calls: list = []
    monkeypatch.setattr(cli, "_agent", _fake_build(tmp_path, calls))
    monkeypatch.delenv("XF_TEST_SETUP_ATTN", raising=False)
    a = _setup_args(tmp_path)
    python = cli.setup(a, tmp_path / "run", 5)
    assert python == str(tmp_path / "venv" / "bin" / "python") and len(calls) == 1
    _, template, context = calls[0]
    assert template == "frontend-setup.md.j2" and context["ref"] == "v2"
    assert os.environ["XF_TEST_SETUP_ATTN"] == "sdpa" and a.framework_note == ["given", "two views"]
    record = json.loads((tmp_path / "venv" / "xe-forge-setup.json").read_text())
    assert record["asked"] == {"framework": "fw", "source": "https://x/fw@v2"}
    monkeypatch.delenv("XF_TEST_SETUP_ATTN")
    b = _setup_args(tmp_path)
    assert cli.setup(b, tmp_path / "run", 5) == python and len(calls) == 1  # no second build
    assert os.environ["XF_TEST_SETUP_ATTN"] == "sdpa" and b.framework_note == ["given", "two views"]
    monkeypatch.delenv("XF_TEST_SETUP_ATTN")
    # the same venv asked for another source is refused, not rebuilt over
    other = _setup_args(tmp_path)
    other.framework_source = "https://x/fw@v3"
    assert cli.setup(other, tmp_path / "run", 5) is None and len(calls) == 1


def test_a_capture_made_on_another_build_of_the_framework_is_refused(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    calls: list = []
    monkeypatch.setattr(cli, "_agent", _fake_build(tmp_path, calls, commit="c1"))
    run = tmp_path / "run"
    assert cli.setup(_setup_args(tmp_path), run, 5)
    (run / "capture.json").write_text("{}")
    (tmp_path / "venv" / "xe-forge-setup.json").unlink()  # the venv rebuilt, at another commit
    monkeypatch.setattr(cli, "_agent", _fake_build(tmp_path, calls, commit="c2"))
    assert cli.setup(_setup_args(tmp_path), run, 5) is None
    monkeypatch.delenv("XF_TEST_SETUP_ATTN", raising=False)


@pytest.mark.parametrize("bad", ["outside", "blind"])
def test_a_framework_build_is_kept_only_in_its_venv_and_seeing_the_device(tmp_path, monkeypatch,
                                                                          bad):  # fmt: skip
    from xe_forge.frontend import cli

    calls: list = []
    build = (_fake_build(tmp_path, calls, python=tmp_path / "elsewhere" / "python")
             if bad == "outside" else _fake_build(tmp_path, calls, sees=False))  # fmt: skip
    monkeypatch.setattr(cli, "_agent", build)
    assert cli.setup(_setup_args(tmp_path), tmp_path / "run", 5) is None
    assert not (tmp_path / "venv" / "xe-forge-setup.json").exists()


def test_the_setup_agent_builds_the_venv_and_only_it():
    from xe_forge.claude.generator import render

    ctx = {
        "framework": "fw",
        "model": "org/model",
        "device": "xpu",
        "source": "https://x/fw",
        "ref": "v1.1.0",
        "venv": "/v",
        "answer": "s.json",
    }
    text = render("frontend-setup.md.j2", **ctx)
    assert "at v1.1.0" in text and "/v/src" in text and "PyTorch's own index" in text
    assert '"dropped"' in text and "Change no source" in text and "Do not load" in text
    assert "latest release" in render("frontend-setup.md.j2", **{**ctx, "source": None})
    assert '"prepend"' in text and "Never record a whole path variable" in text
    redo = render("frontend-setup.md.j2", **{**ctx, "redo": ["env X replaces this node's own"]})
    assert "Keep what is built" in redo and "env X replaces" in redo and "Keep what" not in text
    rules = render("frontend-rules.md.j2", python="py", answer="s.json", venv="/v")
    assert "building the framework's venv, /v" in rules and "never use uv" not in rules
    assert "never use uv" in render("frontend-rules.md.j2", python="py", answer="s.json")


@pytest.mark.parametrize("workload", ["vision", "action"])
def test_a_valid_non_text_run_gets_past_its_flags_to_the_framework(tmp_path, monkeypatch,
                                                                   workload):  # fmt: skip
    from xe_forge.frontend import cli

    image, state = tmp_path / "v.png", tmp_path / "s.npy"
    image.write_bytes(b"png"), state.write_bytes(b"npy")
    reached = []
    monkeypatch.setattr(cli, "setup", lambda a, run_dir, turns: reached.append(a) and None)
    extra = ["--state", str(state)] if workload == "action" else []
    code = cli.main(["run", "-o", str(tmp_path / "run"), "--framework", "x", "--model", "m",
                     "--framework-source", "https://x/y@v1", "--workload", workload,
                     "--input-image", str(image), *extra])  # fmt: skip
    assert code == 1 and len(reached) == 1  # the fake setup builds nothing: stopped there
    assert reached[0].batch == 1 and reached[0].prompt


def test_patch_check_of_a_run_without_tokens_needs_no_input_len(tmp_path):
    work, repos, fw, _ = _patch_dir(tmp_path, [[1, 2, 3, 4]])
    (work / "generate.sh").write_text(
        'test -z "${INPUT_LEN+x}" || exit 9\n'  # not set when not given
        'python3 -I -c "import os, numpy as np; '
        "np.save(os.environ['OUT'] + '/actions.npy', np.ones((16, 7), np.float32))\"\n"
    )
    argv = ["patch", "check", str(work), "--repos", str(repos), "--framework-python", str(fw),
            "--model", "toy", "--batch", "1"]  # fmt: skip
    code, out = _tool(*argv)
    assert code == 0, out
    assert _stage_named(json.loads((work / "check.json").read_text())["quality"], "actions")
    (work / "generate.sh").write_text("echo '[[1, 2, 3, 4]]' > \"$OUT/tokens.json\"\n")
    code, out = _tool(*argv)  # tokens are scored on a prompt: its length is needed
    assert code != 0 and "pass --input-len" in out


def test_a_build_only_adds_to_the_environment_it_runs_in(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    venv = tmp_path / "venv"
    monkeypatch.setenv("XF_TEST_LIBS", "/node/lib")
    full, problems = cli._setup_environ({"prepend": {"XF_TEST_LIBS": [f"{venv}/lib"]}}, venv)
    assert full["XF_TEST_LIBS"] == f"{venv}/lib{os.pathsep}/node/lib" and not problems
    _, problems = cli._setup_environ({"env": {"XF_TEST_LIBS": f"{venv}/lib:/node/lib"}}, venv)
    assert problems and "replaces this node's own" in problems[0]
    _, problems = cli._setup_environ({"prepend": {"XF_TEST_LIBS": "/opt/elsewhere"}}, venv)
    assert problems and "not inside" in problems[0]

    calls: list = []
    monkeypatch.setattr(cli, "_agent", _fake_build(tmp_path, calls))
    monkeypatch.delenv("XF_TEST_SETUP_ATTN", raising=False)
    assert cli.setup(_setup_args(tmp_path), tmp_path / "run", 5) and len(calls) == 1
    record = venv / "xe-forge-setup.json"
    old = json.loads(record.read_text())
    record.write_text(json.dumps({**old, "env": {"XF_TEST_LIBS": "/snapshot"}}))
    assert cli.setup(_setup_args(tmp_path), tmp_path / "run", 5) and len(calls) == 2
    assert calls[1][2]["redo"] and "XF_TEST_LIBS" in calls[1][2]["redo"][0]  # repaired, not reused
    assert os.environ["XF_TEST_LIBS"] == "/node/lib"  # the node's own value survives
    monkeypatch.delenv("XF_TEST_SETUP_ATTN", raising=False)


def test_the_frameworks_environment_stays_out_of_xe_forges_own_sessions(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    monkeypatch.setattr(cli, "_NODE_ENV", {})
    monkeypatch.setenv("XF_TEST_LIBS", "/node/lib")
    monkeypatch.delenv("XF_TEST_SETUP_ATTN", raising=False)
    build = _fake_build(tmp_path, [])

    def with_libs(*args):
        got = build(*args)
        return {**got, "prepend": {"XF_TEST_LIBS": [f"{tmp_path / 'venv'}/lib"]}}

    monkeypatch.setattr(cli, "_agent", with_libs)
    assert cli.setup(_setup_args(tmp_path), tmp_path / "run", 5)
    assert os.environ["XF_TEST_LIBS"].startswith(f"{tmp_path / 'venv'}/lib")  # the framework's
    assert os.environ["XF_TEST_SETUP_ATTN"] == "sdpa"
    own = cli._session_env()
    assert own["XF_TEST_LIBS"] == "/node/lib" and "XF_TEST_SETUP_ATTN" not in own
    monkeypatch.delenv("XF_TEST_SETUP_ATTN")


def test_no_framework_or_model_is_named_in_the_frontend_templates_or_help(capsys):
    # Agents find the framework's specifics; a name in a template or a flag's help steers
    # every other framework toward one framework's options.
    import re

    from xe_forge.frontend import cli

    names = re.compile(
        r"\b(vllm|openvino|sglang|llama|ltx|flux|natten|rldx|qwen|wan2?|"
        r"onednn|onemkl|cutlass|ocloc|grf)\b",
        re.I,
    )
    templates = Path(cli.SRC_DIR) / "xe_forge" / "claude" / "templates"
    files = [*sorted(templates.glob("frontend-*.j2")), templates / "kernel-locator.md.j2"]
    found = [f"{f.name}:{i}: {line.strip()}" for f in files
             for i, line in enumerate(f.read_text().splitlines(), 1) if names.search(line)]  # fmt: skip
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    found += [f"run --help: {line.strip()}" for line in capsys.readouterr().out.splitlines()
              if names.search(line)]  # fmt: skip
    assert not found, "\n".join(found)


def test_a_resume_checks_a_finished_patch_again_rather_than_redoing_it(tmp_path, monkeypatch):
    from xe_forge.frontend import cli

    seen, verdicts = [], iter([1, 0])  # the first check fails, the resumed one passes

    def agent(step, template, ctx, cwd, answer, python, turns):
        seen.append((step, python, tuple(ctx.get("redo") or ())))
        if step == "patch":
            (cwd / "patch.json").write_text('{"python": "/patched/python"}')
            (cwd / "patched.json").write_text("{}")  # agent.run's answer: written once done
            return {"notes": "patched"}
        if python == "/fw/python":
            return {"unsupported": "op", "what": "gqa", "reason": "attn.py:1"}
        return {"launcher": "launcher/run.sh"}

    def tool(py, *argv, problems=None, **env):
        seen.append((argv[0], py, ()))
        return next(verdicts) if argv[0] == "patch" else 0

    monkeypatch.setattr(cli, "_agent", agent)
    monkeypatch.setattr(cli, "_tool", tool)
    a = SimpleNamespace(model="m", framework="fw", workload="action", batch=1, calls=8,
                        steps=None, input_image=["/i/a.png"], state=None, prompt="go")  # fmt: skip
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "actions.npy").write_bytes(b"x")
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")[0] == 1
    assert json.loads((tmp_path / "patch-1" / "request.json").read_text())["what"] == "gqa"
    seen.clear()
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos") == (0, "/patched/python")
    assert [s[0] for s in seen] == ["patch", "capture", "capture"]  # check, capture agent, parse
    assert seen[1][1] == "/patched/python"

    # A kept patch whose check fails again goes back to its agent with the failures.
    (tmp_path / "stack.json").unlink()
    seen.clear()
    verdicts = iter([1, 0])

    def failing(py, *argv, problems=None, **env):
        seen.append((argv[0], py, ()))
        if argv[0] == "patch" and problems is not None:
            problems.append("prompt_embeds does not match")
        return next(verdicts) if argv[0] == "patch" else 0

    monkeypatch.setattr(cli, "_tool", failing)
    assert cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")[0] == 0
    assert seen[1] == ("patch", "/fw/python", ("prompt_embeds does not match",))
    assert "do not start over" in render_patch_redo()


def render_patch_redo():
    from xe_forge.claude.generator import render
    from xe_forge.frontend import cli

    want = {"batch": 1, "steps": None, "calls": 8, "input_images": ["/a"], "state": None,
            "prompt": "go"}  # fmt: skip
    return render("frontend-patch.md.j2", model="m", framework="fw", python="py", cache="/c",
                  inputs={}, precision="auto", quantization="auto", kv_cache="auto",
                  refused=[], answer="a.json", framework_notes=[], workload="action",
                  runtime_keys=cli.WORKLOADS["action"], repos="/r", what="x", reason="y",
                  redo=["prompt_embeds does not match"], **want)  # fmt: skip


def test_a_launcher_that_misrecords_the_run_it_was_asked_for_goes_back_to_its_agent(
    tmp_path, monkeypatch
):
    from xe_forge.frontend import cli, ir

    image, state, run = tmp_path / "v.png", tmp_path / "s.npy", tmp_path / "run"
    image.write_bytes(b"png"), state.write_bytes(b"npy")
    recorded = iter([[0.5, 0.1], str(state)])  # first the values, then, fixed, the path
    calls = []

    def capture(a, python, run_dir, turns, repos, refused=None):
        calls.append(refused)
        rt = {"batch": 1, "calls": 8, "steps": 4, "prompt": a.prompt,
              "input_images": [str(image)], "state": next(recorded)}  # fmt: skip
        (run_dir / "capture.json").write_text(json.dumps(rt))
        return 0, python

    monkeypatch.setattr(cli, "capture", capture)
    monkeypatch.setattr(
        ir,
        "load",
        lambda p: SimpleNamespace(
            run={
                "framework": "x",
                "model": {"name": "m"},
                "runtime": json.loads(Path(p).read_text()),
            }
        ),
    )
    monkeypatch.setattr(cli, "name_and_read", lambda *a, **k: (7, "py"))  # stop after the check
    argv = [
        "run",
        "-o",
        str(run),
        "--framework",
        "x",
        "--model",
        "m",
        "--framework-python",
        "py",
        "--workload",
        "action",
        "--input-image",
        str(image),
        "--state",
        str(state),
    ]
    assert cli.main(argv) == 7
    assert calls[0] is None and "records" in calls[1][0] and "0.5" in calls[1][0]
    other = tmp_path / "t.npy"
    other.write_bytes(b"npy")
    calls.clear()  # a resume asking for another state: not this capture, and not its launcher's
    assert cli.main([*argv[:-1], str(other)]) == 1 and calls == []


@pytest.mark.parametrize("b_calls", [[100, 50, 50], [100, 100, None]])
def test_compare_reads_calls_that_differ_across_an_arms_captures_as_calls_differ(tmp_path, b_calls):
    from xe_forge.frontend import compare, ir
    from xe_forge.frontend.ir import CaptureRun, Naming, Workload

    def cap(name, calls):
        ws = [] if calls is None else [
            Workload(workload_id="gemm/a", family="gemm", dims={}, dtypes=[], layout=[], attrs={},
                     var_axes={}, calls=calls, device_us=float(calls), share=1.0, ops=[],
                     naming=Naming("kernel"))]  # fmt: skip
        run = {"framework": "f", "model": {"name": "m"}, "total_device_us": 100.0}
        ir.dump(CaptureRun(run=run, ops=[], workloads=ws), tmp_path / f"{name}.json")
        return tmp_path / f"{name}.json"

    a = [cap(f"a{i}", 100) for i in range(3)]
    b = [cap(f"b{i}", c) for i, c in enumerate(b_calls)]
    row = next(r for r in compare.compare(a, b) if r["workload_id"] == "gemm/a")
    assert row["status"] == "calls_differ" and row["speedup"] is None
    assert row["per_call_b"] is None and row["calls_a"] == 100


def _compare_cap(path: Path, config: dict, runtime: dict, total: float) -> Path:
    from xe_forge.frontend import ir
    from xe_forge.frontend.ir import CaptureRun

    run = {"framework": "f", "model": {"name": "m", "config": config}, "runtime": runtime,
           "total_device_us": total}  # fmt: skip
    ir.dump(CaptureRun(run=run, ops=[], workloads=[]), path)
    return path


@pytest.mark.parametrize(
    "a, b, same",
    [(({"layers": 2}, {}), ({"layers": 4}, {}), False),
     (({"layers": 2, "heads": 8}, {"inputs": {"p": {"file": "/x/p.txt", "sha256": "aa"}},
                                   "opts": {"u": 1, "v": 2}}),
      ({"heads": 8, "layers": 2}, {"opts": {"v": 2, "u": 1},
                                   "inputs": {"p": {"file": "/y/p.txt", "sha256": "aa"}}}), True),
     (({}, {"inputs": {"p": {"file": "/x/p.txt", "sha256": "aa"}}}),
      ({}, {"inputs": {"p": {"file": "/x/p.txt", "sha256": "bb"}}}), False),
     (({}, {"inputs": {"p": {"file": "/x/p.txt"}}}),
      ({}, {"inputs": {"p": {"file": "/y/q.txt"}}}), False)],
)  # fmt: skip
def test_compare_keys_arms_by_config_and_input_content(tmp_path, a, b, same):
    from xe_forge.frontend import compare

    pa = [_compare_cap(tmp_path / "a.json", *a, 200.0)]
    pb = [_compare_cap(tmp_path / "b.json", *b, 100.0)]
    if same:
        assert compare.compare(pa, pb)[0]["speedup"] == 2.0
    else:
        with pytest.raises(SystemExit, match="different work"):
            compare.compare(pa, pb)
