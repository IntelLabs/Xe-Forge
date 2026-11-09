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
from xe_forge.skills.capture import _counters

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


@pytest.mark.parametrize("config_args, n", [(None, 1), (["ARG1"], 2)])
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
 "runtime": {"batch": 1}, "device": {"type": "xpu"}}
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
    assert code == 0 and "rms_v2" in [r["kernel"] for r in json.loads(out)], out


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
{"framework": "toyfw", "model": {"name": "toy"}, "runtime": {"batch": 1}, "device": {}}
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
    float32 pass fails); each call's argv appended to ``argv``."""
    table = json.dumps({"bfloat16": [list(g) for g in bf16],
                        "float32": [list(g) for g in fp32] if fp32 else None})  # fmt: skip
    helper = path.with_suffix(".py")
    helper.write_text(f"""import json, sys
table = json.loads({table!r})
dtype, rows = sys.argv[9], sys.argv[10]
if {str(argv)!r} != "None":
    open({str(argv)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
if table[dtype] is None:
    sys.exit("float32 does not load")
rows = [int(i) for i in rows.split(",")] if rows else range(len(table[dtype]))
print(json.dumps({{"gaps": {{str(i): table[dtype][i] for i in rows}}, "device": "cpu",
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
    ([[0] * 4, [0, 0, 0.0, 0]], 0, [0, 0.0], ["bf16", "fp32"]),  # bf16 wrong there: fp32 decides
    ([[0] * 4, [0, 0, 1.2, 0]], 1, [0, 1.2], ["bf16", "fp32"]),  # fp32 agrees: still a FAIL
    ([[0] * 4, [0.5, 0, 0, 0]], 1, [0, 0.5], ["bf16", "fp32"]),  # fp32 judges every position
    (None, 1, [0, 1.4375], ["bf16", "bf16"]),  # the float32 pass fails: bf16's verdict stands
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
    assert rec["bf16_gaps"][1] == [0, 0, 1.4375, 0]
    calls = [json.loads(line) for line in argv.read_text().splitlines()]
    assert [(c[8], c[9]) for c in calls] == [("bfloat16", ""), ("float32", "1")]  # only row 1


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
    a = SimpleNamespace(model="m", framework="fw", batch=1, input_len=4, output_len=2,
                        reference_python="/ref/python")  # fmt: skip
    code, python = cli.capture(a, "/fw/python", tmp_path, 5, tmp_path / "repos")
    assert (code, python) == (0, "/patched/python")
    assert seen == [("capture", "/fw/python"), ("patch", "/fw/python"), ("patch", "/fw/python"),
                    ("capture", "/patched/python"), ("capture", "/patched/python")]  # fmt: skip
    assert json.loads((tmp_path / "stack.json").read_text())["python"] == "/patched/python"
    assert checks[0][-2:] == ("--reference-python", "/ref/python")


def test_prompt_file_gives_each_row_its_own_slice_and_refuses_too_little_text(tmp_path):
    from xe_forge.frontend import profile

    vocab = profile.Vocab(lambda text: [ord(c) for c in text], range(10))
    f = tmp_path / "prompts.txt"
    f.write_text("abcdefgh")
    rows = profile.token_ids(vocab, 2, 4, f)
    assert rows == [[97, 98, 99, 100], [101, 102, 103, 104]] == profile.token_ids(vocab, 2, 4, f)
    with pytest.raises(SystemExit, match="need 9"):
        profile.token_ids(vocab, 3, 3, f)  # 8 ids for 3 rows of 3


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


def test_capture_parse_refuses_inputs_that_changed_since_the_run(tmp_path):
    from xe_forge.skills.capture import _sha256

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("some text")
    inputs = json.dumps({"prompts": {"file": str(prompts), "sha256": _sha256(prompts)}})
    unitrace = _exe(tmp_path / "unitrace", UNITRACE_STUB)
    launcher = _exe(tmp_path / "run.sh", (LAUNCHER % "").replace(
        '"runtime": {"batch": 1}', f'"runtime": {{"batch": 1, "inputs": {inputs}}}'))  # fmt: skip
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
    a = SimpleNamespace(model="m", framework="fw", batch=1, input_len=4, output_len=2,
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
    launcher = _exe(tmp_path / "run.sh", (LAUNCHER % "").replace('"float16"', '"f16"', 1))
    raw = tmp_path / "raw"
    assert (
        _tool("capture", "run", str(launcher), "-o", str(raw), "--unitrace", str(unitrace))[0] == 0
    )
    code, out = _tool("capture", "parse", str(raw), "-o", str(tmp_path / "capture.json"))
    assert code == 1 and "dtype 'f16'" in out, out


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
