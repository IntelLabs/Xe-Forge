import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

AGENT = Path(__file__).resolve().parents[1] / "scripts" / "serve_agent"
spec = importlib.util.spec_from_file_location("serve_trial", AGENT / "serve_trial.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

BASE = {
    "node": "n",
    "vllm": "v",
    "output_tok_s": 1000.0,
    "floor_pct": 1.0,
    "spread_pct": 0.5,
    "probe": 0.9,
}


def result(**kw):
    r = {
        "node": "n",
        "vllm": "v",
        "output_tok_s": 1100.0,
        "spread_pct": 0.5,
        "mean_tpot_ms": 20.0,
        "tokens_same": 4,
        "tokens_total": 4,
        "probe": 0.9,
    }
    return {**r, **kw}


def test_verdicts_in_order():
    conf = {"models": ["m"], "gpus": 4}
    assert "stack" in st.invalid({"model": "m", "patches": ["p"]}, conf)
    assert st.invalid({"model": "m", "tp": 2, "serve_args": ["-dp", "4"]}, conf).startswith(
        "world size 8"
    )
    assert st.invalid({"model": "x"}, conf).startswith("model")
    assert (
        st.invalid({"model": "m", "tp": 2, "serve_args": ["--data-parallel-size=2"]}, conf) is None
    )
    assert st.judge(result(failed=True), BASE, 50)[0] == "ARM_FAILED"
    assert st.judge(result(vllm="other"), BASE, 50)[0] == "UNCOMPARABLE"
    assert st.judge(result(mean_tpot_ms=60), BASE, 50)[0] == "OVER_CAP"
    assert st.judge(result(tokens_same=2, probe=0.8), BASE, 50)[0] == "QUALITY_FAILED"
    assert st.judge(result(tokens_same=2, probe=0.9), BASE, 50)[0] == "WIN"
    assert st.judge(result(output_tok_s=1005), BASE, 50)[0] == "NOISE"
    assert st.judge(result(output_tok_s=900), BASE, 50)[0] == "LOSS"


FAKE = """#!/bin/bash
# tok/s from the trial: 1000, +200 with --max-num-seqs, +-3 between launches.
n=$(ls "$(dirname "$RUN_DIR")" | wc -l)
tok=$((1000 + n % 2 * 3))
[[ "$SERVE_ARGS" == *max-num-seqs* ]] && tok=$((tok + 200))
mkdir -p "$RUN_DIR"
python3 -c "import json,sys; p=json.load(open(sys.argv[1])); json.dump([x['answer'][0] for x in p], open(sys.argv[2],'w'))" \\
    "$GREEDY_PROMPTS" "$RUN_DIR/greedy.json"
echo "NODE: n"; echo "VLLM: v"
echo "ENV_PATCHES: $PATCHES"; echo "ENV_OVERLAY: $XPU_KERNELS_OVERLAY"; echo "ENV_GPU_VARS: $GPU_VARS"
echo "ENV_GREEDY_TOKENS: $GREEDY_TOKENS"
echo "OUTPUT_TOK_S: $tok"; echo "SPREAD_PCT: 0.1"; echo "MEAN_TPOT_MS: 20"; echo "P99_TPOT_MS: 25"
echo "MEAN_TTFT_MS: 300"; echo "KV_USAGE_PCT: 6.0"; echo "DONE: $RUN_DIR"
"""


def fake_stack(tmp_path):
    """A venv (this python), a vLLM source folder, a kernels overlay and a patch that applies."""
    src = tmp_path / "vllm-src"
    (src / "vllm").mkdir(parents=True)
    (src / "vllm" / "__init__.py").write_text("__version__ = '0.0.test'\n")
    kernels = tmp_path / "overlay" / "vllm_xpu_kernels"
    kernels.mkdir(parents=True)
    (kernels / "__init__.py").write_text("")
    (kernels / "_C.py").write_text("")
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").symlink_to(sys.executable)
    patch = tmp_path / "vllm-x.patch"
    patch.write_text(
        "--- a/vllm/__init__.py\n+++ b/vllm/__init__.py\n@@ -1 +1,2 @@\n"
        " __version__ = '0.0.test'\n+PATCHED = True\n"
    )
    gpu_vars = tmp_path / "gpu_vars.sh"
    gpu_vars.write_text("export FAKE_GPU_VARS_SOURCED=1\n")
    return {
        "--vllm-venv": tmp_path / "venv",
        "--vllm-src": src,
        "--xpu-kernels-overlay": tmp_path / "overlay",
        "--patch": patch,
        "--gpu-vars": gpu_vars,
    }


def cli(*args, **kw):
    return subprocess.run(
        [sys.executable, str(AGENT / "cli.py"), *map(str, args)],
        capture_output=True,
        text=True,
        **kw,
    )


def init_args(ws, model, stack, **over):
    flags = {**stack, **over}
    out = ["init", "--workspace", ws, "--model", model, "--tp", "2", "--tpot-cap-ms", "50"]
    for k, v in flags.items():
        if v is not None:
            out += [k, v]
    return out


def test_init_then_trials_against_fake_serve(tmp_path, monkeypatch):
    model = tmp_path / "moe"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "n_routed_experts": 128,
                "num_experts_per_tok": 6,
                "hybrid_override_pattern": "MEM*EME",
            }
        )
    )
    stack = fake_stack(tmp_path)
    ws = tmp_path / "ws"
    r = cli(
        *init_args(ws, model, stack, **{"--gpus": 4}),
        "--num-prompts",
        "4",
        "--timeout-min",
        "1",
        "--greedy-tokens",
        "128",
        "--lessons",
        tmp_path / "lessons",
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"PATCHES: {stack['--patch']}" in r.stdout and "CARDS: 4" in r.stdout
    facts, claude = (ws / "model_facts.md").read_text(), (ws / "CLAUDE.md").read_text()
    assert "'M' x3" in facts and "| 4 | 1 | 8 | 1, replicated | 32 |" in facts
    assert "serve_trial.py --workspace" in claude and "TP x DP x PP <= 4" in claude
    assert str(stack["--vllm-src"]) in claude and str(stack["--patch"]) in claude
    assert "patches:" not in claude and "patches.md" not in claude
    assert (tmp_path / "lessons" / "serving-moe-4gpu.md").exists()
    conf = yaml.safe_load((ws / "config.yaml").read_text())
    assert conf["stack"]["patches"] == [str(stack["--patch"])]
    assert conf["stack"]["xpu_kernels_overlay"] == str(stack["--xpu-kernels-overlay"])
    assert conf["stack"]["vllm"].startswith("0.0.test from ")
    assert "patches" not in yaml.safe_load((ws / "trials" / "t0.yaml").read_text())
    # A second init must not reuse a workspace that holds trials; checked below once it does.
    fake = tmp_path / "fake_serve.sh"
    fake.write_text(FAKE)
    monkeypatch.setenv("SERVE_SH", str(fake))
    monkeypatch.setenv("GPU_VARS", "/not/the/recorded/one")

    def tool(*args):
        r = subprocess.run(
            [sys.executable, str(AGENT / "serve_trial.py"), "--workspace", str(ws), *args],
            capture_output=True,
            text=True,
        )
        assert r.stdout.rstrip().endswith("DONE"), r.stdout + r.stderr
        return dict(line.split(": ", 1) for line in r.stdout.splitlines() if ": " in line)

    assert tool("baseline", "trials/t0.yaml")["VERDICT"] == "BASELINE"
    (ws / "trials" / "t1.yaml").write_text(
        yaml.safe_dump({"model": str(model), "tp": 2, "env": {"X": "1"}})
    )
    assert (
        tool("run", "trials/t1.yaml", "--parent", "t0", "--strategy", "noop")["VERDICT"] == "NOISE"
    )
    assert tool("finalize")["VERDICT"] == "NO_WIN"
    (ws / "trials" / "t2.yaml").write_text(
        yaml.safe_dump(
            {
                "model": str(model),
                "tp": 2,
                "serve_args": ["--max-num-seqs", "128"],
                "concurrency": 64,
            }
        )
    )
    keys = tool("run", "trials/t2.yaml", "--parent", "t0", "--strategy", "batch")
    assert keys["VERDICT"] == "WIN" and "OUTPUT_TOK_S" in keys
    for bad, tp_args in (({"patches": []}, {}), ({}, {"tp": 4, "serve_args": ["-dp", "2"]})):
        (ws / "trials" / "t3.yaml").write_text(
            yaml.safe_dump({"model": str(model), "tp": 2, **bad, **tp_args})
        )
        assert (
            tool("run", "trials/t3.yaml", "--parent", "t2", "--strategy", "x")["VERDICT"]
            == "INVALID_CONFIG"
        )
    # Every launch ran the workspace's stack, whatever the caller's environment held.
    runs = sorted((ws / "runs").glob("*/serve.out"))
    assert len(runs) == 4
    for out in runs:
        text = out.read_text()
        assert f"ENV_PATCHES: {stack['--patch']}" in text
        assert f"ENV_OVERLAY: {stack['--xpu-kernels-overlay']}" in text
        assert f"ENV_GPU_VARS: {stack['--gpu-vars']}" in text and "ENV_GREEDY_TOKENS: 128" in text
    assert tool("finalize")["VERDICT"] == "FINALIZED"
    best = (ws / "output" / "serve_best.sh").read_text()
    assert "export SERVE_ARGS='--max-num-seqs 128'" in best
    assert f"export PATCHES={stack['--patch']}" in best and "export IN_LEN=1024" in best
    subprocess.run(["bash", "-n"], input=best, text=True, check=True)
    state = json.loads((ws / "trials" / "state.json").read_text())
    assert state["trials"]["t2"]["parent"] == "t0" and len(state["trials"]) == 3
    r = cli(*init_args(ws, model, stack, **{"--gpus": 4}))
    assert r.returncode != 0 and "already holds trials" in r.stderr

    # compare: the best trial's config on this stack against the same stack without patches.
    fake_ab = tmp_path / "fake_ab.sh"
    fake_ab.write_text(FAKE_AB)
    monkeypatch.setenv("SERVE_SH", str(fake_ab))
    r = cli("compare", "--workspace", ws)
    assert r.returncode != 0 and "--without-patches" in r.stderr
    r = cli("compare", "--workspace", ws, "--without-patches", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [ln for ln in r.stdout.splitlines() if ln.startswith("LAUNCH:")] == [
        "LAUNCH: moe other 1",
        "LAUNCH: moe stack 1",
        "LAUNCH: moe other 2",
        "LAUNCH: moe stack 2",
    ]
    spec = yaml.safe_load((ws / "compare" / "t2" / "compare.yaml").read_text())
    other, mine = spec["arms"]["other"]["env"], spec["arms"]["stack"]["env"]
    assert "PATCHES" not in other and "XPU_KERNELS_OVERLAY" not in other
    assert mine["PATCHES"] == str(stack["--patch"])
    assert other["SERVE_ARGS"] == mine["SERVE_ARGS"] == "--max-num-seqs 128"
    r = cli("compare", "--workspace", ws, "--without-patches")
    assert r.returncode == 0 and r.stdout.rstrip().endswith("DONE"), r.stdout + r.stderr
    rows = {x["arm"]: x for x in json.loads((ws / "compare" / "t2" / "summary.json").read_text())}
    assert rows["other"]["verdict"] == "BASELINE" and rows["stack"]["verdict"] == "WIN"
    # --with-defaults: both stacks at t0 as well, all judged against the other stack at t0.
    out = tmp_path / "vs-defaults"
    r = cli("compare", "--workspace", ws, "--without-patches", "--with-defaults", "--out", out)
    assert r.returncode == 0 and r.stdout.rstrip().endswith("DONE"), r.stdout + r.stderr
    spec = yaml.safe_load((out / "compare.yaml").read_text())
    assert list(spec["arms"]) == ["other_t0", "stack_t0", "other", "stack"]
    assert spec["baseline"] == "other_t0"
    t0_env, best_env = spec["arms"]["other_t0"]["env"], spec["arms"]["stack"]["env"]
    assert t0_env["SERVE_ARGS"] == "" and best_env["SERVE_ARGS"] == "--max-num-seqs 128"
    t2 = yaml.safe_load((ws / "trials" / "t2.yaml").read_text())
    assert best_env["CONCURRENCY"] == str(t2["concurrency"]) != t0_env["CONCURRENCY"]
    rows = {x["arm"]: x for x in json.loads((out / "summary.json").read_text())}
    assert rows["other_t0"]["verdict"] == "BASELINE" and rows["stack"]["verdict"] == "WIN"


def test_init_refuses_a_stack_it_cannot_run(tmp_path, monkeypatch):
    stack = fake_stack(tmp_path)
    model, ws = tmp_path / "m", tmp_path / "ws"
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GPU_VARS", raising=False)

    def refused(**over):
        r = cli(*init_args(ws, model, stack, **{"--gpus": 2, **over}))
        assert r.returncode != 0 and not (ws / "config.yaml").exists(), r.stdout
        return r.stderr

    assert "--vllm-venv" in refused(**{"--vllm-venv": None})
    stale = tmp_path / "stale.patch"
    stale.write_text(
        "--- a/vllm/__init__.py\n+++ b/vllm/__init__.py\n@@ -1 +1 @@\n"
        "-__version__ = '0.0.other'\n+__version__ = '0.0.new'\n"
    )
    assert "does not apply" in refused(**{"--patch": stale})
    outside = tmp_path / "csrc.patch"
    outside.write_text("--- a/csrc/x.cpp\n+++ b/csrc/x.cpp\n@@ -0,0 +1 @@\n+x\n")
    assert "not only vllm/" in refused(**{"--patch": outside})
    (tmp_path / "overlay" / "vllm_xpu_kernels" / "_C.py").unlink()
    assert "does not load vllm_xpu_kernels" in refused()
    (tmp_path / "overlay" / "vllm_xpu_kernels" / "_C.py").write_text("")
    # No cards visible and no --gpus: a torch that sees none, first on the path.
    torch = stack["--vllm-src"] / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text(
        "class xpu:\n    is_available = staticmethod(lambda: True)\n"
        "    device_count = staticmethod(lambda: 0)\n"
    )
    assert "--gpus" in refused(**{"--gpus": None})
    # A workspace made for 2 cards does not start a session on a node that has fewer.
    assert cli(*init_args(ws, model, stack, **{"--gpus": 2})).returncode == 0
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "claude").write_text("#!/bin/sh\necho SESSION_STARTED\n")
    (bin_ / "claude").chmod(0o755)
    creds = dict.fromkeys(("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"), "x")
    env = {**os.environ, **creds, "PATH": f"{bin_}:{os.environ['PATH']}"}
    r = cli("launch", "--workspace", ws, env=env)
    assert r.returncode != 0 and "sees 0 card(s)" in r.stderr and "for 2" in r.stderr
    assert "SESSION_STARTED" not in r.stdout


def test_nothing_box_specific_in_the_agent():
    for f in AGENT.iterdir():
        if f.is_file():
            text = f.read_text()
            for word in ("/swtools", "claude_serve_runs", "claude_lessons", "*venv*", "patches.md"):
                assert word not in text, f"{f.name} holds {word}"


def test_scripts_parse():
    for f in ("serve.sh", "build_xpu_kernels.sh"):
        subprocess.run(["bash", "-n", str(AGENT / f)], check=True)
    for f in ("serve.sh", "kernel_gate.sbatch"):
        subprocess.run(["bash", "-n", str(AGENT.parent / "nemotron30b" / f)], check=True)


FAKE_AB = """#!/bin/bash
# Patched arms: +5% tok/s and a new Triton kernel; the wheel arm is another vLLM.
mkdir -p "$RUN_DIR"
python3 -c "import json,sys; p=json.load(open(sys.argv[1])); json.dump([x['answer'][0] for x in p], open(sys.argv[2],'w'))" \\
    "$GREEDY_PROMPTS" "$RUN_DIR/greedy.json"
tok=1000; kernels=base_kernel
# FAKE_GREEDY=salad: the same wrong answers every launch; random: different ones every launch.
[[ -n "$FAKE_GREEDY" ]] && python3 -c "import json,random,sys; p=sys.argv[1]; n=len(json.load(open(p)))
json.dump(['salad' if sys.argv[2] == 'salad' else str(random.random()) for _ in range(n)], open(p, 'w'))" \\
    "$RUN_DIR/greedy.json" "$FAKE_GREEDY"
[[ -n "$PATCHES" ]] && tok=1050 && kernels="base_kernel\\nnew_kernel"
printf "$kernels\\n" >"$RUN_DIR/triton_kernels.txt"
loaded="$VLLM_SRC/vllm"; [[ -n "$PATCHES" ]] && loaded="$RUN_DIR/overlay/vllm"
echo "NODE: n"; echo "VLLM: ${VLLM_VENV:-v}"; echo "VLLM_LOADED_FROM: $loaded"
echo "XPU_KERNELS_FROM: $XPU_KERNELS_OVERLAY/vllm_xpu_kernels"
echo "OUTPUT_TOK_S: $tok"; echo "SPREAD_PCT: 0.1"; echo "MEAN_TPOT_MS: 20"; echo "P99_TPOT_MS: 25"
echo "MEAN_TTFT_MS: 300"; echo "KV_USAGE_PCT: 6.0"; echo "TRITON_KERNELS: 1"; echo "DONE: $RUN_DIR"
"""


def test_serve_ab_proves_arms_and_judges(tmp_path, monkeypatch):
    fake = tmp_path / "fake_ab.sh"
    fake.write_text(FAKE_AB)
    monkeypatch.setenv("SERVE_SH", str(fake))
    src, ovl = tmp_path / "src", tmp_path / "ovl"
    monkeypatch.setenv("AB_SRC", str(src))
    base = {"VLLM_SRC": "${AB_SRC}", "XPU_KERNELS_OVERLAY": str(ovl / "clean")}
    expect = {"vllm_loaded_from": str(src), "xpu_kernels_from": str(ovl / "clean")}
    spec = {
        "tp": 1,
        "models": ["m"],
        "forbid_env": ["FORBIDDEN_KNOB"],
        "baseline": "orig",
        "arms": {
            "orig": {"env": base, "expect": {**expect, "triton_absent": ["^new_"]}},
            "patched": {
                "env": {**base, "PATCHES": "p", "XPU_KERNELS_OVERLAY": str(ovl / "patched")},
                "expect": {
                    "vllm_patched": True,
                    "xpu_kernels_from": str(ovl / "patched"),
                    "triton_present": ["^new_kernel$"],
                },
            },
            "wheel": {"env": {"VLLM_VENV": "w"}, "launches": 1},
        },
    }

    def ab(spec, *extra):
        (tmp_path / "ab.yaml").write_text(yaml.safe_dump(spec))
        r = subprocess.run(
            [
                sys.executable,
                str(AGENT / "serve_ab.py"),
                "--spec",
                str(tmp_path / "ab.yaml"),
                "--out",
                str(tmp_path / "out"),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        assert r.stdout.rstrip().endswith("DONE"), r.stdout + r.stderr
        return r

    order = [ln for ln in ab(spec, "--dry-run").stdout.splitlines() if ln.startswith("LAUNCH:")]
    assert order == [
        "LAUNCH: m orig 1",
        "LAUNCH: m patched 1",
        "LAUNCH: m wheel 1",
        "LAUNCH: m orig 2",
        "LAUNCH: m patched 2",
    ]
    r = ab(spec)
    assert r.returncode == 0
    rows = {x["arm"]: x for x in json.loads((tmp_path / "out" / "summary.json").read_text())}
    assert rows["orig"]["verdict"] == "BASELINE" and rows["patched"]["verdict"] == "WIN"
    assert rows["wheel"]["verdict"] == "REFERENCE" and rows["wheel"]["launches"] == 1
    assert abs(rows["patched"]["delta_pct"] - 5.0) < 1e-9

    # The patched arm serving the clean kernels, or missing its Triton kernel, is not applied.
    spec["arms"]["patched"]["env"]["XPU_KERNELS_OVERLAY"] = str(ovl / "clean")
    r = ab(spec)
    assert (
        r.returncode == 1 and "VERDICT: NOT_APPLIED" in r.stdout and "OUTPUT_TOK_S" not in r.stdout
    )
    spec["arms"]["patched"]["env"]["XPU_KERNELS_OVERLAY"] = str(ovl / "patched")
    del spec["arms"]["patched"]["env"]["PATCHES"]
    r = ab(spec)
    assert r.returncode == 1 and "not this run's patched copy" in r.stdout, r.stdout
    spec["arms"]["patched"]["env"]["PATCHES"] = "p"
    spec["arms"]["orig"]["env"]["FORBIDDEN_KNOB"] = "1"
    assert "VERDICT: INVALID_SPEC" in ab(spec).stdout
    del spec["arms"]["orig"]["env"]["FORBIDDEN_KNOB"]
    spec["arms"]["orig"]["env"]["VLLM_SRC"] = "${AB_UNSET}"
    assert "$AB_UNSET is unset" in ab(spec, "--dry-run").stdout
    spec["arms"]["orig"]["env"]["VLLM_SRC"] = "${AB_SRC}"
    spec["arms"]["patched"]["env"]["XPU_KERNELS_OVERLAY"] = str(ovl / "patched")

    # A baseline that cannot answer the probes, or disagrees with itself, judges nothing.
    for mode, verdict in (("salad", "BASELINE_QUALITY"), ("random", "BASELINE_UNSTABLE")):
        spec["arms"]["orig"]["env"]["FAKE_GREEDY"] = mode
        r = ab(spec)
        assert r.returncode == 1 and f"VERDICT: {verdict}" in r.stdout, r.stdout
        assert "OUTPUT_TOK_S" not in r.stdout and "WIN" not in r.stdout


def test_grade_reads_the_answer_after_reasoning():
    probes = [{"answer": ["Paris"]}, {"answer": ["4"]}]
    assert st.grade(["<think>maybe Lyon</think>\n\nParis.", "4"], probes) == 1.0
    assert st.grade(["<think>Paris? no</think>\nLyon", "<think>4"], probes) == 0.0


def test_overlay_is_reserved_and_forwarded(tmp_path):
    conf = {"models": ["m"], "gpus": 1}
    for name in ("XPU_KERNELS_OVERLAY", "PATCHES", "GPU_VARS"):
        assert name in st.invalid({"model": "m", "env": {name: "/x"}}, conf)
    stack = {
        "vllm_venv": "/v",
        "vllm_src": None,
        "vllm": "x",
        "patches": ["/p1", "/p2"],
        "xpu_kernels_overlay": "/o",
        "gpu_vars": None,
    }
    workload = {"in_len": 1, "out_len": 1, "num_prompts": 1, "concurrency": 8}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **conf,
                **workload,
                "tp": 1,
                "stack": stack,
                "gpu_util": 0.5,
                "max_len": 64,
                "bench_repeats": 3,
                "greedy_tokens": 7,
            }
        )
    )
    env = st.Workspace(tmp_path).env_for({"model": "m", "max_len": 128}, tmp_path / "r")
    assert env["XPU_KERNELS_OVERLAY"] == "/o" and env["PATCHES"] == "/p1 /p2"
    assert (env["GPU_UTIL"], env["MAX_LEN"], env["GREEDY_TOKENS"]) == ("0.5", "128", "7")
    assert "GPU_VARS" not in env
    assert env["DATASET"] == ""


def test_dataset_is_workload_not_a_trial_knob(tmp_path):
    conf = {"models": ["m"], "gpus": 1}
    assert "DATASET" in st.invalid({"model": "m", "env": {"DATASET": "/x"}}, conf)
    stack = {
        "vllm_venv": "/v",
        "vllm_src": None,
        "vllm": "x",
        "patches": [],
        "xpu_kernels_overlay": None,
        "gpu_vars": None,
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **conf,
                "in_len": 1,
                "out_len": 512,
                "num_prompts": 1,
                "concurrency": 1,
                "dataset": "/data/sharegpt.json",
                "tp": 1,
                "stack": stack,
                "gpu_util": 0.5,
                "max_len": 64,
                "bench_repeats": 1,
                "greedy_tokens": 7,
            }
        )
    )
    env = st.Workspace(tmp_path).env_for({"model": "m"}, tmp_path / "r")
    assert (env["DATASET"], env["OUT_LEN"]) == ("/data/sharegpt.json", "512")
