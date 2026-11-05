"""``xe-forge generate``: an agent writes a native C engine for a model and makes its hot paths
fast (``native-engine.md.j2``, reading ``knowledge_base/native``); tools measure it.

inspect -> agent -> verify the agent's engine on its surrogate and on the model (the gate).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from xe_forge.frontend.cli import REPO_DIR, SRC_DIR, _agent

KB = REPO_DIR / "knowledge_base" / "native"


def _py(python: str, module: str, *args: str) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH")) if p)
    cmd = [python, "-m", module, *args]
    print(f"generate: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def _cpu() -> str:
    """This node, as lscpu describes it."""
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    keep = ("Model name", "Socket(s)", "Core(s) per socket", "NUMA node(s)", "L2 cache",
            "L3 cache")  # fmt: skip
    facts = [line.split(":", 1) for line in out.splitlines() if ":" in line]
    text = "; ".join(f"{k.strip()}: {v.strip()}" for k, v in facts if k.strip() in keep)
    flags = next((v.split() for k, v in facts if k.strip() == "Flags"), [])
    isa = " ".join(f for f in flags if f.startswith(("avx", "amx", "fma")))
    return f"{text}; ISA: {isa}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="xe-forge generate")
    p.add_argument("--model", required=True, help="hub id or local directory")
    p.add_argument("--framework-python", help="interpreter for the reference and the tools")
    p.add_argument("-o", "--output", type=Path, required=True, help="run directory")
    p.add_argument("--reference-source", type=Path,
                   help="the model's Python when transformers cannot run it")  # fmt: skip
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--numa", type=int, default=0)
    p.add_argument("--new", type=int, default=16, help="greedy tokens verified")
    p.add_argument("--prefill", default="512,2048,8192,32768", help="prompt lengths profiled")
    p.add_argument("--depths", default="0,8192,32768", help="contexts decode is profiled at")
    p.add_argument("--amx-tokens", type=int, default=64,
                   help="prompt at which AMX must run and is verified")  # fmt: skip
    p.add_argument("--agent-max-turns", type=int, default=500)
    a = p.parse_args(argv)
    python = a.framework_python or sys.executable
    cpu = _cpu()
    if "amx_tile" not in cpu:
        print(f"generate: AMX is required and this CPU has none: {cpu}", file=sys.stderr)
        return 1
    run_dir = a.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not (run_dir / "model.json").is_file() and _py(
        python, "xe_forge.frontend.surrogate", "inspect", "--model", a.model,
        "-o", str(run_dir / "model.json"),
    ) != 0:  # fmt: skip
        return 1
    info = json.loads((run_dir / "model.json").read_text())

    if not (run_dir / "engine.json").is_file():
        context = {
            "model": info["model"], "model_dir": info["dir"], "modeling": info.get("modeling"),
            "transformers": info.get("transformers"),
            "reference_source": str(a.reference_source) if a.reference_source else None,
            "kb": str(KB), "cpu": cpu, "threads": a.threads, "numa": a.numa,
            "amx_tokens": a.amx_tokens,
            "prefill": [int(x) for x in a.prefill.split(",")],
            "depths": [int(x) for x in a.depths.split(",")],
            "tool": f"PYTHONPATH={SRC_DIR} {python} -m",
        }  # fmt: skip
        if _agent("engine", "native-engine.md.j2", context, run_dir, "engine.json", python,
                  a.agent_max_turns) is None:  # fmt: skip
            return 1
    answer = json.loads((run_dir / "engine.json").read_text())

    # the gate: the tools' verdict on the agent's engine, not the agent's
    results = {}
    reference = answer.get("reference")
    long_prompt = ["--prompt-tokens", str(a.amx_tokens)]
    for name, model_dir, extra in (
        ("surrogate", answer.get("surrogate"), []),
        ("model", info["dir"], []),
        ("surrogate-amx", answer.get("surrogate"), long_prompt),
        ("model-amx", info["dir"], [*long_prompt, "--require-amx"]),
    ):
        if not model_dir:
            continue
        out = run_dir / f"gate-{name}.json"
        out.unlink(missing_ok=True)
        args = ["--model-dir", str(model_dir), "--engine", answer["engine"], "--new", str(a.new),
                "--threads", str(a.threads), "--numa", str(a.numa), "-o", str(out), *extra]  # fmt: skip
        if reference:
            args += ["--reference", reference]
        _py(python, "xe_forge.frontend.verify", *args)
        results[name] = json.loads(out.read_text()) if out.is_file() else {"verdict": "FAIL"}
    lines = [f"# {info['model']}: native engine", "", f"engine: `{answer['engine']}`", ""]
    for name, r in results.items():
        lines.append(f"- {name}: {r['verdict']} -- " + "; ".join(
            f"{g['level']} {'PASS' if g['pass'] else 'REJECT'}" for g in r.get("gates", [])))  # fmt: skip
    if answer.get("timing"):
        lines += ["", f"timing (agent's measurement): {json.dumps(answer['timing'])}"]
    if answer.get("amx"):
        lines += ["", f"amx (agent's report): {json.dumps(answer['amx'])}"]
    if answer.get("notes"):
        lines += ["", answer["notes"]]
    (run_dir / "GENERATE.md").write_text("\n".join(lines) + "\n")
    print((run_dir / "GENERATE.md").read_text())
    return 0 if results and all(r["verdict"] == "PASS" for r in results.values()) else 1
