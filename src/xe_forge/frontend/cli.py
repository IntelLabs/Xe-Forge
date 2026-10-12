"""``xe-forge capture | analyze | optimize | run``.

    xe-forge capture  --framework vllm --model <id> -o run/capture.json \\
                      [--framework-python <venv>/bin/python] [adapter flags]
    xe-forge analyze  run/capture.json [--top 5] [--min-gain 0.005] [-o run/]
    xe-forge optimize run/kernels.yaml [--engine claude] [--only NAME] [--all-variants]
    xe-forge run      --framework vllm --model <id> -o run/ [all of the above] [--stack-check] [--prove]

``optimize`` runs, per enabled (kernel, variant), the same command a hand-written manifest
gets from ``scripts/gen_kernel_sbatch.py``: ``python -m xe_forge.cli --name --kernel-repo
--spec --variant`` with no reference, the baseline copy being the oracle. Anything after
``--`` is passed to every one of those runs unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import xe_forge

COMMANDS = ("capture", "analyze", "optimize", "run", "compare", "calibrate")
SRC_DIR = Path(xe_forge.__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent


# ── capture ────────────────────────────────────────────────────────────────


def capture(argv: list[str], framework_python: str | None) -> int:
    """Run the capture under the framework's interpreter, this package on its path."""
    python = framework_python or sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH")) if p)
    cmd = [python, "-m", "xe_forge.frontend.capture", *argv]
    print(f"capture: {shlex.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def _agent(step: str, template: str, context: dict, cwd: Path, answer: str, python, turns):
    """One agent step; None when it gave no answer (the reason is printed)."""
    from xe_forge.frontend import agent

    try:
        return agent.run(template, context, cwd, answer, python or sys.executable, turns)
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        return None


def fits(capture_args: list[str], python: str | None, run_dir: Path, max_turns: int) -> int:
    """An agent judges whether the model fits before anything is spent on it. Only a
    definite "no" stops the run; no answer is not a no."""
    context = {"args": shlex.join(capture_args)}
    answer = _agent("fit", "frontend-fit.md.j2", context, run_dir / "fit", "fit.json", python,
                    max_turns)  # fmt: skip
    if answer is None:
        return 0
    if answer.get("fits") is False:
        print(f"fit: does not fit -- {answer.get('reason', '')}", file=sys.stderr)
        return 3
    print(f"fit: {answer.get('need_gb')} of {answer.get('have_gb')} GB -- {answer.get('reason')}")
    return 0


def prepare(capture_args: list[str], python: str | None, run_dir: Path, max_turns: int) -> int:
    """An adapter whose framework needs the model prepared first (OpenVINO: an exported IR)
    names the task, and an agent does it -- formats, calibration, what to install where."""
    from xe_forge.frontend import adapters

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--framework")
    framework = pre.parse_known_args(capture_args)[0].framework
    adapter = adapters.get(framework) if framework else None
    if not hasattr(adapter, "prepare"):
        return 0
    p = argparse.ArgumentParser(add_help=False)
    adapter.add_arguments(p)
    task = adapter.prepare(p.parse_known_args(capture_args)[0])
    if task is None:
        return 0
    out, template, context = task
    print(f"prepare: an agent prepares the model -> {out}", flush=True)
    done = _agent("prepare", template, context, run_dir / "prepare", "prepared.json", python,
                  max_turns)  # fmt: skip
    return 0 if done is not None else 1


def name(
    capture_path: Path, run_dir: Path, python, max_turns: int, repos: Path, coverage: float
) -> Path | None:
    """The naming agent: where each costly op is implemented, and which argument is what
    (``frontend-name.md.j2``). Writes ``<run>/names.json``; None when it gave no answer."""
    from xe_forge.frontend import ir

    cap = ir.load(capture_path)
    work = run_dir / "name"
    work.mkdir(parents=True, exist_ok=True)
    total = sum(op.device_us for op in cap.ops) or 1.0
    by_key: dict[str, list] = {}
    for op in cap.ops:
        by_key.setdefault(op.framework_meta.get("name_key", op.framework_op), []).append(op)
    ranked = sorted(by_key.items(), key=lambda kv: -sum(o.device_us for o in kv[1]))
    listing, covered = {}, 0.0
    for key, ops in ranked:
        if covered >= coverage:
            break
        us = sum(o.device_us for o in ops)
        covered += us / total
        top = max(ops, key=lambda o: o.device_us)
        arg_names = top.framework_meta.get("arg_names") or []
        listing[key] = {
            "framework_op": top.framework_op,
            "owner": top.framework_meta.get("owner"),
            "share": round(us / total, 4),
            "kernels": dict(sorted(top.kernel_symbols.items(), key=lambda kv: -kv[1])[:5]),
            "args": [
                {
                    "name": arg_names[i] if i < len(arg_names) else f"arg{i}",
                    **(
                        {"shape": a.shape, "dtype": a.dtype}
                        if a
                        else {"value": top.scalars[i] if i < len(top.scalars) else None}
                    ),
                }
                for i, a in enumerate(top.args)
            ],
            "call_site": top.call_site,
        }
    (work / "ops.json").write_text(json.dumps(listing, indent=1) + "\n")
    model = cap.run.get("model") or {}
    (work / "config.json").write_text(json.dumps(model.get("config") or {}, indent=1) + "\n")
    context = {
        "model": model.get("name"),
        "framework": cap.run.get("framework"),
        "coverage": f"{covered:.0%}",
        "repos": repos,
    }
    answer = _agent("name", "frontend-name.md.j2", context, work, "names.json", python, max_turns)
    if answer is None:
        return None
    path = run_dir / "names.json"
    path.write_text(json.dumps(answer, indent=1) + "\n")
    return path


def _split_framework_python(argv: list[str]) -> tuple[list[str], str | None]:
    out, python, it = [], None, iter(argv)
    for a in it:
        if a == "--framework-python":
            python = next(it, None)
        elif a.startswith("--framework-python="):
            python = a.split("=", 1)[1]
        else:
            out.append(a)
    return out, python


# ── analyze ────────────────────────────────────────────────────────────────


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    from xe_forge.frontend import analyze as an
    from xe_forge.frontend import manifest

    p.add_argument("--top", type=int, default=3, help="kernels to optimize")
    p.add_argument("--min-gain", type=float, default=0.005, help="smallest share worth it")
    p.add_argument(
        "--at-roofline", type=float, default=an.AT_ROOFLINE,
        help="efficiency of its calibrated bound at which an op counts as done",
    )  # fmt: skip
    p.add_argument(
        "--coverage", type=float, default=manifest.COVERAGE,
        help="share of an op's time its benchmark variants must cover",
    )  # fmt: skip
    p.add_argument("--max-variants", type=int, default=manifest.MAX_VARIANTS)
    p.add_argument("--partition", help="Slurm partition recorded in kernels.yaml, if any")
    p.add_argument("--calibration", type=Path, help="xe-forge calibrate output")


def analyze(capture_path: Path, out_dir: Path, a: argparse.Namespace) -> Path:
    from xe_forge.frontend import analyze as an
    from xe_forge.frontend import ir, manifest

    cap = ir.load(capture_path)
    calib = json.loads(a.calibration.read_text()) if a.calibration else None
    result = an.analyze(
        cap, top=a.top, min_gain=a.min_gain, calibration=calib, at_roofline=a.at_roofline
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result.as_dict(), indent=1) + "\n")
    (out_dir / "gates.log").write_text(an.gates_log(result))
    (out_dir / "report.md").write_text(an.report(result, cap))
    path = manifest.emit(
        result, cap, capture_path, out_dir, partition=a.partition,
        coverage=a.coverage, max_variants=a.max_variants,
    )  # fmt: skip
    enabled = [e for e in result.entries if e.enabled]
    print(
        f"analyze: {len(cap.workloads)} workloads -> {len(result.entries)} kernels, "
        f"{len(enabled)} enabled ({', '.join(e.name for e in enabled) or 'none'}) -> {path}"
    )
    return path


# ── optimize ───────────────────────────────────────────────────────────────


def xe_forge_command(entry: dict, variant: str, repo: Path, workspace: Path, opts) -> list[str]:
    cmd = [
        sys.executable, "-m", "xe_forge.cli",
        "--engine", opts.engine,
        "--auto-launch",
        "--dsl", entry["dsl"],
        "--device", opts.device,
        "--workspace", str(workspace),
        "--max-turns", str(opts.max_turns),
        "--name", entry["name"],
        "--kernel-repo", str(repo),
        "--spec", entry["spec"],
        "--variant", variant,
    ]  # fmt: skip
    if entry.get("record"):
        cmd += ["--dataset-record", entry["record"]]
    if entry.get("target") == "library":
        cmd += ["--target", "library"]
    if getattr(opts, "stack_python", None):  # the interpreter the stack check builds over
        cmd += ["--stack-python", opts.stack_python]
    if entry["dsl"] == "sycl" and opts.compiler_flags:
        cmd += ["--compiler-flags", opts.compiler_flags.replace("{kernel_repo}", str(repo))]
    return cmd + list(opts.passthrough)


def optimize(manifest_path: Path, run_dir: Path, opts) -> list[tuple[str, str]]:
    """One Xe-Forge run per enabled (kernel, variant), in sequence. Returns the pairs."""
    from xe_forge.frontend import manifest

    data = manifest.load(manifest_path)
    repos_dir = Path(opts.kernel_repos).resolve() if opts.kernel_repos else REPO_DIR.parent
    ws_root = run_dir / "workspaces"
    ws_root.mkdir(parents=True, exist_ok=True)
    if opts.engine == "claude" and shutil.which("claude") is None:
        raise SystemExit("optimize: --engine claude needs the Claude Code CLI ('claude') on PATH")

    pairs = []
    for entry in data["kernels"]:
        key = manifest.key(entry)
        if not entry.get("enabled") or (opts.only and not {entry["name"], key} & set(opts.only)):
            continue
        variants = entry["variants"] if opts.all_variants else entry["variants"][:1]
        for variant in variants:
            pairs.append((key, variant))
            workspace = ws_root / f"{key}__{variant}"
            status_path = workspace.with_suffix(".status.json")
            repo = manifest.resolve_repo(entry["repo"], repos_dir)
            if not repo.is_dir():
                status = {"skipped": f"kernel repo not found: {repo}"}
                status_path.write_text(json.dumps(status, indent=1) + "\n")
                print(f"optimize: {entry['name']} ({variant}): {status['skipped']}")
                continue
            cmd = xe_forge_command(entry, variant, repo, workspace, opts)
            log = workspace.with_suffix(".log")
            print(
                f"optimize: {entry['name']} ({variant}) -> {workspace}\n  {shlex.join(cmd)}",
                flush=True,
            )
            started = time.time()
            with open(log, "w") as f:
                code = subprocess.call(cmd, cwd=REPO_DIR, stdout=f, stderr=subprocess.STDOUT)
            status = {
                "exit_code": code,
                "seconds": round(time.time() - started, 1),
                "command": shlex.join(cmd),
                "log": str(log),
            }
            status_path.write_text(json.dumps(status, indent=1) + "\n")
            print(
                f"optimize: {entry['name']} ({variant}) exit {code} in {status['seconds']}s; log {log}"
            )
    return pairs


def _add_optimize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", default="claude", choices=("claude", "dspy-agent"))
    p.add_argument("--device", default="xpu")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--only", nargs="*", help="kernel names to run (default: every enabled)")
    p.add_argument("--all-variants", action="store_true", help="run every listed variant")
    p.add_argument(
        "--kernel-repos",
        help="directory holding the kernel repositories (default: beside this checkout)",
    )
    p.add_argument(
        "--compiler-flags",
        default=os.environ.get("XE_FORGE_SYCL_FLAGS"),
        help="SYCL compile flags for sycl entries; {kernel_repo} expands (env XE_FORGE_SYCL_FLAGS)",
    )


# ── entry ──────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"usage: xe-forge {{{','.join(COMMANDS)}}} ...")
    command, rest = argv[0], argv[1:]
    passthrough: list[str] = []
    if "--" in rest:
        i = rest.index("--")
        rest, passthrough = rest[:i], rest[i + 1 :]

    if command == "capture":
        rest, python = _split_framework_python(rest)
        return capture(rest, python)

    if command == "calibrate":
        from xe_forge.frontend import calibrate

        p = argparse.ArgumentParser(prog="xe-forge calibrate")
        p.add_argument("-o", "--output", type=Path, help="default: the per-device cache")
        p.add_argument("--read-bytes", type=int, default=1 << 30, help="> the card's caches")
        p.add_argument("--matmul-n", type=int, default=8192, help="square matmul size")
        p.add_argument("--iters", type=int, default=20)
        a = p.parse_args(rest)
        return calibrate.main(a.output, a.read_bytes, a.matmul_n, a.iters)

    if command == "compare":
        from xe_forge.frontend import compare

        p = argparse.ArgumentParser(prog="xe-forge compare")
        p.add_argument("--a", nargs="+", type=Path, required=True, help="baseline capture.json(s)")
        p.add_argument("--b", nargs="+", type=Path, required=True, help="candidate capture.json(s)")
        p.add_argument("-o", "--output", type=Path, required=True, help="directory")
        a = p.parse_args(rest)
        rows = compare.compare(a.a, a.b)
        path = compare.write(rows, a.output.resolve())
        t = rows[0]
        print(
            f"compare: total {t['us_a']} -> {t['us_b']} us (x{t['speedup']}); {len(rows) - 1} workloads -> {path}"
        )
        return 0

    if command == "analyze":
        p = argparse.ArgumentParser(prog="xe-forge analyze")
        p.add_argument("capture", type=Path)
        p.add_argument("-o", "--output", type=Path, help="directory (default: the capture's)")
        _add_analyze_args(p)
        a = p.parse_args(rest)
        analyze(a.capture.resolve(), (a.output or a.capture.parent).resolve(), a)
        return 0

    if command == "optimize":
        from xe_forge.frontend import collect, manifest

        p = argparse.ArgumentParser(prog="xe-forge optimize")
        p.add_argument("manifest", type=Path)
        _add_optimize_args(p)
        a = p.parse_args(rest)
        a.passthrough = passthrough
        run_dir = a.manifest.resolve().parent
        pairs = optimize(a.manifest.resolve(), run_dir, a)
        summary = collect.collect(run_dir, manifest.load(a.manifest), pairs)
        print(summary.read_text())
        return 0

    # run: fit -> [prepare ->] capture -> name -> analyze -> optimize -> collect [-> prove]
    from xe_forge.frontend import collect, manifest

    rest, python = _split_framework_python(rest)
    p = argparse.ArgumentParser(prog="xe-forge run", allow_abbrev=False)
    p.add_argument("-o", "--output", type=Path, required=True, help="run directory")
    _add_analyze_args(p)
    p.add_argument(
        "--name-coverage", type=float, default=0.99,
        help="share of device time whose ops the naming agent places",
    )  # fmt: skip
    p.add_argument("--analyze-only", action="store_true", help="stop after the manifest")
    p.add_argument(
        "--prove", action="store_true", help="build the stack with the winning patches and "
        "measure them in the model (prove.py)"
    )  # fmt: skip
    p.add_argument("--prove-repeat", type=int, default=3, help="captures per arm")
    p.add_argument(
        "--stack-check", action="store_true", help="each port-back agent builds and tests "
        "its port in the framework's stack (Xe-Forge --stack-python)"
    )  # fmt: skip
    _add_optimize_args(p)
    a, capture_args = p.parse_known_args(rest)
    a.passthrough = passthrough
    run_dir = a.output.resolve()
    capture_path = run_dir / "capture.json"
    if not capture_path.is_file():
        code = (
            fits(capture_args, python, run_dir, a.max_turns)
            or prepare(capture_args, python, run_dir, a.max_turns)
            or capture([*capture_args, "-o", str(capture_path)], python)
        )
        if code != 0:
            print(f"run: stopped before analysis (exit {code})", file=sys.stderr)
            return code
    else:
        print(f"run: reusing {capture_path}")
    names = run_dir / "names.json"
    if not names.is_file():
        repos = Path(a.kernel_repos).resolve() if a.kernel_repos else REPO_DIR.parent
        if name(capture_path, run_dir, python, a.max_turns, repos, a.name_coverage) is None:
            print("run: no names.json; only ATen ops are named", file=sys.stderr)
    if names.is_file():
        # Rebuild the capture with the agent's names; later captures (prove) carry them too.
        capture_args = [*capture_args, "--names", str(names)]
        reparse = [
            "--reparse",
            str(run_dir / "raw"),
            "--names",
            str(names),
            "-o",
            str(capture_path),
        ]
        if capture(reparse, python) != 0:
            print("run: re-reading the capture with names.json failed", file=sys.stderr)
            return 1
    path = analyze(capture_path, run_dir, a)
    if a.analyze_only:
        return 0
    a.stack_python = (python or sys.executable) if a.stack_check else None
    pairs = optimize(path, run_dir, a)
    summary = collect.collect(run_dir, manifest.load(path), pairs)
    if a.prove:
        from xe_forge.frontend import prove

        section = prove.prove(run_dir, capture, capture_args, python or sys.executable,
                              a.prove_repeat, a.max_turns)  # fmt: skip
        summary.write_text(summary.read_text() + "\n" + section)
    print(summary.read_text())
    return 0
