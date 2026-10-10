"""``xe-forge capture | analyze | optimize | run``.

    xe-forge capture  --framework vllm --model <id> -o run/capture.json \\
                      [--framework-python <venv>/bin/python] [adapter flags]
    xe-forge analyze  run/capture.json [--top 5] [--min-gain 0.005] [-o run/]
    xe-forge optimize run/kernels.yaml [--engine claude] [--only NAME] [--all-variants]
    xe-forge run      --framework vllm --model <id> -o run/ [all of the above]

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

COMMANDS = ("capture", "analyze", "optimize", "run")
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


def analyze(capture_path: Path, out_dir: Path, top: int, min_gain: float, partition: str) -> Path:
    from xe_forge.frontend import analyze as an
    from xe_forge.frontend import ir, manifest

    cap = ir.load(capture_path)
    result = an.analyze(cap, top=top, min_gain=min_gain)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result.as_dict(), indent=1) + "\n")
    (out_dir / "gates.log").write_text(an.gates_log(result))
    (out_dir / "report.md").write_text(an.report(result, cap))
    path = manifest.emit(result, cap, capture_path, out_dir, partition=partition)
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
        if not entry.get("enabled") or (opts.only and entry["name"] not in opts.only):
            continue
        variants = entry["variants"] if opts.all_variants else entry["variants"][:1]
        for variant in variants:
            pairs.append((entry["name"], variant))
            workspace = ws_root / f"{entry['name']}__{variant}"
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

    if command == "analyze":
        p = argparse.ArgumentParser(prog="xe-forge analyze")
        p.add_argument("capture", type=Path)
        p.add_argument("-o", "--output", type=Path, help="directory (default: the capture's)")
        p.add_argument("--top", type=int, default=5)
        p.add_argument("--min-gain", type=float, default=0.005)
        p.add_argument("--partition", default="b70")
        a = p.parse_args(rest)
        analyze(
            a.capture.resolve(),
            (a.output or a.capture.parent).resolve(),
            a.top,
            a.min_gain,
            a.partition,
        )
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

    # run: capture -> analyze -> optimize -> collect
    from xe_forge.frontend import collect, manifest

    rest, python = _split_framework_python(rest)
    p = argparse.ArgumentParser(prog="xe-forge run", allow_abbrev=False)
    p.add_argument("-o", "--output", type=Path, required=True, help="run directory")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.005)
    p.add_argument("--partition", default="b70")
    p.add_argument("--analyze-only", action="store_true", help="stop after the manifest")
    _add_optimize_args(p)
    a, capture_args = p.parse_known_args(rest)
    a.passthrough = passthrough
    run_dir = a.output.resolve()
    capture_path = run_dir / "capture.json"
    if not capture_path.is_file():
        code = capture([*capture_args, "-o", str(capture_path)], python)
        if code != 0:
            print(f"run: capture failed (exit {code}); nothing to analyze", file=sys.stderr)
            return code
    else:
        print(f"run: reusing {capture_path}")
    path = analyze(capture_path, run_dir, a.top, a.min_gain, a.partition)
    if a.analyze_only:
        return 0
    pairs = optimize(path, run_dir, a)
    summary = collect.collect(run_dir, manifest.load(path), pairs)
    print(summary.read_text())
    return 0
