"""The vISA lowering agent: a headless Claude Code session in a sealed workspace.

The workspace holds only what the model may see -- ``INSTRUCTIONS.md``,
``contract.yaml``, ``knowledge.md`` (empty in mode A) -- plus two commands
(:mod:`xe_forge.lowering.visa.verify_cli`): ``visa-verify``, the oracle, and
``abi-probe``, the ground truth about the payload for a launch configuration.
Every decision beyond those two is the model's. Compiler artefacts (the ABI stub's
dumps, the reference kernel's Triton cache, the target's own IGC output) live
outside the workspace.

The session runs with ``--restricted``: file tools are confined to the workspace,
no tool that runs code is available except Bash, and Bash may run nothing but
``./visa-verify`` and ``./abi-probe`` (``--permission-prompts none`` denies
everything else). After
the session, the transcript is checked by the artifact guard as well.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

INSTRUCTIONS = """# Task: implement one Triton kernel directly in Intel vISA

You are the compiler. Produce a complete Intel vISA kernel (`.visaasm` text) for an Intel Xe2 GPU
that computes exactly what the Triton kernel in `contract.yaml` computes. The vISA finalizer
allocates registers and encodes instructions: write virtual-register vISA, never physical registers.

## What is fixed, and what is yours

Fixed: the kernel's name and parameters (the host passes them as in the Triton launch) and the
outputs: for every input, your kernel must write the same results as the Triton kernel, within
the contract's precision. That is all that is checked.

Everything else is your decision: the algorithm, the work distribution, the launch geometry
(number of work-groups, sub-groups per work-group, SIMD width), shared local memory and barriers,
loop structure, memory access pattern. The Triton kernel shows *what* to compute; its launch
(in `contract.yaml`) is only a default and may be a poor fit for this GPU. Be as aggressive as you
like.

To launch differently, put one line at the top of your kernel:

    // @launch num_warps=<1..32, power of 2> threads_per_warp=<16|32> slm_bytes=<bytes> grid=(<expr>, <expr>, <expr>)

Every field is optional (omitted fields keep the default). `grid` expressions may use the
kernel's scalar parameters and constexprs by name, `grid0/grid1/grid2` (the Triton launch's own
grid), integers, `+ - * // %`, and `cdiv`, `min`, `max`. A work-group is `num_warps *
threads_per_warp` work-items. `slm_bytes > 0` gives each work-group that much shared local memory
and enables barriers.

## Your tools

- `./abi-probe [num_warps=N] [threads_per_warp=16|32] [slm_bytes=B]` -- ground truth: for that
  launch configuration, prints how the runtime delivers data to your kernel: the compiler's
  `.ze_info` payload table (argument index, offset, size, address space) and the `.input` /
  `.kernel_attr` lines of an empty kernel with this signature. Working out which payload slot is
  which parameter, and declaring the matching `.decl`/`.input` lines and kernel attributes in
  your kernel, is your job. Run it whenever you change the launch configuration.
- `./visa-verify <file>` -- the oracle: finalizes your kernel, runs it against the Triton kernel on
  many input sizes and prints structured feedback ending in `VERDICT:` and `ATTEMPTS_LEFT:`.
  `finalizer_error` quotes the finalizer's message and line; `runtime_error`/`timeout` mean it
  built but failed or hung; `incorrect` gives the failing case and the first wrong values;
  `ABI_MISMATCH` (not counted as an attempt) means your `.input` lines do not match the payload
  for your launch configuration. You have {budget} counted attempts.

`knowledge.md` holds vISA documentation and examples selected for this kernel{knowledge_note}.
Everything you need is in this directory; do not look for other files. {stop_rule}
"""

STOP_CORRECT = "Stop as soon as the verdict is CORRECT, and reply with a one-paragraph summary."
STOP_OPTIMIZE = (
    "Once the verdict is CORRECT, make the kernel faster while keeping it correct (fewer and "
    "wider memory messages, loads before uses, fewer live registers; `performance` in the "
    "feedback compares against the Triton kernel), then stop with a one-paragraph summary."
)


@dataclass
class SessionResult:
    returncode: int
    log_path: Path
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    num_turns: int | None = None
    is_error: bool = False
    error: str = ""


def claude_env(api_base: str | None, api_key: str | None, model: str | None) -> dict[str, str]:
    """Environment for the CLI, mapped from Xe-Forge's LLM config (as the Claude engine does)."""
    env = os.environ.copy()
    if api_base:
        env.setdefault("ANTHROPIC_BASE_URL", api_base)
    if api_key:
        env.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
    if model:
        env.setdefault("ANTHROPIC_MODEL", model.split("/")[-1])
    env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    return env


def write_workspace(
    workspace: Path,
    *,
    contract: str,
    knowledge: str,
    state_path: Path,
    budget: int,
    optimize: bool,
    python: str = sys.executable,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "contract.yaml").write_text(contract)
    (workspace / "knowledge.md").write_text(knowledge or "(no vISA knowledge is provided in this run)\n")
    note = "" if knowledge else " (empty in this run: rely on the contract)"
    (workspace / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(
        knowledge_note=note, budget=budget, stop_rule=STOP_OPTIMIZE if optimize else STOP_CORRECT))
    for name, sub, what in (("visa-verify", "verify", "the oracle"), ("abi-probe", "probe", "ground truth")):
        script = workspace / name
        script.write_text(
            f"#!/bin/sh\n# {what}; see INSTRUCTIONS.md.\n"
            f'exec "{python}" -m xe_forge.lowering.visa.verify_cli {sub} "{state_path}" "$@"\n'
        )
        script.chmod(0o755)


def launch(
    workspace: Path,
    env: dict[str, str],
    *,
    max_turns: int,
    timeout_s: float,
    log_name: str = "claude-session.jsonl",
) -> SessionResult:
    """Run the headless session to completion; return usage from its final result message."""
    claude = shutil.which("claude")
    log_path = workspace.parent / log_name
    if not claude:
        return SessionResult(127, log_path, is_error=True, error="'claude' CLI not found in PATH")
    prompt = (
        "Read INSTRUCTIONS.md, contract.yaml and knowledge.md, then implement the kernel in vISA "
        "in kernel.visaasm, using ./abi-probe and ./visa-verify as the instructions say."
    )
    cmd = [
        claude, "-p", prompt,
        "--restricted",
        "--tools", "Read", "Write", "Edit", "Glob", "Grep", "Bash",
        "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep",
        "Bash(./visa-verify:*)", "Bash(./visa-verify *)", "Bash(./abi-probe:*)", "Bash(./abi-probe *)",
        "--permission-mode", "acceptEdits",
        "--permission-prompts", "none",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--max-turns", str(max_turns),
        "--output-format", "stream-json", "--verbose",
    ]
    result = SessionResult(0, log_path)
    try:
        with open(log_path, "w", buffering=1) as log:
            proc = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                _note(line)
                if line.startswith("{") and '"type":"result"' in line.replace(" ", ""):
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    result.usage = msg.get("usage") or {}
                    result.cost_usd = msg.get("total_cost_usd")
                    result.num_turns = msg.get("num_turns")
                    result.is_error = bool(msg.get("is_error"))
                    if result.is_error:
                        result.error = str(msg.get("result", ""))[:500]
            result.returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        result.returncode, result.is_error, result.error = -9, True, f"session exceeded {timeout_s:.0f}s"
    except OSError as e:
        result.returncode, result.is_error, result.error = 1, True, f"could not run claude: {e}"
    return result


def _note(line: str) -> None:
    """Echo the verify verdicts as they happen, so a watched run shows progress."""
    if "VERDICT:" in line:
        for part in line.split("\\n"):
            if "VERDICT:" in part:
                print("  " + part.strip().strip('"'), flush=True)
                break
