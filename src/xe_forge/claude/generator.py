"""Generate a Claude Code workspace for interactive kernel optimization.

Creates CLAUDE.md, config.yaml, .claude/commands/, .claude/agents/,
and copies kernel files into the workspace.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from xe_forge.config import Config


def generate_workspace(
    workspace: Path,
    config: Config,
    kernel_name: str,
    kernel_code: str,
    reference_code: str = "",
    spec_path: str | None = None,
    variant_type: str = "bench-gpu",
    target_dtype: str | None = None,
) -> None:
    """Generate a complete Claude Code workspace."""
    workspace.mkdir(parents=True, exist_ok=True)

    dsl = config.device_config.dsl
    device = config.device_config.device
    max_trials = config.trial.max_trials
    vtune_enabled = config.profiler.vtune_enabled
    vtune_bin = config.profiler.vtune_bin

    _write_claude_md(workspace, dsl, device, max_trials, vtune_enabled, vtune_bin, kernel_name)
    _write_config_yaml(workspace, max_trials, vtune_enabled, vtune_bin)
    _write_claude_commands(workspace, dsl)
    _write_claude_agents(workspace)
    _write_kernel_files(workspace, kernel_name, kernel_code, reference_code, spec_path)
    _symlink_knowledge_base(workspace)
    _git_init(workspace)


def _write_claude_md(
    workspace: Path,
    dsl: str,
    device: str,
    max_trials: int,
    vtune_enabled: bool,
    vtune_bin: str,
    kernel_name: str,
) -> None:
    profile_tool_row = ""
    if vtune_enabled:
        profile_tool_row = (
            """| **Profile** | `xe-forge-skill profile <triton_file> --spec <spec.yaml>` |"""
        )

    profile_step = ""
    next_step = 6
    if vtune_enabled:
        profile_step = f"""{next_step}. **Profile (after t1, when vtune_enabled)** — Run `xe-forge-skill profile <triton_file> --spec <spec.yaml>`. Use output to guide next trial."""
        next_step = 7

    content = f"""# Xe-Forge Kernel Optimizer

Optimize kernels into high-performance {dsl.upper()} implementations for {device.upper()}.

## CONFIGURATION — Read `config.yaml` first

At the start of every session, read `config.yaml` in the workspace root. It controls:
- **`max_trials`** — hard cap on optimization trials ({max_trials})
- **`vtune_enabled`** — {"enabled" if vtune_enabled else "disabled"}
- **`vtune_bin`** — path to VTune binary

## RULES — NEVER VIOLATE

1. **ONLY create** kernel files (trial files `t<N>.py` or output files).
2. **NEVER create** benchmark scripts, test scripts, helper utilities, or any other files.
3. **NEVER write custom scripts** to measure performance — ONLY use `xe-forge-skill benchmark`.
4. If a tool fails, **STOP and report the error**. Do NOT work around it with custom scripts.
5. Generated kernels must be **self-contained** — all helper functions inline.
6. You **MUST run all {max_trials} trials**. Do NOT stop early due to plateau. Only stop early if speedup > 5x.

## MANDATORY TOOLS — Use these and ONLY these

**Delegate all tool execution** to the `tool-runner` agent to keep the main context clean.

**CRITICAL — Single-XPU serialization**: There is only ONE XPU. NEVER dispatch multiple tool-runner agents in parallel if any runs `benchmark` or `profile`. These GPU workloads must execute strictly one at a time.

| Task | Command |
|------|---------|
| **Analyze** | `xe-forge-skill analyze <pytorch_file>` |
| **Validate** | `xe-forge-skill validate <kernel_file> --dsl {dsl}` |
| **Benchmark** | `xe-forge-skill benchmark <baseline> <optimized> --spec <spec.yaml> [--baseline-us <cached>]` |
| **Init trials** | `xe-forge-skill trial init <kernel_name> <baseline_file>` |
| **Save trial** | `xe-forge-skill trial save <kernel_name> <file> [--parent <id>] [--strategy "..."]` |
| **Record result** | `xe-forge-skill trial result <kernel_name> <trial_id> --correctness <pass\\|fail> --speedup <float> --baseline-us <float> --triton-us <float>` |
| **Check status** | `xe-forge-skill trial status <kernel_name>` |
| **Best trial** | `xe-forge-skill trial best <kernel_name>` |
| **Baseline time** | `xe-forge-skill trial baseline-us <kernel_name>` |
| **Finalize** | `xe-forge-skill trial finalize <kernel_name> <output_file>` |
{profile_tool_row}

## WORKFLOW — Follow these steps in order

### Step 1: Analyze
- Read the baseline source file `test_kernels/{kernel_name}.py`. Identify shapes, dtypes, operations.
- Run `xe-forge-skill analyze test_kernels/{kernel_name}.py` to get AST-based analysis.
- Read relevant knowledge_base/ files for optimization patterns.

### Step 2: Initialize
```bash
xe-forge-skill trial init {kernel_name} test_kernels/{kernel_name}.py
```

### Step 3: Trial Loop (always run all {max_trials} trials)
For each trial:
1. **Write kernel** — start from template or modify previous trial.
2. **Validate** — `xe-forge-skill validate <file> --dsl {dsl}` (fix until passing).
3. **Save** — `xe-forge-skill trial save {kernel_name} <file> --parent <id> --strategy "description"`.
4. **Benchmark** (MANDATORY every trial):
   - **Trial t0:** `xe-forge-skill benchmark test_kernels/{kernel_name}.py <triton_file> --spec test_kernels/{kernel_name}.yaml`
   - **Trials t1+:** Get cached baseline via `xe-forge-skill trial baseline-us {kernel_name}`, then `xe-forge-skill benchmark test_kernels/{kernel_name}.py <triton_file> --spec test_kernels/{kernel_name}.yaml --baseline-us <cached>`
5. **Record** — `xe-forge-skill trial result {kernel_name} <trial_id> --correctness <pass|fail> --speedup <float> --baseline-us <float> --triton-us <float>`
{profile_step}
{next_step}. **Decide next action**:
   - Speedup > 5x -> stop, finalize
   - Speedup improved -> continue on this branch
   - Speedup regressed -> branch back to best trial, try different strategy
   - Correctness failed -> fix on same branch
   - Plateau -> try fundamentally different approach

### Step 4: Finalize
```bash
xe-forge-skill trial finalize {kernel_name} output/{kernel_name}_optimized.py
```

## CRITICAL CORRECTNESS CONSTRAINTS

- NO default values for `@triton.autotune` meta-parameters in kernel signature
- Use 1D grid when applying tile swizzling (GROUP_SIZE_M)
- `boundary_check` uses dimension indices (0, 1), not booleans
- Cast batch indices to `int64` before stride multiplication
- Model class must be compatible with ai-bench (nn.Module with get_inputs/get_init_inputs)
"""
    (workspace / "CLAUDE.md").write_text(content)


def _write_config_yaml(
    workspace: Path,
    max_trials: int,
    vtune_enabled: bool,
    vtune_bin: str,
) -> None:
    content = f"""max_trials: {max_trials}
vtune_enabled: {"true" if vtune_enabled else "false"}
vtune_bin: "{vtune_bin}"
"""
    (workspace / "config.yaml").write_text(content)


def _write_claude_commands(workspace: Path, dsl: str) -> None:
    cmd_dir = workspace / ".claude" / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    content = f"""Optimize `$ARGUMENTS` into a {dsl.upper()} kernel.

Resolve the argument to a file in `test_kernels/` — it could be a full filename, partial name, or just a number. Glob to find the match; if ambiguous, ask.

First read `config.yaml` for session settings (max_trials, vtune_enabled).
Follow the CLAUDE.md workflow exactly — every step, in order:

1. Read the baseline kernel file in `test_kernels/` to understand the operations.
2. Run `xe-forge-skill analyze test_kernels/<name>.py`.
3. Read relevant knowledge_base/ files.
4. Initialize: `xe-forge-skill trial init <kernel_name> test_kernels/<name>.py`
5. Run ALL max_trials trials (from config.yaml). For EACH trial you MUST:
   - Validate with `xe-forge-skill validate`
   - Save with `xe-forge-skill trial save`
   - Benchmark with `xe-forge-skill benchmark` — NEVER create custom test scripts
   - Record with `xe-forge-skill trial result`
6. Finalize the best correct trial

CRITICAL: ONLY create kernel files — NO benchmark scripts, NO test scripts, NO helpers.
"""
    (cmd_dir / "optimize-kernel.md").write_text(content)


def _write_claude_agents(workspace: Path) -> None:
    agent_dir = workspace / ".claude" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    content = """---
name: tool-runner
description: "Delegate ALL mandatory tool execution to this agent: xe-forge-skill commands. Use this for every tool invocation to keep the parent context clean."
tools: Bash, Glob, Grep, Read
model: inherit
color: orange
---

You are a precise tool execution agent for the kernel optimizer. Your sole job is to run xe-forge-skill commands and return their output cleanly.

## The Tools You Handle

1. **Analyze**: `xe-forge-skill analyze <pytorch_file>`
2. **Validate**: `xe-forge-skill validate <kernel_file> --dsl <dsl>`
3. **Benchmark**: `xe-forge-skill benchmark <baseline> <optimized> --spec <spec.yaml> [--baseline-us <cached>]`
4. **Trial Manager**: `xe-forge-skill trial <subcommand> <args...>`
5. **Profiler**: `xe-forge-skill profile <kernel_file> --spec <spec.yaml>`

## CRITICAL: Single-XPU Constraint

There is only ONE XPU. **Benchmark** and **Profile** both run GPU workloads. They must run strictly one at a time, never concurrently.

Safe to parallelize: analyze, validate, trial (CPU-only).

## Execution Rules

1. Run exactly the command specified. Do not modify arguments.
2. If the command fails, report the full error. Do NOT attempt to fix or retry.
3. Do NOT create, modify, or delete any files.

## Output Rules

For benchmark:
- Extract ONLY: Correctness (PASSED/FAILED), Performance (baseline_us, triton_us, speedup), Errors.
- Do NOT include configuration header or decorative separators.

For profile:
- Summarize into: Hotspot Kernels, Key Metrics, Bottleneck, Recommendations, Raw Numbers.

## Response Format

```
**Tool**: <tool name>
**Command**: <exact command run>
**Result**:
<output or summary>
```
"""
    (agent_dir / "tool-runner.md").write_text(content)


def _write_kernel_files(
    workspace: Path,
    kernel_name: str,
    kernel_code: str,
    reference_code: str,
    spec_path: str | None,
) -> None:
    tk_dir = workspace / "test_kernels"
    tk_dir.mkdir(parents=True, exist_ok=True)

    (tk_dir / f"{kernel_name}.py").write_text(kernel_code)
    if reference_code:
        (tk_dir / f"{kernel_name}_pytorch.py").write_text(reference_code)
    if spec_path and Path(spec_path).exists():
        shutil.copy2(spec_path, tk_dir / f"{kernel_name}.yaml")


def _symlink_knowledge_base(workspace: Path) -> None:
    """Create a symlink to the installed knowledge_base directory."""
    kb_link = workspace / "knowledge_base"
    if kb_link.exists() or kb_link.is_symlink():
        return

    import xe_forge

    pkg_dir = Path(xe_forge.__file__).parent
    candidates = [
        pkg_dir.parent.parent / "knowledge_base",
        pkg_dir.parent / "knowledge_base",
        Path("./knowledge_base"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            kb_link.symlink_to(candidate.resolve())
            return


def _git_init(workspace: Path) -> None:
    """Initialize workspace as a git repo so Claude Code can operate."""
    if (workspace / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(workspace), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial workspace", "--allow-empty"],
        cwd=str(workspace),
        capture_output=True,
        env={
            **__import__("os").environ,
            "GIT_AUTHOR_NAME": "xe-forge",
            "GIT_AUTHOR_EMAIL": "xe-forge@local",
            "GIT_COMMITTER_NAME": "xe-forge",
            "GIT_COMMITTER_EMAIL": "xe-forge@local",
        },
    )
