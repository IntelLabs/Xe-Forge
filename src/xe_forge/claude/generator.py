"""Generate a Claude Code workspace for kernel optimization.

Creates CLAUDE.md, config.yaml, .claude/commands/, .claude/agents/,
and copies kernel files into the workspace. All text artifacts are
rendered from Jinja templates under ``templates/``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xe_forge.config import Config

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


def _render(template_name: str, **context: object) -> str:
    return _env.get_template(template_name).render(**context)


def _read_template_raw(template_name: str) -> str:
    """Read a template file verbatim, bypassing Jinja.

    Used for the C++ starter kernel: its initializer lists contain ``{{ }}``
    which would collide with Jinja's variable delimiters. The starter kernel
    needs no substitution, so raw text is exactly right.
    """
    return (_TEMPLATES_DIR / template_name).read_text()


# DSLs whose kernel source is C++ (.cpp) rather than Python (.py).
_CPP_DSLS = ("sycl", "cuda")


def _kernel_ext(dsl: str) -> str:
    return ".cpp" if dsl in _CPP_DSLS else ".py"


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

    # Select DSL-specific templates. SYCL gets its own CLAUDE.md / command
    # (C++ workflow, file-IO contract, no PyTorch-AST analyze step); other DSLs
    # keep the existing Triton/PyTorch templates.
    is_sycl = dsl == "sycl"
    claude_template = "CLAUDE.sycl.md.j2" if is_sycl else "CLAUDE.md.j2"
    optimize_template = "optimize-kernel.sycl.md.j2" if is_sycl else "optimize-kernel.md.j2"
    ext = _kernel_ext(dsl)

    (workspace / "CLAUDE.md").write_text(
        _render(
            claude_template,
            dsl=dsl,
            device=device,
            kernel_name=kernel_name,
            kernel_ext=ext,
        )
    )
    (workspace / "config.yaml").write_text(
        _render(
            "config.yaml.j2",
            max_trials=config.trial.max_trials,
            vtune_enabled=config.profiler.vtune_enabled,
            vtune_bin=config.profiler.vtune_bin,
        )
    )

    cmd_dir = workspace / ".claude" / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (cmd_dir / "optimize-kernel.md").write_text(_render(optimize_template, dsl=dsl, kernel_ext=ext))

    agent_dir = workspace / ".claude" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "tool-runner.md").write_text(_render("tool-runner.md.j2"))

    _write_kernel_files(workspace, kernel_name, kernel_code, reference_code, spec_path, dsl)
    _symlink_knowledge_base(workspace)

    if config.engine.git_init:
        _git_init(workspace)


def _write_kernel_files(
    workspace: Path,
    kernel_name: str,
    kernel_code: str,
    reference_code: str,
    spec_path: str | None,
    dsl: str = "triton",
) -> None:
    tk_dir = workspace / "test_kernels"
    tk_dir.mkdir(parents=True, exist_ok=True)

    ext = _kernel_ext(dsl)

    if dsl in _CPP_DSLS:
        # If the input is PyTorch-only or spec-only (no C++ source), substitute a
        # compilable starter stub honouring the file-IO contract so the t0
        # baseline produces a real timing.
        if "#include" not in kernel_code:
            kernel_code = _read_template_raw("starter_kernel.sycl.cpp.j2")

    (tk_dir / f"{kernel_name}{ext}").write_text(kernel_code)

    # The reference is PyTorch even when the kernel is C++; always write it as
    # {name}_pytorch.py so the benchmark skill can locate the golden reference.
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
    """Initialize workspace as a git repo. Opt-in via EngineConfig.git_init."""
    if (workspace / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(workspace), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial workspace", "--allow-empty"],
        cwd=str(workspace),
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "xe-forge",
            "GIT_AUTHOR_EMAIL": "xe-forge@local",
            "GIT_COMMITTER_NAME": "xe-forge",
            "GIT_COMMITTER_EMAIL": "xe-forge@local",
        },
    )
