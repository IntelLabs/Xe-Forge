"""Generate a Claude Code workspace for kernel optimization.

Creates CLAUDE.md, config.yaml, .claude/commands/, .claude/agents/,
.claude/settings.json (MCP servers), knowledge_base/ markdown files,
and copies kernel files into the workspace. All text artifacts are
rendered from Jinja templates under ``templates/``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xe_forge.config import Config

logger = logging.getLogger(__name__)

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

    # Materialize KB markdown BEFORE rendering CLAUDE.md so @imports are valid
    kb_materialized = False
    if config.knowledge.materialize_markdown:
        kb_materialized = _materialize_knowledge_base(workspace, str(dsl), str(device))

    (workspace / "CLAUDE.md").write_text(
        _render(
            "CLAUDE.md.j2",
            dsl=dsl,
            device=device,
            kernel_name=kernel_name,
            kb_materialized=kb_materialized,
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
    (cmd_dir / "optimize-kernel.md").write_text(_render("optimize-kernel.md.j2", dsl=dsl))

    agent_dir = workspace / ".claude" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "tool-runner.md").write_text(_render("tool-runner.md.j2"))

    _write_kernel_files(workspace, kernel_name, kernel_code, reference_code, spec_path)
    _write_settings_json(workspace, config)

    if config.engine.git_init:
        _git_init(workspace)


def _materialize_knowledge_base(workspace: Path, dsl: str, device: str) -> bool:
    """Write KB topic markdown files to workspace/knowledge_base/.

    Uses intel_kernel_kb's MarkdownExporter to write per-topic .md files
    (constraints.md, patterns.md, examples.md) that CLAUDE.md.j2 can
    @import. Falls back to the legacy knowledge_base symlink approach when
    intel_kernel_kb is not installed.

    Returns True when markdown files were written, False on fallback.
    """
    try:
        from intel_kernel_kb.exporters.base import ExportConfig
        from intel_kernel_kb.exporters.markdown import MarkdownExporter
        from intel_kernel_kb.loader import KBLoader, LoaderConfig
    except ImportError:
        logger.warning("intel-kernel-kb not installed — falling back to knowledge_base symlink")
        _symlink_knowledge_base(workspace)
        return False

    kb_dir = workspace / "knowledge_base"
    kb_dir.mkdir(exist_ok=True)

    loader_config = LoaderConfig(
        language=dsl,
        device=device,
        include_tooling=False,
        include_examples=True,
        allow_v0_migration=True,
    )
    try:
        catalog = KBLoader().load(loader_config)
    except Exception as exc:
        logger.warning("intel-kernel-kb catalog load failed: %s — using symlink fallback", exc)
        _symlink_knowledge_base(workspace)
        return False

    export_config = ExportConfig(
        language=[dsl],
        split_files=True,
        include_tooling=False,
        include_examples=True,
    )
    try:
        files: dict[str, str] = MarkdownExporter().export_files(catalog, export_config)
    except Exception as exc:
        logger.warning("intel-kernel-kb export failed: %s — using symlink fallback", exc)
        _symlink_knowledge_base(workspace)
        return False

    written = []
    for fname, content in files.items():
        if fname == "CLAUDE.md":
            # We have our own CLAUDE.md.j2 template; skip the ikb-generated index.
            # Topic files are @imported directly from knowledge_base/ in CLAUDE.md.j2.
            continue
        dest = kb_dir / fname
        dest.write_text(content, encoding="utf-8")
        written.append(fname)

    if written:
        logger.info("KB materialized to %s: %s", kb_dir, ", ".join(written))
        return True

    logger.warning("intel-kernel-kb produced no topic files — using symlink fallback")
    _symlink_knowledge_base(workspace)
    return False


def _write_settings_json(workspace: Path, config: Config) -> None:
    """Write .claude/settings.json with MCP server registrations.

    Claude Code reads this file when the workspace is first opened so MCP
    tools are available from the very first message — no manual setup needed.
    """
    mcp = config.mcp
    servers: dict = {}

    if mcp.intel_perf_enabled:
        servers["intel-perf"] = {
            "command": "intel-perf-mcp",
            "type": "stdio",
        }

    if mcp.intel_profiler_mcp_enabled:
        servers["intel-profiler"] = {
            "command": "intel-profiler-mcp",
            "type": "stdio",
        }

    servers.update(mcp.extra_servers)

    settings: dict = {"mcpServers": servers}

    settings_dir = workspace / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Wrote .claude/settings.json with %d MCP server(s): %s",
        len(servers),
        list(servers.keys()),
    )


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
    """Create a symlink to the installed knowledge_base directory (legacy fallback)."""
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
