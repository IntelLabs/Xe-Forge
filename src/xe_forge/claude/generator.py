"""Generate a Claude Code workspace for kernel optimization.

Creates CLAUDE.md, config.yaml, .claude/commands/, .claude/agents/,
and copies kernel files into the workspace. All text artifacts are
rendered from Jinja templates under ``templates/``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xe_forge.config import Config
from xe_forge.core.dataset_record import load_dataset_record
from xe_forge.core.lessons import load_lessons_log
from xe_forge.core.spec_loader import load_spec

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Durable for the whole session, like dataset.profile_path: the kernel-locator agent
# writes it once, and every later step reads it instead of re-exploring the repo.
KERNEL_LOCATOR_PROFILE_PATH = "experiments/kernel_profile.md"

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
    ext = _kernel_ext(dsl)
    # What the session is told its numbers come from. The commands it runs are the
    # same either way -- the delegation happens inside xe-forge-skill -- but the
    # verdicts it may see, and so the rules it has to follow, are not.
    measurement = "host" if config.external.benchmark else "builtin"
    # The data behind the spec's variants, when the host attached a record of it.
    # None: no dataset record attached, so the templates omit every workload-data section.
    dataset = load_dataset_record(config.external.dataset_record)
    # Where this session records what it measures, and reads what earlier ones did.
    # Deliberately outside the workspace, which is scratch.
    lessons = load_lessons_log(config.external.lessons, kernel_name)
    if lessons is not None:
        # Seeded with its own format so entries stay comparable across sessions.
        # Exclusive create: a second session must not truncate the first's entries.
        try:
            with open(lessons.own_file, "x") as f:
                f.write(_render("lessons.md.j2", kernel_name=kernel_name, dsl=dsl, device=device))
        except FileExistsError:
            pass
    # Absolute: the session's cwd is the workspace.
    kernel_repo = (
        str(Path(config.external.kernel_repo).expanduser().resolve())
        if config.external.kernel_repo
        else None
    )

    spec_has_inputs = bool(spec_path and load_spec(spec_path).inputs)
    # Without a PyTorch reference, the baseline -- a copy of the repo's kernel that also
    # builds the inputs the repo's own test builds -- is what every trial must agree with.
    baseline_oracle = (
        measurement == "builtin"
        and bool(kernel_repo)
        and not reference_code
        and not spec_has_inputs
    )
    reference_file = (
        f"test_kernels/{kernel_name}{ext}"
        if baseline_oracle
        else f"test_kernels/{kernel_name}_pytorch.py"
    )
    variant_args = ["--variant", variant_type] if spec_path and variant_type else []

    benchmark_args = ["--device", str(device), "--dsl", str(dsl)]
    if spec_path:
        benchmark_args.extend(["--spec", f"test_kernels/{kernel_name}.yaml"])
    if measurement == "builtin":
        benchmark_args.append("--builtin-benchmark")
        if reference_code or baseline_oracle:
            benchmark_args.extend(["--reference", reference_file])
    benchmark_args.extend(variant_args)
    benchmark_options = shlex.join(benchmark_args)
    # A spec without inputs gives the reference its dims; one with inputs builds its own.
    reference_workload = (
        measurement == "builtin" and bool(reference_code) and not spec_has_inputs
    ) or baseline_oracle
    profile_args = []
    if reference_workload:
        profile_args.extend(["--reference", reference_file])
    if spec_path:
        profile_args.extend(["--spec", f"test_kernels/{kernel_name}.yaml"])
    profile_args.extend(variant_args)
    if config.profiler.unitrace_enabled:
        profile_args.extend(["--unitrace-bin", config.profiler.unitrace_bin])
        profile_args.extend(["--metric-group", config.profiler.unitrace_metric_group])
    require_profiles = (
        config.profiler.unitrace_enabled and config.profiler.unitrace_metric_group == "all"
    )
    if require_profiles:
        profile_args.extend(["--kernel-name", kernel_name])
    if config.profiler.vtune_enabled:
        profile_args.extend(["--vtune-bin", config.profiler.vtune_bin])
    if config.profiler.unitrace_enabled != config.profiler.vtune_enabled:
        profile_args.extend(["--tool", "unitrace" if config.profiler.unitrace_enabled else "vtune"])
    profile_options = shlex.join(profile_args)

    compiler_flags = config.engine.compiler_flags
    # Parsed the way a shell would, so a quoted flag (-DNAME='a b') reaches the
    # compiler as one argument.
    try:
        compiler_flag_list = shlex.split(compiler_flags) if compiler_flags else []
    except ValueError as exc:
        raise ValueError(f"cannot parse compiler_flags {compiler_flags!r}: {exc}") from None
    # With a kernel repo the baseline is a copy of the repo's kernel, never a from-scratch seed.
    if (
        not kernel_code
        and dsl == "sycl"
        and measurement == "builtin"
        and reference_code
        and not kernel_repo
    ):
        kernel_code = _render(
            "sycl_torch_extension_seed.py.j2",
            kernel_name=kernel_name,
            compiler_flag_list=compiler_flag_list,
        )
    (workspace / "CLAUDE.md").write_text(
        _render(
            "CLAUDE.md.j2",
            dsl=dsl,
            device=device,
            kernel_name=kernel_name,
            ext=ext,
            measurement=measurement,
            benchmark_options=benchmark_options,
            profile_options=profile_options,
            require_profiles=require_profiles,
            reference_workload=reference_workload,
            has_spec=bool(spec_path),
            variant=variant_type,
            baseline_oracle=baseline_oracle,
            dataset=dataset,
            lessons=lessons,
            compiler_flags=compiler_flags,
            kernel_repo=kernel_repo,
            kernel_locator_profile_path=KERNEL_LOCATOR_PROFILE_PATH,
        )
    )
    (workspace / "config.yaml").write_text(
        _render(
            "config.yaml.j2",
            max_trials=config.trial.max_trials,
            dsl=dsl,
            device=device,
            measurement=measurement,
            vtune_enabled=config.profiler.vtune_enabled,
            vtune_bin=config.profiler.vtune_bin,
            unitrace_enabled=config.profiler.unitrace_enabled,
            unitrace_bin=config.profiler.unitrace_bin,
            compiler_flags=compiler_flags,
        )
    )

    cmd_dir = workspace / ".claude" / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (cmd_dir / "optimize-kernel.md").write_text(
        _render(
            "optimize-kernel.md.j2",
            dsl=dsl,
            require_profiles=require_profiles,
            measurement=measurement,
            benchmark_options=benchmark_options,
            dataset=dataset,
            lessons=lessons,
            kernel_repo=kernel_repo,
            kernel_locator_profile_path=KERNEL_LOCATOR_PROFILE_PATH,
        )
    )

    agent_dir = workspace / ".claude" / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "tool-runner.md").write_text(
        _render(
            "tool-runner.md.j2",
            measurement=measurement,
            benchmark_options=benchmark_options,
            reference_workload=reference_workload,
            profile_options=profile_options,
        )
    )
    if dataset is not None:
        # Only when there is something to inspect. An agent offered a dataset that
        # is not there would be a step the session runs, fails, and reports.
        (agent_dir / "workload-inspector.md").write_text(
            _render("workload-inspector.md.j2", dataset=dataset, kernel_name=kernel_name)
        )
    if kernel_repo:
        # Only when there is a repo to explore -- same reasoning as workload-inspector.
        (agent_dir / "kernel-locator.md").write_text(
            _render(
                "kernel-locator.md.j2",
                kernel_name=kernel_name,
                kernel_repo=kernel_repo,
                profile_path=KERNEL_LOCATOR_PROFILE_PATH,
                baseline_oracle=baseline_oracle,
            )
        )
        (agent_dir / "port-back.md").write_text(
            _render(
                "port-back.md.j2",
                kernel_name=kernel_name,
                kernel_repo=kernel_repo,
                profile_path=KERNEL_LOCATOR_PROFILE_PATH,
                ext=ext,
            )
        )

    _write_kernel_files(workspace, kernel_name, kernel_code, reference_code, spec_path, ext)
    _symlink_knowledge_base(workspace)

    if config.engine.git_init:
        _git_init(workspace)


def _kernel_ext(dsl: str) -> str:
    """Suffix for a kernel written in *dsl*."""
    from xe_forge.models import DSL

    try:
        return DSL(str(dsl)).kernel_ext
    except ValueError:
        return ".py"


def _write_kernel_files(
    workspace: Path,
    kernel_name: str,
    kernel_code: str,
    reference_code: str,
    spec_path: str | None,
    ext: str = ".py",
) -> None:
    tk_dir = workspace / "test_kernels"
    tk_dir.mkdir(parents=True, exist_ok=True)

    # Empty means from-scratch: Claude creates this file itself as its first action.
    if kernel_code:
        (tk_dir / f"{kernel_name}{ext}").write_text(kernel_code)
    # The reference stays .py whatever the kernel is written in: it is PyTorch,
    # and it is what `analyze` can actually read.
    if reference_code:
        (tk_dir / f"{kernel_name}_pytorch.py").write_text(reference_code)
    if spec_path and Path(spec_path).exists():
        # A host that generates the workspace layout itself -- writing the kernel, the
        # reference and the spec into `test_kernels/` and then naming those paths on the
        # command line -- hands us a source that is already the destination. That is the
        # arrangement working as intended, not an error to raise on.
        dest = tk_dir / f"{kernel_name}.yaml"
        if Path(spec_path).resolve() != dest.resolve():
            shutil.copy2(spec_path, dest)


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
