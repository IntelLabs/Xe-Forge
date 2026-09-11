"""xe-forge-skill validate: Static kernel validation.

Like ``benchmark``, this delegates to a host-supplied command when one is
configured (``--external-validate``, ``EXTERNAL_VALIDATE``, or
``external.validate``). Xe-Forge's own validator is an AST pass over the source:
useful for Triton, where the mistakes it knows about are Triton's, and close to
silent on a compiled DSL, where "does it compile for this part, with these
extensions, against these libraries" is the question and only a real build
answers it.
"""

import sys

#: What counts as a source when a trial is a directory rather than a single file.
SOURCE_SUFFIXES = (".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".sycl")


def _external_template(args) -> str | None:
    explicit = getattr(args, "external_validate", None)
    if explicit:
        return explicit
    from xe_forge.config import get_config

    return get_config().external.validate


def _run_external(args, template: str) -> int:
    from pathlib import Path

    from xe_forge.config import get_config
    from xe_forge.external import ExternalCommandError, run_external

    kernel_file = Path(args.kernel_file)
    try:
        result = run_external(
            template,
            timeout=get_config().external.timeout,
            kernel=kernel_file.stem,
            trial=str(kernel_file.resolve()),
            kernel_file=str(kernel_file.resolve()),
            dsl=args.dsl or "",
            stage=args.stage or "",
            workspace=str(Path.cwd()),
        )
    except ExternalCommandError as exc:
        print(f"VERDICT: EXTERNAL_ERROR\nError: {exc}")
        return 1

    # Passed through whole: for a compiled DSL the useful part of a failed
    # validation is the compiler's own diagnostic, and summarising it would
    # throw away the line number. The host's own VERDICT line is already in it.
    print(result.raw.rstrip())
    return result.returncode


def run(args):
    from pathlib import Path

    template = _external_template(args)
    if template:
        sys.exit(_run_external(args, template))

    from xe_forge.core.validator import KernelValidator, format_issues

    # A kernel may be one file or a directory of them. The AST pass reads one source at
    # a time, so a directory is every source under it, each reported under its own path:
    # an issue attributed to "the kernel" when the kernel is four files is not something
    # an editor can act on.
    path = Path(args.kernel_file)
    if path.is_dir():
        # By extension, because a directory may legitimately hold things that are not
        # source -- a tuning table, a captured tensor, a note -- and failing on those
        # would make a multi-file trial unvalidatable for a reason that is not about the
        # kernel. Named here rather than inferred: guessing from content would quietly
        # skip a source the validator should have read.
        sources = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES)
        if not sources:
            print(f"{path}: no source files to validate (looked for {', '.join(SOURCE_SUFFIXES)})")
            sys.exit(1)
    else:
        sources = [path]

    validator = KernelValidator()
    errors = 0
    for source in sources:
        try:
            code = source.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            # Not a finding about the kernel: say what could not be read rather than
            # counting an unreadable file as clean.
            print(f"{source}: could not be read ({exc})")
            errors += 1
            continue
        issues = validator.validate(code, dsl=args.dsl, stage=args.stage)
        if len(sources) > 1:
            print(f"--- {source.relative_to(path)}")
        print(format_issues(issues))
        errors += sum(1 for i in issues if i.severity == "error")

    sys.exit(1 if errors else 0)
