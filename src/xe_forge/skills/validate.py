"""xe-forge-skill validate: Static kernel validation."""

import sys


def run(args):
    from pathlib import Path

    from xe_forge.core.validator import KernelValidator, format_issues

    code = Path(args.kernel_file).read_text()
    validator = KernelValidator()
    issues = validator.validate(code, dsl=args.dsl, stage=args.stage)
    print(format_issues(issues))

    errors = [i for i in issues if i.severity == "error"]
    sys.exit(1 if errors else 0)
