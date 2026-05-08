"""Skill CLI wrappers for xe-forge-skill entry point.

Provides thin CLI access to core modules, used by Claude Code agent
and for standalone ad-hoc testing.

Usage:
    xe-forge-skill analyze <pytorch_file>
    xe-forge-skill validate <kernel_file> [--dsl triton]
    xe-forge-skill benchmark <baseline> <optimized> --spec <spec.yaml> [--baseline-us N]
    xe-forge-skill trial {init|save|result|status|best|baseline-us|finalize} [args]
    xe-forge-skill profile <kernel_file> --spec <spec.yaml> [--warmup 5] [--iters 20]
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="xe-forge-skill",
        description="Xe-Forge skill tools (used by Claude Code and standalone)",
    )
    subparsers = parser.add_subparsers(dest="skill", required=True)

    # -- analyze --
    p_analyze = subparsers.add_parser("analyze", help="AST-based PyTorch kernel analysis")
    p_analyze.add_argument("pytorch_file", help="Path to PyTorch reference file")

    # -- validate --
    p_validate = subparsers.add_parser("validate", help="Static kernel validation")
    p_validate.add_argument("kernel_file", help="Path to kernel file")
    p_validate.add_argument("--dsl", default="triton", choices=["triton", "sycl", "gluon", "cuda"])
    p_validate.add_argument("--stage", default=None, help="Current optimization stage")

    # -- benchmark --
    p_bench = subparsers.add_parser("benchmark", help="Correctness + performance comparison")
    p_bench.add_argument("baseline", help="Path to baseline kernel file")
    p_bench.add_argument("optimized", help="Path to optimized kernel file")
    p_bench.add_argument("--spec", "-s", required=True, help="YAML spec file")
    p_bench.add_argument("--variant", default="bench-gpu", help="Spec variant")
    p_bench.add_argument("--baseline-us", type=float, default=None, help="Cached baseline time")
    p_bench.add_argument("--device", default="xpu", help="Target device")
    p_bench.add_argument("--dsl", default="triton", choices=["triton", "sycl", "gluon", "cuda"])
    p_bench.add_argument("--triton-baseline", action="store_true", help="Baseline is Triton kernel")

    # -- trial --
    p_trial = subparsers.add_parser("trial", help="Trial tree management")
    trial_sub = p_trial.add_subparsers(dest="trial_command", required=True)

    t_init = trial_sub.add_parser("init")
    t_init.add_argument("kernel_name")
    t_init.add_argument("baseline_file")
    t_init.add_argument("--triton-baseline", action="store_true")
    t_init.add_argument("--trials-dir", default="./trials")

    t_save = trial_sub.add_parser("save")
    t_save.add_argument("kernel_name")
    t_save.add_argument("trial_file")
    t_save.add_argument("--parent", default=None)
    t_save.add_argument("--strategy", default="")
    t_save.add_argument("--trials-dir", default="./trials")

    t_result = trial_sub.add_parser("result")
    t_result.add_argument("kernel_name")
    t_result.add_argument("trial_id")
    t_result.add_argument("--validation", choices=["pass", "fail"])
    t_result.add_argument("--correctness", choices=["pass", "fail"])
    t_result.add_argument("--speedup", type=float)
    t_result.add_argument("--baseline-us", type=float)
    t_result.add_argument("--triton-us", type=float)
    t_result.add_argument("--trials-dir", default="./trials")

    t_status = trial_sub.add_parser("status")
    t_status.add_argument("kernel_name")
    t_status.add_argument("--trials-dir", default="./trials")

    t_best = trial_sub.add_parser("best")
    t_best.add_argument("kernel_name")
    t_best.add_argument("--trials-dir", default="./trials")

    t_baseline = trial_sub.add_parser("baseline-us")
    t_baseline.add_argument("kernel_name")
    t_baseline.add_argument("--trials-dir", default="./trials")

    t_finalize = trial_sub.add_parser("finalize")
    t_finalize.add_argument("kernel_name")
    t_finalize.add_argument("output_file")
    t_finalize.add_argument("--trials-dir", default="./trials")

    # -- profile --
    p_profile = subparsers.add_parser("profile", help="VTune GPU profiling")
    p_profile.add_argument("kernel_file", help="Path to kernel file")
    p_profile.add_argument("--spec", "-s", default=None, help="YAML spec file")
    p_profile.add_argument("--variant", default="bench-gpu")
    p_profile.add_argument("--warmup", type=int, default=5)
    p_profile.add_argument("--iters", type=int, default=20)
    p_profile.add_argument("--vtune-bin", default="vtune")

    args = parser.parse_args()

    if args.skill == "analyze":
        from xe_forge.skills.analyze import run
    elif args.skill == "validate":
        from xe_forge.skills.validate import run
    elif args.skill == "benchmark":
        from xe_forge.skills.benchmark import run
    elif args.skill == "trial":
        from xe_forge.skills.trial import run
    elif args.skill == "profile":
        from xe_forge.skills.profile import run

    run(args)


if __name__ == "__main__":
    main()
