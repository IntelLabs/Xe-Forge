"""Skill CLI wrappers for xe-forge-skill entry point.

Provides thin CLI access to core modules, used by Claude Code agent
and for standalone ad-hoc testing.

Usage:
    xe-forge-skill analyze <pytorch_file>
    xe-forge-skill validate <kernel_file|kernel_dir> [--dsl triton]
    xe-forge-skill benchmark <baseline> <optimized> --spec <spec.yaml> [--builtin-benchmark] [--baseline-us N] [--kernel-name K --trial-id T]
    xe-forge-skill trial {init|save|result|status|best|baseline-us|finalize} [args]
    xe-forge-skill upstream {clone <repo> [--name N]|patch <kernel_name> [--kind K] [--notes TEXT]}
    xe-forge-skill profile <kernel_file> --spec <spec.yaml> [--tool auto|unitrace|vtune] [--warmup 5] [--iters 20]
"""

import argparse
import os


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
    p_validate.add_argument(
        "kernel_file", help="Path to the kernel: one source file, or a directory of them"
    )
    p_validate.add_argument(
        "--dsl", default="triton", choices=["triton", "sycl", "gluon", "cuda", "opencl"]
    )
    p_validate.add_argument("--stage", default=None, help="Current optimization stage")
    p_validate.add_argument(
        "--external-validate",
        default=None,
        help="Host command to validate with instead (overrides EXTERNAL_VALIDATE)",
    )

    # -- benchmark --
    p_bench = subparsers.add_parser("benchmark", help="Correctness + performance comparison")
    p_bench.add_argument("baseline", help="Path to baseline kernel file")
    p_bench.add_argument("optimized", help="Path to optimized kernel file")
    p_bench.add_argument("--spec", "-s", default=None, help="YAML spec for shape-based inputs")
    p_bench.add_argument(
        "--reference",
        default=None,
        help="Semantic reference; without --spec, owns get_inputs() and get_init_inputs()",
    )
    p_bench.add_argument(
        "--variant", default=None, help="Spec variant (defaults to the spec's default_variant)"
    )
    p_bench.add_argument("--baseline-us", type=float, default=None, help="Cached baseline time")
    p_bench.add_argument("--device", default="xpu", help="Target device")
    p_bench.add_argument(
        "--dsl", default="triton", choices=["triton", "sycl", "gluon", "cuda", "opencl"]
    )
    p_bench.add_argument("--triton-baseline", action="store_true", help="Baseline is Triton kernel")
    p_bench.add_argument(
        "--external-benchmark",
        default=None,
        help="Host command to benchmark with instead (overrides EXTERNAL_BENCHMARK)",
    )
    p_bench.add_argument("--kernel-name", default=None, help="Trial tree to record the result in")
    p_bench.add_argument(
        "--trial-id", default=None, help="Saved trial the optimized file is; records the result"
    )
    p_bench.add_argument("--trials-dir", default="./trials")
    p_bench.add_argument(
        "--builtin-benchmark",
        action="store_true",
        help=(
            "Measure with the built-in executor using a spec or reference input factories. "
            "Required when no host benchmark command is configured "
            "(or XE_FORGE_BUILTIN_BENCHMARK=1)"
        ),
    )

    # -- trial --
    p_trial = subparsers.add_parser("trial", help="Trial tree management")
    trial_sub = p_trial.add_subparsers(dest="trial_command", required=True)

    t_init = trial_sub.add_parser("init")
    t_init.add_argument("kernel_name")
    t_init.add_argument("baseline_file")
    t_init.add_argument("--triton-baseline", action="store_true")
    t_init.add_argument(
        "--require-profiles",
        nargs="+",
        choices=["ComputeBasic", "EuStallSampling", "VTune"],
        help="Profile groups every correct trial must attempt before the next save or finalize",
    )
    t_init.add_argument("--trials-dir", default="./trials")

    t_save = trial_sub.add_parser("save")
    t_save.add_argument("kernel_name")
    t_save.add_argument(
        "trial_file",
        help="The trial: one source file, or a directory holding all of its sources",
    )
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
    # Neutral name; the Triton spelling stays an alias because SYCL, CUDA and
    # Gluon trials record the same field. The on-disk key is unchanged.
    t_result.add_argument("--kernel-us", "--triton-us", type=float, dest="kernel_us")
    # The gate a host named when it withheld `--speedup`. Recorded so the tree says why
    # a trial carries no ratio; a trial with both times and no ratio ranks at parity
    # whether or not the gate was named, so this is a record, not a control.
    t_result.add_argument("--verdict")
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
    t_finalize.add_argument(
        "--require-measured",
        action="store_true",
        help="Only consider trials whose results `benchmark --trial-id` recorded",
    )
    t_finalize.add_argument("--trials-dir", default="./trials")

    # -- profile --
    p_profile = subparsers.add_parser("profile", help="GPU hardware counter profiling")
    p_profile.add_argument("kernel_file", help="Path to kernel file")
    p_profile.add_argument(
        "--spec",
        "-s",
        default=None,
        help="YAML spec file; with --reference, its variant supplies dims, dtype and tolerances",
    )
    p_profile.add_argument(
        "--reference",
        default=None,
        help="Immutable reference defining input factories and state validation",
    )
    p_profile.add_argument("--variant", default=None)
    p_profile.add_argument("--warmup", type=int, default=5)
    p_profile.add_argument("--iters", type=int, default=20)
    p_profile.add_argument(
        "--tool",
        choices=["auto", "unitrace", "vtune"],
        default="auto",
        help="Profiler backend; auto tries unitrace ComputeBasic first, then VTune. "
        "Hardware counters require device/driver support and permissions.",
    )
    p_profile.add_argument("--unitrace-bin", default="unitrace")
    p_profile.add_argument(
        "--metric-group",
        choices=["ComputeBasic", "EuStallSampling", "all"],
        default="ComputeBasic",
        help="Unitrace collection mode; all collects utilization and EU stalls in separate sequential runs",
    )
    p_profile.add_argument(
        "--sampling-interval-us",
        type=int,
        default=None,
        help="Unitrace sampling interval in microseconds (default: installed tool's default)",
    )
    p_profile.add_argument(
        "--assembly",
        action="store_true",
        help="Capture IGC shader dumps with unitrace (disables persistent JIT caches for collection)",
    )
    p_profile.add_argument(
        "--parent-profile",
        default=None,
        help="Retained unitrace artifact directory to compare kernel resources against",
    )
    p_profile.add_argument("--vtune-bin", default="vtune")
    p_profile.add_argument(
        "--kernel-name", default=None, help="Trial tree to record the attempt in"
    )
    p_profile.add_argument("--trial-id", default=None, help="Saved trial this kernel file is")
    p_profile.add_argument("--trials-dir", default="./trials")

    # -- upstream --
    p_up = subparsers.add_parser("upstream", help="Port a finalized winner back as a patch")
    up_sub = p_up.add_subparsers(dest="upstream_command", required=True)
    u_clone = up_sub.add_parser("clone", help="Private shared clone into upstream/")
    u_clone.add_argument("repo")
    u_clone.add_argument("--name", default=None, help="Clone into upstream-<name>/ instead")
    u_clone.add_argument("--rev", default=None, help="Base commit (default: the repo's HEAD)")
    u_patch = up_sub.add_parser("patch", help="Write and verify output/upstream.patch")
    u_patch.add_argument("kernel_name")
    u_patch.add_argument("--kind", choices=["upstream", "serving-only"], default="upstream")
    u_patch.add_argument("--notes", default="", help="What the patch changes and why")
    u_patch.add_argument(
        "--stack", default="", help='your stack check verdict: "PASS|FAIL <evidence>"'
    )
    u_patch.add_argument("--trials-dir", default="./trials")

    # -- capture --
    p_cap = subparsers.add_parser("capture", help="Run a launcher under the instruments; check it")
    cap_sub = p_cap.add_subparsers(dest="capture_command", required=True)
    c_run = cap_sub.add_parser(
        "run", help="Launcher -> raw/ (its trace, unitrace kernels, counters)"
    )
    c_run.add_argument("launcher", help="executable that runs one profiled batch into $RAW_DIR")
    c_run.add_argument("-o", "--raw", required=True, help="raw/ directory to write")
    c_run.add_argument("--unitrace", default=os.environ.get("UNITRACE_BIN", "unitrace"))
    c_parse = cap_sub.add_parser("parse", help="raw/ -> capture.json, checked against unitrace")
    c_parse.add_argument("raw", help="raw/ directory capture run wrote")
    c_parse.add_argument("-o", "--output", required=True, help="capture.json to write")
    c_parse.add_argument("--names", default=None, help="names.json to apply")

    # -- patch --
    p_patch = subparsers.add_parser("patch", help="Gate a patch that makes a framework run a model")
    patch_sub = p_patch.add_subparsers(dest="patch_command", required=True)
    pc = patch_sub.add_parser("check", help="git diff, fresh apply, import, tokens vs transformers")
    pc.add_argument("dir", help="holds patch.json and generate.sh")
    pc.add_argument("--repos", required=True, help="the clones' directory (--kernel-repos)")
    pc.add_argument(
        "--framework-python", required=True, help="the unpatched framework's interpreter"
    )
    pc.add_argument(
        "--reference-python", default=None, help="GPU torch + transformers, if not that"
    )
    pc.add_argument("--model", required=True)
    pc.add_argument("--batch", type=int, required=True)
    pc.add_argument("--input-len", type=int, required=True)
    pc.add_argument("--output-len", type=int, required=True)
    pc.add_argument("--min-match", type=int, default=8, help="greedy tokens that must agree")
    pc.add_argument("--trust-remote-code", action="store_true")

    # -- bound --
    p_bound = subparsers.add_parser("bound", help="Each workload's efficiency against its bound")
    p_bound.add_argument("capture", help="capture.json")
    p_bound.add_argument("--calibration", default=None, help="xe-forge calibrate output")
    p_bound.add_argument("--min-share", type=float, default=0.005, help="smallest share listed")
    p_bound.add_argument("--json", action="store_true", help="rows as JSON")

    # -- inputs --
    p_inputs = subparsers.add_parser("inputs", help="Gate the shapes and weights trials use")
    inputs_sub = p_inputs.add_subparsers(dest="inputs_command", required=True)
    ic = inputs_sub.add_parser("check", help="each chosen shape captured, weights sum to 1")
    ic.add_argument("inputs", help="inputs.json: {kernel: {weights, why}}")
    ic.add_argument("--manifest", required=True, help="kernels.yaml")

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
    elif args.skill == "upstream":
        from xe_forge.skills.upstream import run
    elif args.skill == "capture":
        from xe_forge.skills.capture import run
    elif args.skill == "bound":
        from xe_forge.skills.bound import run
    elif args.skill == "patch":
        from xe_forge.skills.patch import run
    elif args.skill == "inputs":
        from xe_forge.skills.inputs import run

    # A tool's verdict is its exit code: a failed check must not exit 0.
    return run(args)


if __name__ == "__main__":
    main()
