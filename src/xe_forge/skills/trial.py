"""xe-forge-skill trial: Trial tree management."""

import sys


def run(args):
    from xe_forge.core.trial_manager import TrialManager

    mgr = TrialManager(args.trials_dir)

    if args.trial_command == "init":
        mgr.init(args.kernel_name, args.baseline_file, triton_baseline=args.triton_baseline)
        print(f"Initialized trial tree for '{args.kernel_name}'")

    elif args.trial_command == "save":
        trial_id = mgr.save_trial(
            args.kernel_name,
            args.trial_file,
            parent=args.parent,
            strategy=args.strategy,
        )
        print(f"Saved trial {trial_id}")

    elif args.trial_command == "result":
        trial = mgr.record_result(
            args.kernel_name,
            args.trial_id,
            validation=args.validation,
            correctness=args.correctness,
            speedup=args.speedup,
            baseline_us=args.baseline_us,
            triton_us=args.triton_us,
        )
        status_icon = {"completed": "+", "failed": "X", "partial": "~", "saved": "?"}
        icon = status_icon.get(trial["status"], "?")
        print(
            f"[{icon}] {args.trial_id}: correctness={trial['correctness']}, speedup={trial['speedup']}"
        )

    elif args.trial_command == "status":
        print(mgr.get_status(args.kernel_name))

    elif args.trial_command == "best":
        best = mgr.get_best(args.kernel_name)
        if best:
            print(f"best_trial: {best['id']}")
            print(f"speedup: {best.get('speedup')}")
            print(f"strategy: {best.get('strategy')}")
            print(f"file: {best.get('file_path')}")
        else:
            print("No correct trials yet.")
            sys.exit(1)

    elif args.trial_command == "baseline-us":
        val = mgr.get_baseline_us(args.kernel_name)
        if val:
            print(",".join(f"{v:.2f}" for v in val))
        else:
            print("No baseline_us cached yet.", file=sys.stderr)
            sys.exit(1)

    elif args.trial_command == "finalize":
        best_id = mgr.finalize(args.kernel_name, args.output_file)
        if best_id:
            print(f"Finalized {best_id} -> {args.output_file}")
        else:
            print("No correct trials to finalize.", file=sys.stderr)
            sys.exit(1)
