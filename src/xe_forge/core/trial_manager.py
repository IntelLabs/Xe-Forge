#!/usr/bin/env python3
"""Trial Tree State Manager for iterative kernel optimization.

Manages a tree of optimization trials for each kernel, tracking parent-child
relationships, strategies, correctness, and speedup results. Supports
branching back to the best ancestor when a trial regresses.

Usage:
    python src/xe_forge/core/trial_manager.py init <kernel_name> <pytorch_file>
    python src/xe_forge/core/trial_manager.py save <kernel_name> <trial_file> --parent <parent_id> --strategy "description"
    python src/xe_forge/core/trial_manager.py result <kernel_name> <trial_id> --validation <pass|fail> --correctness <pass|fail> --speedup <float> --baseline_us <float> --triton_us <float>
    python src/xe_forge/core/trial_manager.py status <kernel_name>
    python src/xe_forge/core/trial_manager.py best <kernel_name>
    python src/xe_forge/core/trial_manager.py finalize <kernel_name> <output_file>
"""

import argparse
import json
import os
import shutil
import sys
from types import SimpleNamespace


def _trials_dir() -> str:
    """Trial root directory. Resolves to CWD/trials unless XE_FORGE_TRIALS_DIR set."""
    env = os.environ.get("XE_FORGE_TRIALS_DIR", "").strip()
    if env:
        return env
    try:
        from xe_forge.config import get_config

        return get_config().trial.trials_dir
    except Exception:
        return os.path.join(os.getcwd(), "trials")


def _output_dir() -> str:
    """Finalize output directory. Resolves to CWD/output unless XE_FORGE_OUTPUT_DIR set."""
    env = os.environ.get("XE_FORGE_OUTPUT_DIR", "").strip()
    if env:
        return env
    try:
        from xe_forge.config import get_config

        return get_config().trial.output_dir
    except Exception:
        return os.path.join(os.getcwd(), "output")


# Back-compat module-level aliases — evaluated lazily via properties on a
# namespace; reading TRIALS_DIR / OUTPUT_DIR gives the current value.
def __getattr__(name):
    if name == "TRIALS_DIR":
        return _trials_dir()
    if name == "OUTPUT_DIR":
        return _output_dir()
    raise AttributeError(name)


def _state_path(kernel_name):
    return os.path.join(_trials_dir(), kernel_name, "state.json")


def _trial_dir(kernel_name):
    return os.path.join(_trials_dir(), kernel_name)


def _load_state(kernel_name):
    path = _state_path(kernel_name)
    if not os.path.exists(path):
        print(f"Error: No trial tree found for '{kernel_name}'. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        state = json.load(f)
    # Backward compat: old state files lack baseline_type
    state.setdefault("baseline_type", "pytorch")
    # Backward compat: rename pytorch_us -> baseline_us in trials
    for trial in state.get("trials", {}).values():
        if "pytorch_us" in trial and "baseline_us" not in trial:
            trial["baseline_us"] = trial.pop("pytorch_us")
    return state


def _save_state(kernel_name, state):
    path = _state_path(kernel_name)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# Commands
# ============================================================================


def cmd_init(args):
    """Initialize a new trial tree for a kernel."""
    kernel_name = args.kernel_name
    pytorch_file = args.pytorch_file

    trial_dir = _trial_dir(kernel_name)
    if os.path.exists(_state_path(kernel_name)):
        print(
            f"Warning: Trial tree for '{kernel_name}' already exists. Use a different name or delete trials/{kernel_name}/."
        )
        # sys.exit(1)
    else:
        os.makedirs(trial_dir, exist_ok=True)

        baseline_type = "triton" if args.triton_baseline else "pytorch"
        state = {
            "kernel_name": kernel_name,
            "pytorch_file": pytorch_file,
            "baseline_type": baseline_type,
            "trials": {},
            "best_trial": None,
            "next_id": 0,
            "baseline_us": None,
        }
        _save_state(kernel_name, state)
        baseline_label = "Triton" if baseline_type == "triton" else "PyTorch"
        print(f"Initialized trial tree for '{kernel_name}' in trials/{kernel_name}/")
        print(f"  Baseline ({baseline_label}): {pytorch_file}")


def cmd_save(args):
    """Save a trial by copying the kernel file into the trial directory."""
    kernel_name = args.kernel_name
    trial_file = args.trial_file
    parent = args.parent  # None or "t0", "t1", etc.
    strategy = args.strategy or ""

    if not os.path.exists(trial_file):
        print(f"Error: Trial file '{trial_file}' not found.", file=sys.stderr)
        sys.exit(1)

    state = _load_state(kernel_name)

    # Validate parent exists (gracefully handle first trial with mistaken --parent)
    if parent is not None and parent not in state["trials"]:
        if state["next_id"] == 0:
            print(
                f"Warning: Ignoring --parent '{parent}' for first trial (no trials exist yet).",
                file=sys.stderr,
            )
            parent = None
        else:
            print(
                f"Error: Parent trial '{parent}' not found. Available: {list(state['trials'].keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    trial_id = f"t{state['next_id']}"
    state["next_id"] += 1

    # Copy file
    dest = os.path.join(_trial_dir(kernel_name), f"{trial_id}.py")
    try:
        shutil.copy2(trial_file, dest)
    except Exception as e:
        print(f"File already written", file=sys.stderr)

    state["trials"][trial_id] = {
        "parent": parent,
        "file": f"{trial_id}.py",
        "strategy": strategy,
        "validation": None,
        "correctness": None,
        "speedup": None,
        "baseline_us": None,
        "triton_us": None,
        "status": "saved",
    }
    _save_state(kernel_name, state)
    print(f"Saved trial {trial_id}: {strategy}")
    print(f"  Parent: {parent or 'root'}")
    print(f"  File: trials/{kernel_name}/{trial_id}.py")


def cmd_result(args):
    """Record results for a trial."""
    kernel_name = args.kernel_name
    trial_id = args.trial_id

    state = _load_state(kernel_name)

    if trial_id not in state["trials"]:
        args.trial_file = os.path.join(_trial_dir(kernel_name), f"{trial_id}.py")
        args.parent = None  # consider changing
        args.strategy = None
        print(f"Error: Trial '{trial_id}' not found. Saving state", file=sys.stderr)
        cmd_save(args)  # Auto-save if trial doesn't exist
        state = _load_state(kernel_name)  # Reload state after saving
        # print(f"Error: Trial '{trial_id}' not found. Available: {list(state['trials'].keys())}", file=sys.stderr)
        # sys.exit(1)

    trial = state["trials"][trial_id]

    if args.validation:
        trial["validation"] = args.validation
    if args.correctness:
        trial["correctness"] = args.correctness
    if args.speedup is not None:
        trial["speedup"] = args.speedup
    if args.baseline_us is not None:
        trial["baseline_us"] = args.baseline_us
    if args.triton_us is not None:
        trial["triton_us"] = args.triton_us

    # Cache baseline_us at kernel level on first recording (for --baseline-us skip on later trials)
    if args.baseline_us is not None and state.get("baseline_us") is None:
        state["baseline_us"] = [args.baseline_us]

    # Update status
    if trial["validation"] == "fail" or trial["correctness"] == "fail":
        trial["status"] = "failed"
    elif trial["correctness"] == "pass" and trial["speedup"] is not None:
        trial["status"] = "completed"
    else:
        trial["status"] = "partial"

    # Update best trial (highest speedup among correct trials)
    best_speedup = -1.0
    best_id = None
    for tid, t in state["trials"].items():
        if t.get("correctness") == "pass" and t.get("speedup") is not None:
            if t["speedup"] > best_speedup:
                best_speedup = t["speedup"]
                best_id = tid
    state["best_trial"] = best_id

    _save_state(kernel_name, state)

    status_icon = {"completed": "+", "failed": "X", "partial": "~", "saved": "?"}
    icon = status_icon.get(trial["status"], "?")
    runtime_str = ""
    if trial.get("baseline_us") is not None and trial.get("triton_us") is not None:
        runtime_str = f", baseline={trial['baseline_us']:.2f}us, triton={trial['triton_us']:.2f}us"
    print(
        f"[{icon}] {trial_id}: validation={trial['validation']}, correctness={trial['correctness']}, speedup={trial['speedup']}{runtime_str}"
    )
    if state["best_trial"]:
        best = state["trials"][state["best_trial"]]
        best_runtime = ""
        if best.get("baseline_us") is not None and best.get("triton_us") is not None:
            best_runtime = (
                f", baseline={best['baseline_us']:.2f}us, triton={best['triton_us']:.2f}us"
            )
        print(f"  Best trial: {state['best_trial']} ({best['speedup']}x{best_runtime})")


def cmd_status(args):
    """Show trial tree status as ASCII tree."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    baseline_label = "Triton" if state.get("baseline_type") == "triton" else "PyTorch"
    print(f"Trial tree: {state['kernel_name']}")
    print(f"  Baseline ({baseline_label}): {state['pytorch_file']}")
    print(f"  Best: {state['best_trial'] or 'none'}")
    print(f"  Trials: {len(state['trials'])}")
    print()

    if not state["trials"]:
        print("  (no trials yet)")
        return

    # Build children map
    children = {}
    roots = []
    for tid, t in state["trials"].items():
        parent = t["parent"]
        if parent is None:
            roots.append(tid)
        else:
            children.setdefault(parent, []).append(tid)

    # Sort by trial number
    def sort_key(tid):
        return int(tid[1:])

    roots.sort(key=sort_key)
    for k in children:
        children[k].sort(key=sort_key)

    # Print tree recursively
    def print_node(tid, prefix="", is_last=True):
        trial = state["trials"][tid]
        connector = "└── " if is_last else "├── "

        # Status indicators
        is_best = tid == state["best_trial"]
        status_icon = {"completed": "+", "failed": "X", "partial": "~", "saved": "?"}
        icon = status_icon.get(trial["status"], "?")

        speedup_str = f"{trial['speedup']:.2f}x" if trial["speedup"] is not None else "---"
        runtime_str = ""
        if trial.get("baseline_us") is not None and trial.get("triton_us") is not None:
            runtime_str = f" (bl={trial['baseline_us']:.0f}us, tr={trial['triton_us']:.0f}us)"
        best_marker = " <<<< BEST" if is_best else ""
        strategy_short = trial["strategy"][:60] if trial["strategy"] else ""

        print(
            f"{prefix}{connector}[{icon}] {tid}: {speedup_str}{runtime_str} | {strategy_short}{best_marker}"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = children.get(tid, [])
        for i, child in enumerate(kids):
            print_node(child, child_prefix, i == len(kids) - 1)

    for i, root in enumerate(roots):
        print_node(root, "  ", i == len(roots) - 1)


def cmd_best(args):
    """Get the best trial info."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    if state["best_trial"] is None:
        print("No correct trials yet.")
        sys.exit(1)

    best_id = state["best_trial"]
    best = state["trials"][best_id]
    best_file = os.path.join(_trial_dir(kernel_name), best["file"])

    print(f"best_trial: {best_id}")
    print(f"speedup: {best['speedup']}")
    if best.get("baseline_us") is not None:
        print(f"baseline_us: {best['baseline_us']}")
    if best.get("triton_us") is not None:
        print(f"triton_us: {best['triton_us']}")
    print(f"strategy: {best['strategy']}")
    print(f"file: {best_file}")
    print(f"parent: {best['parent'] or 'root'}")


def cmd_baseline_us(args):
    """Print cached baseline time(s) as comma-separated floats."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    baseline_us = state.get("baseline_us")
    if baseline_us is None:
        print(
            "No baseline_us cached yet. Run benchmark and record result for t0 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(",".join(f"{v:.2f}" for v in baseline_us))


def cmd_finalize(args):
    """Copy the best correct trial to the output path.

    If output_file has no directory component it is placed inside the configured
    output directory (``output/`` at the project root) which is created automatically.
    """
    kernel_name = args.kernel_name
    output_file = args.output_file

    state = _load_state(kernel_name)

    if state["best_trial"] is None:
        print("Error: No correct trials to finalize.", file=sys.stderr)
        sys.exit(1)

    best_id = state["best_trial"]
    best = state["trials"][best_id]
    src = os.path.join(_trial_dir(kernel_name), best["file"])

    # Default bare filenames into the configured output directory
    if os.path.dirname(output_file) == "":
        out_dir = _output_dir()
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, output_file)

    shutil.copy2(src, output_file)
    runtime_str = ""
    if best.get("baseline_us") is not None and best.get("triton_us") is not None:
        runtime_str = f", baseline={best['baseline_us']:.2f}us, triton={best['triton_us']:.2f}us"
    print(f"Finalized {best_id} ({best['speedup']}x{runtime_str}) -> {output_file}")
    print(f"  Strategy: {best['strategy']}")


# ============================================================================
# Python API — importable wrappers around the CLI actions
#
# These return dicts instead of printing/exiting, so the in-process trial
# runner can use them without shelling out. The CLI functions (cmd_*) remain
# the authoritative implementations.
# ============================================================================


def init_state(kernel_name: str, baseline_file: str, triton_baseline: bool = False) -> dict:
    """Initialize a new trial tree. Returns the initial state dict."""
    trial_dir = _trial_dir(kernel_name)
    if os.path.exists(_state_path(kernel_name)):
        return _load_state(kernel_name)
    os.makedirs(trial_dir, exist_ok=True)
    state = {
        "kernel_name": kernel_name,
        "pytorch_file": baseline_file,
        "baseline_type": "triton" if triton_baseline else "pytorch",
        "trials": {},
        "best_trial": None,
        "next_id": 0,
        "baseline_us": None,
    }
    _save_state(kernel_name, state)
    return state


def save_trial(
    kernel_name: str,
    trial_file: str,
    parent: str | None = None,
    strategy: str = "",
) -> str:
    """Save a trial by copying the file into the trial directory.

    Returns the trial ID (e.g. ``"t3"``).
    """
    if not os.path.exists(trial_file):
        raise FileNotFoundError(f"Trial file '{trial_file}' not found")

    state = _load_state(kernel_name)
    if parent is not None and parent not in state["trials"]:
        if state["next_id"] == 0:
            parent = None
        else:
            raise ValueError(
                f"Parent trial '{parent}' not found. "
                f"Available: {list(state['trials'].keys())}"
            )

    trial_id = f"t{state['next_id']}"
    state["next_id"] += 1
    dest = os.path.join(_trial_dir(kernel_name), f"{trial_id}.py")
    # Only copy if source and destination differ (runner may have written
    # directly into the trial dir).
    if os.path.abspath(trial_file) != os.path.abspath(dest):
        shutil.copy2(trial_file, dest)

    state["trials"][trial_id] = {
        "parent": parent,
        "file": f"{trial_id}.py",
        "strategy": strategy,
        "validation": None,
        "correctness": None,
        "speedup": None,
        "baseline_us": None,
        "triton_us": None,
        "status": "saved",
    }
    _save_state(kernel_name, state)
    return trial_id


def record_result(
    kernel_name: str,
    trial_id: str,
    validation: str | None = None,
    correctness: str | None = None,
    speedup: float | None = None,
    baseline_us: float | None = None,
    triton_us: float | None = None,
) -> dict:
    """Record results for a trial. Returns the updated trial record."""
    state = _load_state(kernel_name)
    if trial_id not in state["trials"]:
        raise KeyError(
            f"Trial '{trial_id}' not found. Available: {list(state['trials'].keys())}"
        )

    trial = state["trials"][trial_id]
    if validation is not None:
        trial["validation"] = validation
    if correctness is not None:
        trial["correctness"] = correctness
    if speedup is not None:
        trial["speedup"] = speedup
    if baseline_us is not None:
        trial["baseline_us"] = baseline_us
    if triton_us is not None:
        trial["triton_us"] = triton_us

    if baseline_us is not None and state.get("baseline_us") is None:
        state["baseline_us"] = [baseline_us]

    if trial["validation"] == "fail" or trial["correctness"] == "fail":
        trial["status"] = "failed"
    elif trial["correctness"] == "pass" and trial["speedup"] is not None:
        trial["status"] = "completed"
    else:
        trial["status"] = "partial"

    best_speedup = -1.0
    best_id = None
    for tid, t in state["trials"].items():
        if t.get("correctness") == "pass" and t.get("speedup") is not None:
            if t["speedup"] > best_speedup:
                best_speedup = t["speedup"]
                best_id = tid
    state["best_trial"] = best_id

    _save_state(kernel_name, state)
    return trial


def get_state(kernel_name: str) -> dict:
    """Return the current state dict (raises if not initialized)."""
    return _load_state(kernel_name)


def best_trial(kernel_name: str) -> dict | None:
    """Return the best trial record, or None if no correct trial exists."""
    state = _load_state(kernel_name)
    best_id = state.get("best_trial")
    if best_id is None:
        return None
    trial = dict(state["trials"][best_id])
    trial["trial_id"] = best_id
    trial["path"] = os.path.join(_trial_dir(kernel_name), trial["file"])
    return trial


def get_baseline_us(kernel_name: str) -> list[float] | None:
    """Return cached baseline runtime(s) in microseconds, or None."""
    state = _load_state(kernel_name)
    return state.get("baseline_us")


def read_trial(kernel_name: str, trial_id: str) -> str:
    """Read a trial's source code as a string."""
    state = _load_state(kernel_name)
    if trial_id not in state["trials"]:
        raise KeyError(
            f"Trial '{trial_id}' not found. Available: {list(state['trials'].keys())}"
        )
    path = os.path.join(_trial_dir(kernel_name), state["trials"][trial_id]["file"])
    with open(path) as f:
        return f.read()


def finalize(kernel_name: str, output_file: str) -> str:
    """Copy the best correct trial to output_file. Returns the final path."""
    state = _load_state(kernel_name)
    if state["best_trial"] is None:
        raise RuntimeError(f"No correct trials to finalize for '{kernel_name}'")

    best_id = state["best_trial"]
    best = state["trials"][best_id]
    src = os.path.join(_trial_dir(kernel_name), best["file"])

    if os.path.dirname(output_file) == "":
        out_dir = _output_dir()
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, output_file)

    shutil.copy2(src, output_file)
    return output_file


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Trial Tree State Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize trial tree")
    p_init.add_argument("kernel_name", help="Kernel identifier (e.g. 39_Gemm_Scale_BatchNorm)")
    p_init.add_argument("pytorch_file", help="Path to baseline file (PyTorch or Triton)")
    p_init.add_argument(
        "--triton-baseline",
        action="store_true",
        help="Baseline is a Triton kernel (default: PyTorch)",
    )

    # save
    p_save = subparsers.add_parser("save", help="Save a trial")
    p_save.add_argument("kernel_name", help="Kernel identifier")
    p_save.add_argument("trial_file", help="Path to the trial kernel file")
    p_save.add_argument("--parent", default=None, help="Parent trial ID (e.g. t0)")
    p_save.add_argument("--strategy", default="", help="Description of optimization strategy")

    # result
    p_result = subparsers.add_parser("result", help="Record trial results")
    p_result.add_argument("kernel_name", help="Kernel identifier")
    p_result.add_argument("trial_id", help="Trial ID (e.g. t0)")
    p_result.add_argument("--validation", choices=["pass", "fail"], help="Validation result")
    p_result.add_argument("--correctness", choices=["pass", "fail"], help="Correctness result")
    p_result.add_argument("--speedup", type=float, help="Speedup over baseline")
    p_result.add_argument(
        "--baseline_us",
        type=float,
        help="Baseline runtime in microseconds (PyTorch or Triton baseline)",
    )
    p_result.add_argument("--triton_us", type=float, help="Triton kernel runtime in microseconds")

    # status
    p_status = subparsers.add_parser("status", help="Show trial tree status")
    p_status.add_argument("kernel_name", help="Kernel identifier")

    # best
    p_best = subparsers.add_parser("best", help="Get best trial info")
    p_best.add_argument("kernel_name", help="Kernel identifier")

    # baseline-us
    p_baseline_us = subparsers.add_parser("baseline-us", help="Print cached baseline time(s)")
    p_baseline_us.add_argument("kernel_name", help="Kernel identifier")

    # finalize
    p_finalize = subparsers.add_parser("finalize", help="Copy best trial to output")
    p_finalize.add_argument("kernel_name", help="Kernel identifier")
    p_finalize.add_argument(
        "output_file", help="Output file path (bare filename defaults to output/)"
    )

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "save": cmd_save,
        "result": cmd_result,
        "status": cmd_status,
        "best": cmd_best,
        "baseline-us": cmd_baseline_us,
        "finalize": cmd_finalize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
