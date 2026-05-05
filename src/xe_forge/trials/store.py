"""Typed wrapper around trial_manager's module-level functions.

The tree/state is authoritatively implemented in
`xe_forge.core.trial_manager`. This module exposes a stable, typed facade
for the runner/search/writer code so they don't import the CLI helpers
directly.
"""

from xe_forge.core import trial_manager as _tm


def init_state(kernel_name: str, baseline_file: str, triton_baseline: bool = False) -> dict:
    """Initialize a new trial tree (idempotent if one already exists)."""
    return _tm.init_state(kernel_name, baseline_file, triton_baseline=triton_baseline)


def save_trial(
    kernel_name: str,
    trial_file: str,
    parent: str | None = None,
    strategy: str = "",
) -> str:
    """Save a trial kernel file as the next trial. Returns its ID (``t<N>``)."""
    return _tm.save_trial(kernel_name, trial_file, parent=parent, strategy=strategy)


def record_result(
    kernel_name: str,
    trial_id: str,
    validation: str | None = None,
    correctness: str | None = None,
    speedup: float | None = None,
    baseline_us: float | None = None,
    triton_us: float | None = None,
) -> dict:
    """Update a trial's validation / correctness / speedup. Returns the trial record."""
    return _tm.record_result(
        kernel_name,
        trial_id,
        validation=validation,
        correctness=correctness,
        speedup=speedup,
        baseline_us=baseline_us,
        triton_us=triton_us,
    )


def get_state(kernel_name: str) -> dict:
    """Return the raw state dict — trials, best_trial, baseline_us, etc."""
    return _tm.get_state(kernel_name)


def best_trial(kernel_name: str) -> dict | None:
    """Return the best trial record (+ ``trial_id`` and ``path`` keys) or None."""
    return _tm.best_trial(kernel_name)


def read_trial(kernel_name: str, trial_id: str) -> str:
    """Read the source code of a trial."""
    return _tm.read_trial(kernel_name, trial_id)


def get_baseline_us(kernel_name: str) -> list[float] | None:
    """Return cached baseline time(s) in microseconds."""
    return _tm.get_baseline_us(kernel_name)


def finalize(kernel_name: str, output_file: str) -> str:
    """Copy the best trial to output_file. Returns the final path."""
    return _tm.finalize(kernel_name, output_file)


def trial_path(kernel_name: str, trial_id: str) -> str:
    """Absolute path of a trial's source file on disk."""
    state = _tm.get_state(kernel_name)
    if trial_id not in state["trials"]:
        raise KeyError(f"Trial '{trial_id}' not found")
    import os

    return os.path.join(_tm._trial_dir(kernel_name), state["trials"][trial_id]["file"])


def trial_dir(kernel_name: str) -> str:
    """Absolute path of the trial directory."""
    return _tm._trial_dir(kernel_name)
