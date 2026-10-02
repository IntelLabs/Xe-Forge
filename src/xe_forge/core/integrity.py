"""Checks a trial must pass beyond matching the oracle once.

A trial that matches the oracle on one set of inputs can still be wrong in a way the
timing loop rewards: it returns an input or a persistent buffer, caches its result across
calls (the timed calls all see the same inputs at the same addresses), leaves output
elements unwritten, races, or runs its work on a queue the timer does not watch. Each check
below calls the trial a few more times and names what it caught; none of them times
anything.

Three checks are probabilistic -- ``UNWRITTEN_OUTPUT``, ``NONDETERMINISTIC`` and
``OFF_STREAM`` -- a pass says the defect did not show, not that it is absent. The two that
depend on device memory and queues run only on a GPU. A wrong value is seen by every
value check that runs after it, so read the names together: ``UNWRITTEN_OUTPUT`` among
them points at unwritten elements, ``OFF_STREAM`` alone at the queue.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

SEEDS = (1001, 1002, 1003)
REPEATS = 3

# Names with no use in a kernel file; each reads or alters the measurement or the harness.
_HARNESS_ACCESS = re.compile(
    r"\bxe_forge\b|\bai_bench\b|\bset_all_seeds\b|\b_getframe\b|\binspect\.stack\b"
    r"|\bgc\.get_objects\b|\belapsed_time\b|\bEvent\("
)


def scan_source(code: str) -> list[str]:
    """``HARNESS_ACCESS`` failures for trial source that reaches into the harness."""
    found = sorted({m.group(0) for m in _HARNESS_ACCESS.finditer(code)})
    if not found:
        return []
    return [f"HARNESS_ACCESS: trial source references {', '.join(found)}"]


@dataclass
class OracleRecord:
    """What the oracle did, recorded before the trial is loaded."""

    expected: dict[int, Any]  # host copy of the output per seed
    aliases: bool  # returns storage shared with an input
    reuses: bool  # a later call overwrites an earlier output


def record_oracle(oracle: Callable, make_inputs: Callable[[int], list]) -> OracleRecord:
    """Run the oracle on every seed; a defect it has itself is not held against the trial."""
    with torch.no_grad():
        expected = {seed: _snapshot(oracle(*make_inputs(seed))) for seed in SEEDS}
        return OracleRecord(
            expected, _call_aliases(oracle, make_inputs(SEEDS[0])), _reuses(oracle, make_inputs)
        )


def check_trial(
    record: OracleRecord,
    trial: Callable,
    make_inputs: Callable[[int], list],
    match: Callable[[Any, Any], bool],
    device: str,
) -> list[str]:
    """Run every check; return ``"NAME: detail"`` per failure, empty when all pass.

    ``trial`` resets any state it owns on each call. ``make_inputs(seed)`` returns fresh
    inputs on ``device``; ``match(expected, actual)`` compares host copies.
    """
    failures = []
    gpu = torch.device(device).type in ("xpu", "cuda")
    expected = record.expected
    with torch.no_grad():
        x1 = make_inputs(SEEDS[0])
        out1 = trial(*x1)
        if not record.aliases and _aliases(out1, x1):
            failures.append("OUTPUT_ALIASES_INPUT: an output shares storage with an input")
        first = _snapshot(out1)
        out2 = trial(*make_inputs(SEEDS[1]))
        if not record.reuses and not _identical(first, _snapshot(out1)):
            failures.append(
                "OUTPUT_REUSED: a later call overwrote an earlier call's output; "
                "return newly allocated tensors"
            )
        for seed, actual in ((SEEDS[0], first), (SEEDS[1], _snapshot(out2))):
            if not match(expected[seed], actual):
                failures.append(
                    f"STALE_RESULT: output differs from the oracle on new inputs (seed {seed}); "
                    "possible causes: a result cached by shape or from an earlier call"
                )
                break
        del out1, out2

        # Same tensors, new values: what the fixed-input timing loop cannot tell apart.
        for target, source in zip(_tensors(x1), _tensors(make_inputs(SEEDS[2])), strict=True):
            target.copy_(source)
        if not match(expected[SEEDS[2]], _snapshot(trial(*x1))):
            failures.append(
                "CACHED_BY_ADDRESS: output differs after new values are written into the same "
                "input tensors; possible causes: a result cached by address"
            )
        del x1

        if gpu:
            x = make_inputs(SEEDS[0])
            _poison_allocator(expected[SEEDS[0]], device)
            if not match(expected[SEEDS[0]], _snapshot(trial(*x))):
                failures.append(
                    "UNWRITTEN_OUTPUT: output differs when freshly allocated memory is "
                    "poisoned; possible causes: elements never written (probabilistic check)"
                )
            del x

        for _ in range(REPEATS):
            if not match(expected[SEEDS[0]], _snapshot(trial(*make_inputs(SEEDS[0])))):
                failures.append(
                    f"NONDETERMINISTIC: one of {REPEATS} repeated calls differs from the "
                    "oracle; possible causes: a race or an uninitialized read (probabilistic check)"
                )
                break

        if gpu and not _on_current_stream(trial, make_inputs, expected[SEEDS[0]], match, device):
            failures.append(
                "OFF_STREAM: output differs when read on the caller's stream while the default "
                "stream is busy; possible causes: work on another queue, outside what the timer "
                "measures (probabilistic check)"
            )
    return failures


def _on_current_stream(trial, make_inputs, expected, match, device) -> bool:
    """Run the trial on a fresh current stream while the default stream is busy.

    Catches a kernel submitting to a queue of its own; torch work on another torch stream
    was not caught when this was tested.
    """
    dev = torch.device(device)
    x = make_inputs(SEEDS[0])
    torch.accelerator.synchronize()
    busy = torch.randn(2048, 2048, device=dev)
    for _ in range(8):
        busy = busy @ busy
        busy = busy / busy.norm()
    previous = torch.accelerator.current_stream(dev)
    torch.accelerator.set_stream(torch.Stream(device=dev))
    try:
        # The host copy is ordered on the current stream only.
        actual = _snapshot(trial(*x))
    finally:
        torch.accelerator.set_stream(previous)
        torch.accelerator.synchronize()
    return match(expected, actual)


def _poison_allocator(outputs, device: str) -> None:
    """Fill and free blocks the size of each output, so the allocator hands them back."""
    blocks = [
        torch.full((t.numel() * t.element_size(),), 0xFF, dtype=torch.uint8, device=device)
        for t in _tensors(outputs)
    ]
    torch.accelerator.synchronize()
    del blocks


def _reuses(fn, make_inputs) -> bool:
    out = fn(*make_inputs(SEEDS[0]))
    before = _snapshot(out)
    fn(*make_inputs(SEEDS[1]))
    return not _identical(before, _snapshot(out))


def _call_aliases(fn, inputs) -> bool:
    return _aliases(fn(*inputs), inputs)


def _aliases(outputs, inputs) -> bool:
    held = {t.untyped_storage().data_ptr() for t in _tensors(inputs) if t.numel()}
    return any(t.untyped_storage().data_ptr() in held for t in _tensors(outputs) if t.numel())


def _identical(left, right) -> bool:
    """Bitwise equality: NaN equals itself, and every dtype (fp8 included) compares."""
    a, b = list(_tensors(left)), list(_tensors(right))
    return len(a) == len(b) and all(
        x.shape == y.shape and x.dtype == y.dtype and torch.equal(_bytes(x), _bytes(y))
        for x, y in zip(a, b, strict=True)
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value
