"""
Shared record of which kernel implementation actually dispatched (plan §13).

This lives in its own module for a specific reason. The workload is normally run as
`python -m examples.kernel_replacement.workload`, which binds that file to `__main__`.
An override module that then does `from examples.kernel_replacement import workload`
imports a *second, distinct* module object — so a list defined in the workload would be
appended to by the override and read by nobody.

The symptom is the worst possible one for this project: the dispatch assertion sees no
kernel names at all and cannot tell "the override never ran" from "the workload never
reported". Both look like an honest negative (§13).

Keeping the log in a module that is only ever imported by its canonical name removes
the ambiguity: every participant appends to the same list.
"""

from __future__ import annotations

# Kernel identifiers, in dispatch order. Duplicates are meaningful — they show a kernel
# ran more than once — so this is a list rather than a set.
DISPATCHED: list[str] = []


def record(kernel_name: str) -> None:
    DISPATCHED.append(kernel_name)


def clear() -> None:
    DISPATCHED.clear()


def observed() -> list[str]:
    """Unique kernel names in first-dispatch order."""
    seen: list[str] = []
    for name in DISPATCHED:
        if name not in seen:
            seen.append(name)
    return seen
