"""Delegating correctness and timing to a host-supplied command.

The numbers the trial loop branches on -- ``--correctness``, ``--speedup``,
``--baseline-us`` -- decide whether it continues on a branch, returns to the best
trial, or changes approach. Xe-Forge produces them from random tensors of the
spec's shapes and from a wall-clock measurement with no floor. Both are fine for
a GEMM on a large shape and wrong elsewhere:

* ``randn`` is the wrong *domain* for many kernels -- a sampling kernel needs
  rows that sum to one, a ragged kernel a monotonic ``indptr``, a quantized
  kernel positive scales. Fed noise, a correct kernel can be scored wrong.
* with no timing floor, a ratio is reported whether or not the two arms were
  distinguishable. A loop branches just as confidently on noise as on a result.

A host that already has real workload data and a calibrated timer can answer
both properly. It supplies a command; Xe-Forge runs it and reads back a small
line-oriented contract, which keeps the two sides in separate processes -- so
they need not share an interpreter, an environment, or a set of installed
packages.

The contract, on stdout::

    CORRECTNESS: pass|fail
    BASELINE_US: <float>
    TRIAL_US:    <float>
    SPEEDUP:     <float>            # omitted when the comparison was gated
    TIMER:       <methodology>
    VERDICT:     <name of the gate that decided>
    DONE

``DONE`` last is what makes the reading safe: a command that crashed, timed out
or was truncated cannot be mistaken for one that measured something, whatever
its exit code. Absent ``DONE``, this module raises rather than return a partial
result, because every field it would return is one the loop would branch on.

This module lives beside :mod:`xe_forge.config` rather than under
``xe_forge.core`` deliberately: it imports nothing but the standard library, so
delegating measurement does not first require importing ``torch`` and
``ai_bench`` -- which is the point, for a host that is delegating precisely
because it does not want that path.
"""

from __future__ import annotations

import logging
import shlex
import string
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1800

__all__ = [
    "DEFAULT_TIMEOUT",
    "ExternalCommandError",
    "ExternalResult",
    "render_command",
    "run_external",
]


class ExternalCommandError(RuntimeError):
    """The external command could not be rendered, run, or read."""


@dataclass(frozen=True)
class ExternalResult:
    """What a host command reported. Any field may be absent."""

    correctness: bool | None = None
    baseline_us: float | None = None
    trial_us: float | None = None
    speedup: float | None = None
    timer: str | None = None
    verdict: str | None = None
    returncode: int = 0
    raw: str = ""

    @property
    def measured(self) -> bool:
        """True when a usable speedup came back.

        A gated comparison -- one the host refused to turn into a ratio because
        the times sat inside its timing floor -- is *not* a regression and is
        not an improvement. It is an absence of evidence, and the caller has to
        be able to tell the difference.
        """
        return self.speedup is not None


_FIELDS = {
    "CORRECTNESS": "correctness",
    "BASELINE_US": "baseline_us",
    "TRIAL_US": "trial_us",
    "SPEEDUP": "speedup",
    "TIMER": "timer",
    "VERDICT": "verdict",
}


def render_command(template: str, **substitutions: Any) -> list[str]:
    """Split *template* into argv, then substitute ``{placeholders}`` per token.

    Splitting before substituting is deliberate: a path containing a space stays
    one argument instead of becoming two. Nothing goes through a shell.
    """
    try:
        tokens = shlex.split(template)
    except ValueError as exc:
        raise ExternalCommandError(f"cannot parse command template {template!r}: {exc}") from exc
    if not tokens:
        raise ExternalCommandError("command template is empty")

    values = {k: ("" if v is None else str(v)) for k, v in substitutions.items()}
    formatter = string.Formatter()
    argv = []
    for token in tokens:
        try:
            argv.append(formatter.vformat(token, (), values))
        except KeyError as exc:
            raise ExternalCommandError(
                f"unknown placeholder {exc} in command template token {token!r}; "
                f"available: {', '.join(sorted(values))}"
            ) from exc
        except (IndexError, ValueError) as exc:
            raise ExternalCommandError(f"bad placeholder in token {token!r}: {exc}") from exc
    return argv


def _parse(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in output.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        field = _FIELDS.get(key.strip().upper())
        if field is None:
            continue
        value = value.strip()
        if field == "correctness":
            parsed[field] = value.lower() in ("pass", "passed", "true", "ok", "1")
        elif field in ("baseline_us", "trial_us", "speedup"):
            try:
                parsed[field] = float(value)
            except ValueError:
                logger.warning("Ignoring unparsable %s value %r from external command", key, value)
        else:
            parsed[field] = value
    return parsed


def run_external(
    template: str,
    *,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    **substitutions: Any,
) -> ExternalResult:
    """Run a host command and read its result. Raises unless it completed."""
    argv = render_command(template, **substitutions)
    logger.info("Running external command: %s", " ".join(shlex.quote(a) for a in argv))

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except FileNotFoundError:
        raise ExternalCommandError(f"external command not found: {argv[0]!r}") from None
    except subprocess.TimeoutExpired:
        raise ExternalCommandError(
            f"external command timed out after {timeout}s: {argv[0]!r}"
        ) from None
    except OSError as exc:
        raise ExternalCommandError(f"could not run external command {argv[0]!r}: {exc}") from exc

    output = proc.stdout + proc.stderr
    if not any(line.strip() == "DONE" for line in output.splitlines()):
        raise ExternalCommandError(
            f"external command did not complete (no DONE marker, exit {proc.returncode}). "
            f"Nothing in its output can be read as a measurement.\n{output[-2000:]}"
        )

    parsed = _parse(output)
    return ExternalResult(returncode=proc.returncode, raw=output, **parsed)
