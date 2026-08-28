"""
The `orbit-bench` command line: run a command with declared warmup and repetitions,
emit one structured JSON document, and compare two such documents. Deliberately
stdlib plus `xe_forge.orbit.stats`/`models` only — no torch, no GPU, no framework.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from xe_forge.orbit import stats
from xe_forge.orbit.models import SCHEMA_VERSION, Decision

TOOL_NAME = "orbit-bench"

# No accept/reject below this many repetitions; `stats.compare` returns INVALID
# under it. A document measured with fewer still carries its samples, but is
# marked non-decision-grade and `compare` refuses it.
DECISION_MIN_REPETITIONS = 5

# Exit codes for `compare`, one per verdict plus "refused the input".
EXIT_ACCEPT = 0
EXIT_REJECT = 1
EXIT_INCONCLUSIVE = 2
EXIT_INVALID = 3
EXIT_REFUSED = 4

# tests/orbit/test_bench_cli.py asserts the "17.5" reference stays in this help text.
_WARMUP_HELP = (
    "Warmup runs executed before measurement and DISCARDED. The discard is declared "
    "up front, before any timing — which is what separates a warmup from dropping an "
    "outlier after seeing the data (plan §17.5). Default: 1."
)

_RUN_EPILOG = """\
The command to measure goes after `--`:

    orbit-bench run --repetitions 5 --warmup 1 -- python train.py

Warmup runs are executed and discarded as declared warmup: the discard is stated
before measurement, never applied to a sample after seeing it.

Exit status: 0 when every repetition succeeded; 1 when any repetition failed
(the document is then marked "valid": false — a failed run is not a fast run);
2 on usage errors.
"""

_COMPARE_EPILOG = """\
Both inputs must be documents produced by `orbit-bench run`. The verdict is computed
by the same accept/reject arithmetic the rest of Orbit uses, lower wall
time is better, and INCONCLUSIVE is a real outcome, not a soft REJECT.

Exit status:
    0  ACCEPT        candidate is faster, 95% CI excludes zero
    1  REJECT        candidate is slower, 95% CI excludes zero
    2  INCONCLUSIVE  95% CI straddles zero
    3  INVALID       the comparison itself could not be made (e.g. zero baseline)
    4  input refused: unreadable, wrong schema major version, marked
       "valid": false, or not decision grade (fewer than 5 repetitions)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Standalone measurement backbone: run a workload with declared warmup and "
            "repetitions, emit structured JSON, and compare two such documents."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_p = sub.add_parser(
        "run",
        help="run a command with warmup + repetitions and emit a JSON measurement",
        description="Time a command over repeated runs and emit one JSON document.",
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument(
        "--repetitions",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Measured runs. Below "
            f"{DECISION_MIN_REPETITIONS} the document is marked non-decision-grade "
            "and `compare` refuses it. Default: 5."
        ),
    )
    run_p.add_argument("--warmup", type=int, default=1, metavar="N", help=_WARMUP_HELP)
    run_p.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        metavar="S",
        help="Per-repetition timeout in seconds; a timed-out repetition fails the run. Default: 1800.",
    )
    run_p.add_argument(
        "--json",
        default="-",
        metavar="PATH",
        help="Where to write the JSON document: a path, or '-' for stdout (default).",
    )
    run_p.set_defaults(func=cmd_run)

    cmp_p = sub.add_parser(
        "compare",
        help="compare two stored measurement documents and print a verdict",
        description="Decide ACCEPT/REJECT/INCONCLUSIVE/INVALID between two run documents.",
        epilog=_COMPARE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmp_p.add_argument("--baseline", required=True, metavar="A.json")
    cmp_p.add_argument("--candidate", required=True, metavar="B.json")
    cmp_p.set_defaults(func=cmd_compare)

    return parser


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _run_once(command: list[str], timeout: float) -> tuple[int | None, float, str]:
    """One execution: (exit code or None, wall seconds, failure note or '')."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - start, f"timed out after {timeout:g}s"
    except OSError as exc:
        # The command could not be started at all — not a slow run, not a fast one.
        return None, time.perf_counter() - start, f"could not start: {exc}"
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        hint = tail[-1] if tail else "no stderr"
        return proc.returncode, elapsed, f"exit {proc.returncode} ({hint})"
    return 0, elapsed, ""


def cmd_run(args: argparse.Namespace, command: list[str]) -> int:
    if not command:
        print(
            "no workload command given. Put it after `--`, e.g.\n"
            "  orbit-bench run -- python train.py",
            file=sys.stderr,
        )
        return 2

    repetitions = max(1, args.repetitions)
    warmup = max(0, args.warmup)

    # Declared warmup: executed, then discarded — stated here, before any sample
    # exists, so the discard can never be a reaction to the data.
    for _ in range(warmup):
        _run_once(command, args.timeout)

    samples: list[float] = []
    exit_codes: list[int | None] = []
    failures: list[str] = []
    for index in range(repetitions):
        code, elapsed, note = _run_once(command, args.timeout)
        samples.append(elapsed)
        exit_codes.append(code)
        if note:
            failures.append(f"repetition {index} of {repetitions}: {note}")

    wall = stats.estimate(samples, unit="s")
    mde = stats.minimum_detectable_effect(samples)

    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "command": list(command),
        "repetitions": repetitions,
        "warmup": warmup,
        "wall_time": wall.model_dump(mode="json"),
        # None, not Infinity: a single sample has no resolvable effect size, and
        # Infinity is not valid JSON.
        "minimum_detectable_effect_percent": mde if math.isfinite(mde) else None,
        "exit_codes": exit_codes,
        "environment": {
            "hostname": _hostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "valid": not failures,
    }
    if failures:
        # A failed run is not a fast run: its wall time stays in the document for
        # inspection, but the document as a whole is unusable for a decision.
        reason = failures[0]
        if len(failures) > 1:
            reason += f"; {len(failures)} of {repetitions} repetitions failed"
        document["reason"] = reason

    document["decision_grade"] = repetitions >= DECISION_MIN_REPETITIONS
    if repetitions < DECISION_MIN_REPETITIONS:
        document["decision_grade_reason"] = (
            f"{repetitions} repetitions; a decision requires at least "
            f"{DECISION_MIN_REPETITIONS} for any accept/reject"
        )

    rendered = json.dumps(document, indent=2)
    if args.json == "-":
        print(rendered)
    else:
        Path(args.json).write_text(rendered + "\n", encoding="utf-8")
        print(f"measurement written to {args.json}", file=sys.stderr)

    return 0 if not failures else 1


def _hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _load_document(path: str) -> tuple[dict | None, str | None]:
    """Load and vet one measurement document. Returns (doc, refusal reason)."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"{path}: cannot read ({exc})"
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"{path}: not JSON ({exc})"
    if not isinstance(doc, dict):
        return None, f"{path}: not a measurement document (top level is not an object)"

    version = str(doc.get("schema_version", ""))
    ours = SCHEMA_VERSION.split(".")[0]
    theirs = version.split(".")[0] if version else ""
    if theirs != ours:
        return None, (
            f"{path}: schema_version {version or '<missing>'} has major {theirs or '?'}, "
            f"this tool speaks {SCHEMA_VERSION} (major {ours})"
        )

    wall = doc.get("wall_time")
    if not isinstance(wall, dict) or not isinstance(wall.get("samples"), list):
        return None, f"{path}: no wall_time.samples; not an orbit-bench measurement"
    if not wall["samples"]:
        return None, f"{path}: wall_time.samples is empty"

    if doc.get("valid") is not True:
        reason = doc.get("reason", "no reason recorded")
        return None, f"{path}: document is marked invalid ({reason})"

    if doc.get("decision_grade") is not True:
        reason = doc.get(
            "decision_grade_reason",
            f"fewer than {DECISION_MIN_REPETITIONS} repetitions",
        )
        return None, f"{path}: not decision grade ({reason})"

    return doc, None


def cmd_compare(args: argparse.Namespace, command: list[str]) -> int:
    if command:
        print("compare takes no `-- command`; it reads two stored documents", file=sys.stderr)
        return 2

    baseline_doc, refusal = _load_document(args.baseline)
    if refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    candidate_doc, refusal = _load_document(args.candidate)
    if refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    assert baseline_doc is not None and candidate_doc is not None

    base_samples = [float(x) for x in baseline_doc["wall_time"]["samples"]]
    cand_samples = [float(x) for x in candidate_doc["wall_time"]["samples"]]

    # paired=False, explicitly: these documents were measured separately, never
    # interleaved, so pairing sample i with sample i would invent a correlation
    # that was never produced. Welch is the honest estimator here.
    decision, detail = stats.compare(
        base_samples,
        cand_samples,
        lower_is_better=True,
        paired=False,
        min_repetitions=DECISION_MIN_REPETITIONS,
    )

    print(f"verdict: {decision.value}")
    if "improvement_percent" in detail:
        print(
            f"  improvement: {detail['improvement_percent']:+.2f}%  "
            f"95% CI [{detail['ci95_low']:.2f}%, {detail['ci95_high']:.2f}%]"
        )
    if "minimum_detectable_effect" in detail:
        print(f"  MDE: {detail['minimum_detectable_effect']:.2f}%")
    if "reason" in detail:
        print(f"  reason: {detail['reason']}")

    return {
        Decision.ACCEPT: EXIT_ACCEPT,
        Decision.REJECT: EXIT_REJECT,
        Decision.INCONCLUSIVE: EXIT_INCONCLUSIVE,
        Decision.INVALID: EXIT_INVALID,
    }[decision]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Split at the first `--` ourselves: everything after it is the workload command,
    # verbatim — argparse never sees it, so the workload's own flags cannot collide
    # with ours.
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1 :]

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args, command)


if __name__ == "__main__":
    sys.exit(main())
