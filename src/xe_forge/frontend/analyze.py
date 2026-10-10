"""Which captured workloads are worth an Xe-Forge run, decided by arithmetic.

Per workload, over the profiled window:

    S = device time / total device time          (share)
    G = recoverable time / total device time     (end-to-end ceiling)

Without a calibrated bound for the part, nothing says how far below its current time a
workload could go, so G is taken as S and the row says ``bound: unknown``; the roofline
gate reports ``unknown`` rather than passing. That ranks by share alone -- the right order
when all that is known is where the time goes, and why a 5 us kernel called 100,000 times
outranks a 300 us kernel called twice.

Gates, in order, each one line in gates.log::

    GATE <workload_id> <gate> PASS|REJECT|UNKNOWN <evidence>

``routable``   naming found a kernel in a repository Xe-Forge can edit
``shaped``     the trace recorded the call's argument shapes (a spec needs dims)
``bounded``    measured below its roofline bound (unknown until calibrated)
``worth``      G >= --min-gain
``top``        among the --top kernels by summed G

Workloads are then grouped by kernel: one manifest entry per (repository, name), whose
spec carries the variants of every workload that kernel served.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from xe_forge.frontend.ir import CaptureRun, Workload

_ROUTE_REASONS = {
    "library": "library primitive: change the call, not the kernel",
    "aten": "ATen op inside PyTorch: no kernel repository to edit",
    "unnamed": "no naming rule matched",
}


@dataclass
class Decision:
    workload_id: str
    enabled: bool
    share: float
    gain: float
    calls: int
    mean_us: float
    gates: list[tuple[str, str, str]] = field(default_factory=list)
    reason: str = ""


@dataclass
class Entry:
    """One manifest entry: a kernel, and the workloads it served."""

    name: str
    repo: str | None
    dsl: str | None
    workloads: list[Workload]
    decisions: list[Decision]
    enabled: bool = False
    priority: float = 0.0
    reason: str = ""


@dataclass
class Analysis:
    run_id: str
    total_device_us: float
    entries: list[Entry]
    decisions: dict[str, Decision]
    options: dict

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_device_us": self.total_device_us,
            "options": self.options,
            "entries": [
                {
                    "name": e.name,
                    "repo": e.repo,
                    "dsl": e.dsl,
                    "enabled": e.enabled,
                    "priority": e.priority,
                    "reason": e.reason,
                    "workloads": [w.workload_id for w in e.workloads],
                }
                for e in self.entries
            ],
            "decisions": {k: asdict(v) for k, v in self.decisions.items()},
        }


def _decide(w: Workload, min_gain: float) -> Decision:
    d = Decision(
        workload_id=w.workload_id,
        enabled=False,
        share=w.share,
        gain=w.share,  # unbounded: the ceiling is the whole share
        calls=w.calls,
        mean_us=round(w.device_us / w.calls, 3) if w.calls else 0.0,
    )
    n = w.naming
    if n.route == "kernel":
        d.gates.append(("routable", "PASS", f"{n.name} in {n.repo} ({n.dsl}) via {n.via}"))
    else:
        d.gates.append(("routable", "REJECT", f"{n.route}: {_ROUTE_REASONS[n.route]} ({n.via})"))
        d.reason = _ROUTE_REASONS[n.route]
        return d
    if not w.dims:
        d.gates.append(("shaped", "REJECT", "no argument shapes recorded"))
        d.reason = "launched outside any op: no argument shapes for a spec"
        return d
    d.gates.append(("shaped", "PASS", f"{len(w.dims)} dims"))
    d.gates.append(("bounded", "UNKNOWN", "no calibration: ceiling taken as the full share"))
    if d.gain < min_gain:
        d.gates.append(("worth", "REJECT", f"G={d.gain:.4%} < {min_gain:.4%}"))
        d.reason = f"below --min-gain {min_gain:.2%}"
        return d
    d.gates.append(("worth", "PASS", f"G={d.gain:.4%} >= {min_gain:.4%}"))
    d.enabled = True
    return d


def analyze(capture: CaptureRun, top: int = 5, min_gain: float = 0.005) -> Analysis:
    decisions = {w.workload_id: _decide(w, min_gain) for w in capture.workloads}

    groups: dict[tuple, list[Workload]] = defaultdict(list)
    for w in capture.workloads:
        n = w.naming
        groups[(n.route, n.repo, n.name)].append(w)

    entries = []
    for (_, repo, name), ws in groups.items():
        ds = [decisions[w.workload_id] for w in ws]
        e = Entry(
            name=name or ws[0].workload_id,
            repo=repo,
            dsl=ws[0].naming.dsl,
            workloads=ws,
            decisions=ds,
        )
        e.priority = round(sum(d.gain for d in ds if d.enabled), 6)
        e.enabled = any(d.enabled for d in ds)
        share = sum(d.share for d in ds)
        calls = sum(d.calls for d in ds)
        e.reason = f"{share:.2%} of device time; {calls} calls; " + (
            "bound unknown" if e.enabled else ds[0].reason
        )
        entries.append(e)

    entries.sort(key=lambda e: (-e.priority, -sum(w.share for w in e.workloads)))
    rank = 0
    for e in entries:
        if not e.enabled:
            continue
        rank += 1
        verdict = "PASS" if rank <= top else "REJECT"
        for d in e.decisions:
            if d.enabled:
                d.gates.append(("top", verdict, f"kernel rank {rank} of top {top}"))
                d.enabled = verdict == "PASS"
        if verdict == "REJECT":
            e.enabled = False
            e.reason += f"; rank {rank}, beyond --top {top}"

    return Analysis(
        run_id=capture.run.get("id", ""),
        total_device_us=capture.run.get("total_device_us", 0.0),
        entries=entries,
        decisions=decisions,
        options={"top": top, "min_gain": min_gain, "bound": "unknown (no calibration)"},
    )


def gates_log(analysis: Analysis) -> str:
    lines = []
    for wid, d in analysis.decisions.items():
        for gate, verdict, evidence in d.gates:
            lines.append(f"GATE {wid} {gate} {verdict} {evidence}")
    return "\n".join(lines) + "\n"


def report(analysis: Analysis, capture: CaptureRun) -> str:
    r = capture.run
    out = [
        f"# Kernel analysis: {r.get('model', {}).get('name', '?')} under {r.get('framework')}",
        "",
        f"Run `{analysis.run_id}`: {r.get('kernel_invocations')} kernel invocations, "
        f"{len(capture.ops)} ops, {len(capture.workloads)} workloads, "
        f"{analysis.total_device_us / 1e3:.1f} ms device time "
        f"({r.get('attributed_pct')}% attributed to an op).",
        "",
        f"Ceilings are unbounded (no calibration): each workload's ceiling is its whole "
        f"share. `--top {analysis.options['top']}`, `--min-gain {analysis.options['min_gain']}`.",
        "",
        "## Manifest entries",
        "",
        "| # | kernel | repo | dsl | enabled | share | priority | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(analysis.entries, 1):
        share = sum(w.share for w in e.workloads)
        out.append(
            f"| {i} | `{e.name}` | {e.repo or '-'} | {e.dsl or '-'} | "
            f"{'yes' if e.enabled else 'no'} | {share:.2%} | {e.priority:.4f} | {e.reason} |"
        )
    out += [
        "",
        "## Workloads",
        "",
        "| workload | op | calls | mean us | share | dims | decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in capture.workloads:
        d = analysis.decisions[w.workload_id]
        dims = ", ".join(f"{k}={v}" for k, v in w.dims.items())
        out.append(
            f"| `{w.workload_id}` | `{w.framework_meta.get('framework_op')}` | {w.calls} | "
            f"{d.mean_us} | {w.share:.2%} | {dims} | "
            f"{'optimize' if d.enabled else 'skip: ' + d.reason} |"
        )
    return "\n".join(out) + "\n"
