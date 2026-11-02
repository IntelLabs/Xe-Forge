"""Rank captured workloads by recoverable share of device time and gate them into entries.

Gates (routable, buildable, shaped, bounded, worth, top) each write one line to gates.log.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

from xe_forge.frontend import formula
from xe_forge.frontend.canonical import resolve_dims
from xe_forge.frontend.dtypes import storage_bytes
from xe_forge.frontend.ir import CaptureRun, Workload
from xe_forge.models import DSL

_ROUTE_REASONS = {
    "library": "library primitive: change the call, not the kernel",
    "aten": "implemented inside PyTorch: no kernel repository here to edit",
    "unnamed": "not placed by the naming agent",
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
    eff: float | None = None  # fraction of its bound


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
    target: str = "kernel"  # or "library"


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
                    "target": e.target,
                }
                for e in self.entries
            ],
            "decisions": {k: asdict(v) for k, v in self.decisions.items()},
        }


AT_ROOFLINE = 0.8
NOISE = 0.1  # a bound this far above its measurement is timer noise; further is an error


def experts_touched(experts: int, rows: int, topk: int | None) -> float:
    """Expected distinct experts a grouped GEMM reads under uniform routing."""
    k = topk or 1
    if experts <= 1 or rows <= 0:
        return float(max(1, experts))
    return experts * (1.0 - (1.0 - min(k, experts) / experts) ** (rows / k))


def _points(w: Workload) -> list[tuple[int | None, int, float]]:
    """(token count, calls, device us) per captured token count, or the workload as one."""
    if w.var_axes:
        hist = next(iter(w.var_axes.values()))
        return [(int(v), h["calls"], h["device_us"]) for v, h in hist.items()]
    return [(None, w.calls, w.device_us)]


def _against(w: Workload, per_call_bound) -> tuple[float, float] | None:
    """(recoverable us, efficiency) of a per-call bound; None if it exceeds a measurement."""
    recoverable = reached_total = 0.0
    for value, calls, us in _points(w):
        bound = per_call_bound(value)
        if isinstance(bound, tuple):  # (expected, guaranteed)
            expected, sure = bound
            bound = expected if calls * expected <= us * (1 + NOISE) else sure
        if bound is None or calls * bound > us * (1 + NOISE):
            return None
        reached = min(calls * bound, us)
        recoverable += us - reached
        reached_total += reached
    return recoverable, reached_total / w.device_us if w.device_us else 0.0


def gemm_bound(
    w: Workload, calib: dict | None, ctx: dict | None = None
) -> tuple[float, float] | None:
    """(recoverable us, efficiency) against the GEMM roofline; None if unbounded."""
    ctx = ctx or {}
    if not calib or not {"M", "N", "K"} <= set(w.dims) or "var" in (w.dims["N"], w.dims["K"]):
        return None
    dtype = w.dtypes[0] if w.dtypes else None
    size = storage_bytes(dtype)
    bw = calib.get("bandwidth_gbs")
    peaks = {k: v for k, v in (calib.get("matmul_tflops") or {}).items() if v}
    if not bw or not peaks:
        return None
    topk = ctx.get("topk")
    parts = w.attrs.get("parts") or [
        {**w.attrs, "N": w.dims["N"], "K": w.dims["K"], "E": w.dims.get("E")}
    ]

    def part_time(p: dict, m, sure: bool = False) -> float | None:
        if "w_bytes" not in p:
            return None
        a_b, w_b, o_b = p.get("a_bytes", size), p["w_bytes"], p.get("o_bytes", size)
        peak = peaks.get(p.get("compute", dtype)) or max(peaks.values())
        rows, e_eff = m, 1
        if p.get("grouped"):
            e = p.get("E")
            if e is None and ctx.get("experts"):
                e = ctx["experts"] / (ctx.get("ep") or 1)
            if e is None:
                return None  # never assume E=1
            rows = m * (topk or 1) if p.get("rows") == "token" else m
            e_eff = w.attrs.get("active_experts") or (
                # guaranteed: at least top-k experts
                min(e, topk or 1) if sure else experts_touched(e, rows, topk)
            )
        n, k = p["N"], p["K"]
        moved = rows * k * a_b + e_eff * k * n * w_b + rows * n * o_b
        return max(moved / (bw * 1e3), 2 * rows * n * k / (peak * 1e6))

    estimated = any(p.get("grouped") for p in parts) and not w.attrs.get("active_experts")

    def per_call(m):
        m = w.dims["M"] if m is None else m
        times = [part_time(p, m) for p in parts]
        if None in times:
            return None
        if not estimated:
            return sum(times)
        return sum(times), sum(part_time(p, m, sure=True) for p in parts)

    return _against(w, per_call)


def cost_bound(
    w: Workload, calib: dict | None, ctx: dict | None = None
) -> tuple[float, float] | None:
    """(recoverable us, efficiency) against the naming agent's stated per-call cost."""
    cost = w.attrs.get("cost") or {}
    bw = (calib or {}).get("bandwidth_gbs")
    if not bw or "bytes" not in cost:
        return None
    peaks = {k: v for k, v in (calib.get("matmul_tflops") or {}).items() if v}
    dtype = next((d for d in w.dtypes if d.startswith(("bfloat", "float"))), None)
    peak = peaks.get(dtype) or max(peaks.values(), default=None)

    def per_call(value):
        dims = resolve_dims(w, value)
        moved = formula.evaluate(cost["bytes"], dims)
        if moved is None:
            return None
        flops = formula.evaluate(cost["flops"], dims) if cost.get("flops") else 0.0
        if flops is None or (flops and not peak):
            return None
        return max(moved / (bw * 1e3), (flops / (peak * 1e6)) if flops else 0.0)

    return _against(w, per_call)


def measured_sol(
    w: Workload, calib: dict | None, ctx: dict | None = None
) -> tuple[float, float] | None:
    """(recoverable us, efficiency) from the run's hardware counters."""
    parts = _sol_parts(w, calib, ctx)
    if parts is None:
        return None
    eff = parts[0]
    return w.device_us * (1 - eff), eff


def _measured(measured: dict, name: str) -> dict | None:
    """A kernel's counters; the trace may cut long names the counters keep whole."""
    if name in measured:
        return measured[name]
    hits = [v for k, v in measured.items() if k.startswith(name)]
    ns = sum(v["us"] for v in hits)
    if not ns:
        return None
    return {f: sum(v[f] * v["us"] for v in hits) / ns for f in ("gb_per_s", "util_pct")}


def _sol_parts(w: Workload, calib: dict | None, ctx: dict | None) -> tuple | None:
    """(SOL, memory SOL, compute SOL); None when most of the time is unmeasured."""
    measured = (ctx or {}).get("counters") or {}
    bw = (calib or {}).get("bandwidth_gbs")
    kernels = w.framework_meta.get("kernels") or {}
    weight = sol = mem = comp = 0.0
    for name, us in kernels.items():
        k = _measured(measured, name)
        if not k or not us:
            continue
        memory, compute = (k["gb_per_s"] / bw if bw else 0.0), k["util_pct"] / 100
        sol += min(1.0, max(memory, compute)) * us
        mem, comp, weight = mem + memory * us, comp + compute * us, weight + us
    if not weight or weight < 0.5 * sum(kernels.values()):
        return None
    return sol / weight, mem / weight, comp / weight


# A stated cost takes precedence over these (see ``bound``).
BOUNDERS = {"gemm": gemm_bound}


def bound(w: Workload, calib: dict | None, ctx: dict | None = None) -> tuple[float, float] | None:
    """(recoverable us, efficiency) of the stricter known bound; None if none."""
    if w.attrs.get("cost"):
        theory = cost_bound(w, calib, ctx)
    else:
        bounder = BOUNDERS.get(w.family)
        theory = bounder(w, calib, ctx) if bounder else None
    known = [b for b in (theory, measured_sol(w, calib, ctx)) if b is not None]
    return min(known, key=lambda b: b[1]) if known else None


def _decide(
    w: Workload,
    min_gain: float,
    calib: dict | None = None,
    ctx: dict | None = None,
    at_roofline: float = AT_ROOFLINE,
) -> Decision:
    d = Decision(
        workload_id=w.workload_id,
        enabled=False,
        share=w.share,
        gain=w.share,
        calls=w.calls,
        mean_us=round(w.device_us / w.calls, 3) if w.calls else 0.0,
    )
    n = w.naming
    bounded = bound(w, calib, ctx)
    if bounded is not None:
        d.eff = round(bounded[1], 3)
        d.gain = w.share * (bounded[0] / w.device_us) if w.device_us else 0.0
        if d.eff >= at_roofline:
            d.gates.append(("bounded", "REJECT", f"at roofline: eff {d.eff:.2f} of its bound"))
            d.reason = f"at roofline (eff {d.eff:.2f} of the card's measured bound)"
            return d
    if n.route == "kernel" or (n.route == "library" and bounded is not None and n.repo):
        what = "library call" if n.route == "library" else "kernel"
        d.gates.append(("routable", "PASS", f"{what} {n.name} in {n.repo} ({n.dsl}) via {n.via}"))
    else:
        d.gates.append(("routable", "REJECT", f"{n.route}: {_ROUTE_REASONS[n.route]} ({n.via})"))
        d.reason = _ROUTE_REASONS[n.route]
        return d
    if n.dsl not in {x.value for x in DSL}:
        d.gates.append(("buildable", "REJECT", f"dsl {n.dsl!r} not in {[x.value for x in DSL]}"))
        d.reason = f"Xe-Forge has no {n.dsl} path"
        return d
    if not w.dims:
        d.gates.append(("shaped", "REJECT", "no argument shapes recorded"))
        d.reason = "launched outside any op: no argument shapes for a spec"
        return d
    d.gates.append(("shaped", "PASS", f"{len(w.dims)} dims"))
    if bounded is None:
        why = (
            "no bound: no counters for its kernels and no GEMM or stated cost"
            if calib
            else "no calibration"
        )
        d.gates.append(("bounded", "UNKNOWN", f"{why}: ceiling taken as the full share"))
    else:
        parts = _sol_parts(w, calib, ctx)
        sol = f"; measured memory {parts[1]:.0%}, compute {parts[2]:.0%} of peak" if parts else ""
        d.gates.append(
            (
                "bounded",
                "PASS",
                f"eff {d.eff:.2f} of its bound; recoverable {bounded[0]:.0f} us{sol}",
            )
        )
    if d.gain < min_gain:
        d.gates.append(("worth", "REJECT", f"G={d.gain:.4%} < {min_gain:.4%}"))
        d.reason = f"below --min-gain {min_gain:.2%}"
        return d
    d.gates.append(("worth", "PASS", f"G={d.gain:.4%} >= {min_gain:.4%}"))
    d.enabled = True
    return d


def _counted(workloads: list[Workload], measured: dict) -> float:
    """% of device time whose kernels have counters."""
    times = [(us, _measured(measured, k) is not None)
             for w in workloads for k, us in (w.framework_meta.get("kernels") or {}).items()]  # fmt: skip
    total = sum(us for us, _ in times)
    return round(100 * sum(us for us, hit in times if hit) / total, 1) if total else 0.0


def analyze(
    capture: CaptureRun,
    top: int = 3,
    min_gain: float = 0.005,
    calibration: dict | None = None,
    at_roofline: float = AT_ROOFLINE,
) -> Analysis:
    facts = capture.run.get("model_facts") or {}
    ctx = {
        "counters": capture.run.get("counters") or {},
        "topk": facts.get("topk"),
        "experts": facts.get("experts"),
        "ep": (capture.run.get("runtime") or {}).get("ep"),
    }
    decisions = {
        w.workload_id: _decide(w, min_gain, calibration, ctx, at_roofline)
        for w in capture.workloads
    }

    groups: dict[tuple, list[Workload]] = defaultdict(list)
    for w in capture.workloads:
        n = w.naming
        groups[(n.route, n.repo, n.name)].append(w)

    entries = []
    for (route, repo, name), ws in groups.items():
        ds = [decisions[w.workload_id] for w in ws]
        # Worth is per kernel: many small workloads pass when they add up.
        small = [d for d in ds if d.gates and d.gates[-1][:2] == ("worth", "REJECT")]
        total = sum(d.gain for d in ds if d.enabled or d in small)
        if small and total >= min_gain:
            for d in small:
                d.gates[-1] = ("worth", "PASS", f"kernel total G={total:.4%} >= {min_gain:.4%}")
                d.enabled, d.reason = True, ""
        e = Entry(
            name=name or ws[0].workload_id,
            repo=repo,
            dsl=ws[0].naming.dsl,
            workloads=ws,
            decisions=ds,
            target="library" if route == "library" else "kernel",
        )
        e.priority = round(sum(d.gain for d in ds if d.enabled), 6)
        e.enabled = any(d.enabled for d in ds)
        share = sum(d.share for d in ds)
        calls = sum(d.calls for d in ds)
        effs = [d.eff for d in ds if d.eff is not None]
        bound = f"eff {min(effs):.2f} of its bound" if effs else "bound unknown"
        e.reason = f"{share:.2%} of device time; {calls} calls; " + (
            bound if e.enabled else ds[0].reason
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
        options={
            "top": top,
            "min_gain": min_gain,
            "bound": "calibrated" if calibration else "unknown (no calibration)",
            "counters_pct": _counted(capture.workloads, ctx["counters"]),
        },
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
        (
            "Ceilings are bounded by the calibrated card: the stricter of each kernel's measured "
            "speed-of-light (hardware counters, covering "
            f"{analysis.options.get('counters_pct') or 0}% of device time), a GEMM's roofline "
            "and a cost the naming agent stated; an op with no bound takes its whole share."
            if analysis.options.get("bound") == "calibrated"
            else "Ceilings are unbounded (no calibration): each workload's ceiling is its whole share."
        )
        + f" `--top {analysis.options['top']}`, `--min-gain {analysis.options['min_gain']}`.",
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
