"""Rank captured workloads by recoverable share of device time and decide which to optimize.

A workload is skipped with one reason: at roofline, not routable, no Xe-Forge DSL, no shapes,
below --min-gain, or beyond --top. The arithmetic (bounds) is here; the reasons go to
kernels.yaml and analysis.json.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from xe_forge.frontend import formula
from xe_forge.frontend.calibrate import peaks as calib_peaks
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
    reason: str = ""
    small: bool = False  # below --min-gain on its own; its kernel's total may still pass
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
    bw, peaks = calib_peaks(calib)
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
    bw, peaks = calib_peaks(calib)
    if not bw or "bytes" not in cost:
        return None
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
    if not sum(v["us"] for v in hits):
        return None
    out = {"us": sum(v["us"] for v in hits)}
    for f in ("gb_per_s", "util_pct"):  # None = not measured, averaged over those that were
        known = [v for v in hits if v.get(f) is not None]
        ns = sum(v["us"] for v in known)
        out[f] = sum(v[f] * v["us"] for v in known) / ns if ns else None
    return out


def _sol_parts(w: Workload, calib: dict | None, ctx: dict | None) -> tuple | None:
    """(SOL, memory SOL, compute SOL); None when most of the time is unmeasured."""
    measured = (ctx or {}).get("counters") or {}
    shared = (ctx or {}).get("shared") or set()
    bw = calib_peaks(calib)[0]
    kernels = w.framework_meta.get("kernels") or {}
    weight = sol = mem = comp = 0.0
    for name, us in kernels.items():
        # Counters are per kernel name, over every launch the process made: a name several
        # workloads run (one library GEMM for every matmul, prefill and decode of one norm)
        # averages their shapes, so it bounds none of them.
        k = None if name in shared else _measured(measured, name)
        # Metric sampling misses short launches: counters that sampled under half of the
        # kernel's measured time describe a few launches, not the kernel.
        if not k or not us or (k.get("us") or 0.0) < 0.5 * us:
            continue
        # SOL is the max of both sides; an unknown side (or no bandwidth peak) is not 0% of
        # peak, so such a kernel counts as unmeasured.
        if not bw or k.get("gb_per_s") is None or k.get("util_pct") is None:
            continue
        memory, compute = k["gb_per_s"] / bw, k["util_pct"] / 100
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
            d.reason = f"at roofline (eff {d.eff:.2f} of the card's measured bound)"
            return d
    if not (n.route == "kernel" or (n.route == "library" and bounded is not None and n.repo)):
        d.reason = _ROUTE_REASONS[n.route]
        return d
    if n.dsl not in {x.value for x in DSL}:
        d.reason = f"Xe-Forge has no {n.dsl} path"
        return d
    if not w.dims:
        d.reason = "launched outside any op: no argument shapes for a spec"
        return d
    if d.gain < min_gain:  # with no bound, the ceiling is the whole share
        d.reason, d.small = f"below --min-gain {min_gain:.2%}", True
        return d
    d.enabled = True
    return d


def _counted(workloads: list[Workload], measured: dict, shared: set = frozenset()) -> float:
    """% of device time whose kernels have counters that bound them (a name of one workload)."""

    def hit(name):
        k = None if name in shared else _measured(measured, name)
        return bool(k) and k["gb_per_s"] is not None and k["util_pct"] is not None

    def bounds(name, us):
        k = None if name in shared else _measured(measured, name)
        return hit(name) and (k.get("us") or 0.0) >= 0.5 * us

    times = [(us, bounds(k, us))
             for w in workloads for k, us in (w.framework_meta.get("kernels") or {}).items()]  # fmt: skip
    total = sum(us for us, _ in times)
    return round(100 * sum(us for us, hit in times if hit) / total, 1) if total else 0.0


def context(capture: CaptureRun) -> dict:
    """What every bound reads besides the workload and calibration."""
    facts = capture.run.get("model_facts") or {}
    runs: dict[str, set] = defaultdict(set)
    for w in capture.workloads:
        for name in w.framework_meta.get("kernels") or {}:
            runs[name].add(w.workload_id)
    return {
        "counters": capture.run.get("counters") or {},
        "shared": {name for name, ws in runs.items() if len(ws) > 1},
        "topk": facts.get("topk"),
        "experts": facts.get("experts"),
        "ep": (capture.run.get("runtime") or {}).get("ep"),
    }


def analyze(
    capture: CaptureRun,
    top: int = 3,
    min_gain: float = 0.005,
    calibration: dict | None = None,
    at_roofline: float = AT_ROOFLINE,
) -> Analysis:
    ctx = context(capture)
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
        small = [d for d in ds if d.small]
        total = sum(d.gain for d in ds if d.enabled or d.small)
        if small and total >= min_gain:
            for d in small:
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
        if rank > top:
            for d in e.decisions:
                d.enabled = False
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
            "counters_pct": _counted(capture.workloads, ctx["counters"], ctx["shared"]),
        },
    )
