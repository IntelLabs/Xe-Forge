"""Rank captured workloads by recoverable share of device time and decide which to optimize.

A workload is skipped with one reason: at roofline, not routable, no Xe-Forge DSL, no shapes,
below --min-gain, or beyond --top. The arithmetic (bounds) is here; the reasons go to
kernels.yaml and analysis.json.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace

from xe_forge.frontend import formula
from xe_forge.frontend.calibrate import peak_of
from xe_forge.frontend.calibrate import peaks as calib_peaks
from xe_forge.frontend.canonical import resolve_dims
from xe_forge.frontend.dtypes import storage_bytes
from xe_forge.frontend.ir import CaptureRun, Naming, Workload
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
    unpeaked: list[str] = field(default_factory=list)  # dtypes its bound had no peak for


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
    extra: dict = field(default_factory=dict)  # further manifest fields (a fusion's reference)


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
UNKNOWN_DTYPE = "unknown dtype"  # what a bound with flops and no dtype is noted under
NOISE = 0.1  # a bound this far above its measurement is timer noise; further is an error


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
    ctx = {} if ctx is None else ctx  # the caller's: it collects ctx["unpeaked"]
    if not calib or not {"M", "N", "K"} <= set(w.dims) or "var" in (w.dims["N"], w.dims["K"]):
        return None
    dtype = w.dtypes[0] if w.dtypes else None
    size = storage_bytes(dtype)
    bw, peaks = calib_peaks(calib)
    if not bw:
        return None
    topk = ctx.get("topk")
    parts = w.attrs.get("parts") or [
        {**w.attrs, "N": w.dims["N"], "K": w.dims["K"], "E": w.dims.get("E")}
    ]

    def part_time(p: dict, m) -> float | None:
        if "w_bytes" not in p:
            return None
        a_b, w_b, o_b = p.get("a_bytes", size), p["w_bytes"], p.get("o_bytes", size)
        peak = peak_of(peaks, p.get("compute", dtype) or UNKNOWN_DTYPE, ctx)
        rows, e_eff = m, 1
        if p.get("grouped"):
            e = p.get("E")
            if e is None and ctx.get("experts"):
                e = ctx["experts"] / (ctx.get("ep") or 1)
            if e is None:
                return None  # never assume E=1
            rows = m * (topk or 1) if p.get("rows") == "token" else m
            # Measured, else what is certain: at least top-k experts are read. No routing
            # model: a stated cost (the naming agent's) is the way to say more.
            e_eff = w.attrs.get("active_experts") or min(e, topk or 1)
        n, k = p["N"], p["K"]
        moved = rows * k * a_b + e_eff * k * n * w_b + rows * n * o_b
        # No measured peak: the compute bound is unknown, so the bound is the bandwidth's.
        return max(moved / (bw * 1e3), 2 * rows * n * k / (peak * 1e6) if peak else 0.0)

    def per_call(m):
        m = w.dims["M"] if m is None else m
        times = [part_time(p, m) for p in parts]
        return None if None in times else sum(times)

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
    # No dtype to look up is no peak either, and noted like one the calibration lacks.
    peak = peak_of(peaks, dtype or UNKNOWN_DTYPE, ctx) if cost.get("flops") else None

    def per_call(value):
        dims = resolve_dims(w, value)
        moved = formula.evaluate(cost["bytes"], dims)
        if moved is None:
            return None
        flops = formula.evaluate(cost["flops"], dims) if cost.get("flops") else 0.0
        if flops is None:
            return None
        return max(moved / (bw * 1e3), (flops / (peak * 1e6)) if flops and peak else 0.0)

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
    return combine([v for k, v in measured.items() if k.startswith(name)])


# Each rate and the time it was measured over (a capture without the time: all of ``us``).
RATES = {"gb_per_s": "gb_us", "util_pct": "util_us"}


def _rate_us(k: dict, rate: str) -> float:
    """The time ``k``'s ``rate`` was measured over; 0 when it was not measured."""
    if k.get(rate) is None:
        return 0.0
    return k.get(RATES[rate], k.get("us")) or 0.0


def combine(hits: list[dict]) -> dict | None:
    """Counters of several launches of one kernel as one, each rate weighted by the time it
    was measured over, and that time kept beside it."""
    if not sum(v["us"] for v in hits):
        return None
    out = {"us": sum(v["us"] for v in hits)}
    for f, t in RATES.items():  # None = not measured, averaged over those that were
        ns = sum(_rate_us(v, f) for v in hits)
        out[f] = sum(v[f] * _rate_us(v, f) for v in hits if _rate_us(v, f)) / ns if ns else None
        out[t] = ns
    return out


def _covered(k: dict) -> float:
    """The time over which both of ``k``'s rates were measured, at most."""
    return min(_rate_us(k, f) for f in RATES)


def _sol_parts(w: Workload, calib: dict | None, ctx: dict | None) -> tuple | None:
    """(SOL, memory SOL, compute SOL); None when most of the time is unmeasured."""
    measured = (ctx or {}).get("counters") or {}
    ran = (ctx or {}).get("unitrace_kernels") or {}
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
        # kernel's measured time describe a few launches, not the kernel. Judge that against
        # unitrace's own total for the kernel in the pass the counters came from -- `us` is
        # the trace's, from the timed pass, which --trace-steps makes the longer of the two.
        ceiling = ((_measured(ran, name) or {}).get("us") if ran else None) or us
        # Each rate on its own: a rate measured over a few of the launches says no more than
        # counters that sampled a few. An unknown side (or no bandwidth peak) is not 0% of
        # peak, so such a kernel counts as unmeasured: SOL is the max of both sides.
        if not k or not us or _covered(k) < 0.5 * ceiling or not bw:
            continue
        memory, compute = k["gb_per_s"] / bw, k["util_pct"] / 100
        sol += min(1.0, max(memory, compute)) * us
        mem, comp, weight = mem + memory * us, comp + compute * us, weight + us
    if not weight or weight < 0.5 * sum(kernels.values()):
        return None
    return sol / weight, mem / weight, comp / weight


def peak_dtypes(capture: CaptureRun) -> set[str]:
    """The dtypes whose matmul peak a bound of this capture may read: every dtype of its ops'
    arguments and its workloads (a GEMM's compute dtype included) that has a matmul to
    measure."""
    from xe_forge.frontend.calibrate import measurable

    out = {a.dtype for op in capture.ops for a in op.args if a}
    for w in capture.workloads:
        out |= set(w.dtypes)
        out |= {p.get("compute") for p in w.attrs.get("parts") or [w.attrs]}
    return {d for d in out if d and measurable(d)}


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
    ctx = {} if ctx is None else ctx
    mine = {**ctx, "unpeaked": set()}  # this workload's, then added to the run's
    bounded = bound(w, calib, mine)
    d.unpeaked = sorted(mine["unpeaked"])
    ctx.setdefault("unpeaked", set()).update(d.unpeaked)
    if bounded is not None:
        d.eff = round(bounded[1], 3)
        d.gain = w.share * (bounded[0] / w.device_us) if w.device_us else 0.0
        # Without its compute bound, nearness to the bandwidth's says nothing of the roofline.
        if d.eff >= at_roofline and not d.unpeaked:
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

    def bounds(name, us):
        k = None if name in shared else _measured(measured, name)
        return bool(k) and _covered(k) >= 0.5 * us

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
        "unitrace_kernels": capture.run.get("unitrace_kernels") or {},
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

    # A kernel reached through a library call is also that call, tuned as a library route:
    # the same captured shapes, decided again for the route.
    calls = [replace(w, naming=Naming("library", via=str(w.naming.call.get("via", "")),
                                      tolerance=w.naming.tolerance,
                                      **{k: w.naming.call.get(k) for k in ("name", "repo", "dsl")}))
             for w in capture.workloads if w.naming.route == "kernel" and w.naming.call]  # fmt: skip
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for w in capture.workloads:
        n = w.naming
        groups[(n.route, n.repo, n.name)].append((w, decisions[w.workload_id]))
    for w in calls:
        n = w.naming
        groups[(n.route, n.repo, n.name)].append(
            (w, _decide(w, min_gain, calibration, ctx, at_roofline)))  # fmt: skip

    entries = []
    for (route, repo, name), pairs in groups.items():
        ws, ds = [w for w, _ in pairs], [d for _, d in pairs]
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
        e.reason = f"{share:.2%} of device time; {round(calls, 3)} calls;" + (
            bound if e.enabled else ds[0].reason
        )
        unpeaked = sorted({t for d in ds for t in d.unpeaked})
        if unpeaked:
            e.reason += (f"; {', '.join(unpeaked)} has no measured matmul peak, so its compute "
                         "bound is unknown (bounded by bandwidth alone)")  # fmt: skip
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
            "unpeaked": sorted(ctx.get("unpeaked", ())),
        },
    )
