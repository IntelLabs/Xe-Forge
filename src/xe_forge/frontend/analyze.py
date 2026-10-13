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
``buildable``  its DSL is one Xe-Forge writes (``models.DSL``)
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

from xe_forge.frontend.canonical import resolve_dims
from xe_forge.frontend.dtypes import storage_bytes, tensor_bytes
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
    eff: float | None = None  # fraction of its roofline bound, when calibrated


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
    target: str = "kernel"  # or "library": tune how the library is called, not a kernel


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


# At or above this fraction of its roofline bound a GEMM has nothing left worth an agent run.
AT_ROOFLINE = 0.8  # default of --at-roofline
NOISE = 0.1  # a bound this far above its measurement is noise; further is an error


def experts_touched(experts: int, rows: int, topk: int | None) -> float:
    """Expected distinct experts a grouped GEMM reads, under uniform routing.

    ``rows`` = tokens x top-k, so tokens B = rows / k and E_eff = E * (1 - (1 - k/E)^B): about
    rows when rows << E, all E as rows grow. Skewed routing touches fewer; the per-point
    clamp (a bound never above what was measured) keeps that from inventing headroom.
    Without top-k, k = 1 (rows routed independently), which slightly overestimates."""
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
    """Sum a per-call bound over the captured points into (recoverable us, efficiency).

    A point whose bound exceeds what was measured by up to ``NOISE`` is capped at the
    measurement (timer noise): no headroom rather than invented headroom. A bound built on
    an estimate (uniform routing) comes as (expected, guaranteed); where the measurement
    beats the estimate, the guaranteed one is used. Beyond that the bound is impossible --
    an input to it is wrong -- and the bound is unknown, never "at roofline"."""
    recoverable = reached_total = 0.0
    for value, calls, us in _points(w):
        bound = per_call_bound(value)
        if isinstance(bound, tuple):  # (expected, guaranteed): an estimate may be beaten
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
    """Per token count M: max(bytes / bandwidth, 2*M*N*K / peak[compute dtype]), with
    bytes = M*K*a + E_eff*K*N*w + M*N*o, summed over the weights the op multiplies by. ``w``/``a`` are bytes per element measured from the
    captured tensors (scales and packing included); E_eff is the experts a grouped call
    touches -- measured (``active_experts``) when an adapter supplies it, else the uniform-
    routing estimate over the kernel's local E, else the model's experts / expert-parallel
    size. A grouped call with no E at all has no bound: never E=1."""
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
    # One part per weight the op multiplies by (the agent's gemm list); usually one.
    parts = w.attrs.get("parts") or [
        {**w.attrs, "N": w.dims["N"], "K": w.dims["K"], "E": w.dims.get("E")}
    ]

    def part_time(p: dict, m, sure: bool = False) -> float | None:
        if "w_bytes" not in p:
            return None  # weight bytes unmeasured: no agent answer, or a weight the trace hides
        a_b, w_b, o_b = p.get("a_bytes", size), p["w_bytes"], p.get("o_bytes", size)
        # The compute dtype's measured peak; a dtype the card has no measured peak for runs
        # at most at the fastest measured one -- still a lower bound on time.
        peak = peaks.get(p.get("compute", dtype)) or max(peaks.values())
        rows, e_eff = m, 1
        if p.get("grouped"):
            e = p.get("E")
            if e is None and ctx.get("experts"):
                e = ctx["experts"] / (ctx.get("ep") or 1)
            if e is None:
                return None  # never E=1
            # The agent says whether the call's rows are tokens or token x expert rows.
            rows = m * (topk or 1) if p.get("rows") == "token" else m
            e_eff = w.attrs.get("active_experts") or (
                # guaranteed: every token touches top-k experts, so at least that many
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


def streaming_bound(
    w: Workload, calib: dict | None, ctx: dict | None = None
) -> tuple[float, float] | None:
    """Every tensor argument is read or written at least once: per token count, the bytes of
    all of them over the card's bandwidth -- a lower bound for any op, family unknown.

    Refused (None) for ops the naming agent marked ``paged`` -- a KV cache, a block or page
    table: the part they touch depends on the context, not on the shape, so counting the
    buffer whole overstates the bound and skipping it understates it."""
    bw = (calib or {}).get("bandwidth_gbs")
    if not bw or not w.tensors or w.attrs.get("paged") or w.attrs.get("args_incomplete"):
        return None
    if not any(f"{label}_D0" in w.dims for label, _ in w.tensors):
        return None  # dims named by role (a GEMM's M/N/K), not per tensor: no shapes to sum

    def per_call(value):
        dims = resolve_dims(w, value)
        total = 0
        for label, dtype in w.tensors:
            shape = [dims[f"{label}_D{j}"] for j in range(8) if f"{label}_D{j}" in dims]
            size = tensor_bytes(shape, dtype)
            if size is None:
                return None
            total += size
        return total / (bw * 1e3)

    return _against(w, per_call)


# family -> bound. A family without its own bounder gets the streaming bound; attention,
# MLA, linear attention, SSM and collectives each become one entry here when they get one.
BOUNDERS = {"gemm": gemm_bound}


def bound(w: Workload, calib: dict | None, ctx: dict | None = None) -> tuple[float, float] | None:
    """The family's bound; failing that, the streaming bound -- except for a grouped call,
    whose unknown expert count must stay unknown (streaming would count every expert)."""
    bounder = BOUNDERS.get(w.family, streaming_bound)
    result = bounder(w, calib, ctx)
    if result is None and bounder is not streaming_bound and not w.attrs.get("grouped"):
        result = streaming_bound(w, calib, ctx)
    return result


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
        gain=w.share,  # unbounded: the ceiling is the whole share
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
            "no bound for this op (paged buffers or unknown shapes)" if calib else "no calibration"
        )
        d.gates.append(("bounded", "UNKNOWN", f"{why}: ceiling taken as the full share"))
    else:
        d.gates.append(
            ("bounded", "PASS", f"eff {d.eff:.2f} of its bound; recoverable {bounded[0]:.0f} us")
        )
    if d.gain < min_gain:
        d.gates.append(("worth", "REJECT", f"G={d.gain:.4%} < {min_gain:.4%}"))
        d.reason = f"below --min-gain {min_gain:.2%}"
        return d
    d.gates.append(("worth", "PASS", f"G={d.gain:.4%} >= {min_gain:.4%}"))
    d.enabled = True
    return d


def analyze(
    capture: CaptureRun,
    top: int = 3,
    min_gain: float = 0.005,
    calibration: dict | None = None,
    at_roofline: float = AT_ROOFLINE,
) -> Analysis:
    facts = capture.run.get("model_facts") or {}  # the naming agent's: top-k, experts
    ctx = {
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
        # Worth is the kernel's, not one shape's: a kernel whose time is spread over many
        # small workloads passes when they add up.
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
            "Ceilings are bounded by the calibrated card (GEMMs: bandwidth and matmul peak; other "
            "ops: bytes moved); an op with no bound takes its whole share."
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
