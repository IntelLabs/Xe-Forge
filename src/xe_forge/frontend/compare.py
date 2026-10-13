"""Two arms of captures (before/after a patch), workload by workload, in one fixed format.

    xe-forge compare --a A/capture.json [A2/capture.json ...] --b B/capture.json [...] -o out/

Every framework writes the same ``capture.json``, and a workload's id depends on the op,
its shapes, dtypes and layout -- not on the kernel that ran it -- so a patched kernel keeps
its id and the comparison is a join. The output is the same for vLLM, SGLang and OpenVINO:
``compare.csv`` / ``compare.json`` (one row per workload, ``TOTAL`` first, fixed columns)
and ``compare.md``.

Stable by construction: captures of different work are refused (framework, model and the
run's batch/input/output lengths must agree), several captures per arm are reduced to
their median with the spread beside it, rows are sorted by the baseline's time, numbers
are rounded, and a workload whose call count differs between arms is flagged rather than
given a speedup it did not earn.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from xe_forge.frontend import ir

COLUMNS = [
    "framework", "model", "workload_id", "op", "kernel", "route", "status",
    "calls_a", "calls_b", "us_a", "us_b", "per_call_a", "per_call_b",
    "speedup", "delta_us", "share_a", "spread_a", "spread_b",
]  # fmt: skip
# A capture's whole ``runtime`` is its work, except what only describes how it was launched.
_NOT_WORK = ("one_batch",)


def _arm(paths: list[Path]) -> tuple[list[ir.CaptureRun], dict]:
    caps = [ir.load(p) for p in paths]
    key = lambda c: (  # noqa: E731
        c.run.get("framework"),
        (c.run.get("model") or {}).get("name"),
        tuple(
            sorted(
                (k, str(v)) for k, v in (c.run.get("runtime") or {}).items() if k not in _NOT_WORK
            )
        ),
    )
    if len({key(c) for c in caps}) != 1:
        raise SystemExit(
            f"captures in one arm differ in framework/model/run: {[key(c) for c in caps]}"
        )
    return caps, dict(zip(("framework", "model", "run"), key(caps[0]), strict=True))


def _reduce(caps: list[ir.CaptureRun]) -> tuple[dict, dict, dict]:
    """workload id -> (median us, calls, spread), plus totals and workload metadata."""
    per: dict[str, list[tuple[float, int]]] = {}
    meta: dict[str, ir.Workload] = {}
    for c in caps:
        for w in c.workloads:
            per.setdefault(w.workload_id, []).append((w.device_us, w.calls))
            meta.setdefault(w.workload_id, w)
    stats = {}
    for wid, vals in per.items():
        us = [v[0] for v in vals]
        med = statistics.median(us)
        stats[wid] = (med, vals[0][1], (max(us) - min(us)) / med if med and len(us) > 1 else 0.0)
    totals = [c.run.get("total_device_us", 0.0) for c in caps]
    med_t = statistics.median(totals)
    total = (med_t, None, (max(totals) - min(totals)) / med_t if med_t and len(totals) > 1 else 0.0)
    return stats, meta, {"TOTAL": total}


def compare(a_paths: list[Path], b_paths: list[Path]) -> list[dict]:
    a_caps, a_key = _arm(a_paths)
    b_caps, b_key = _arm(b_paths)
    if a_key != b_key:
        raise SystemExit(f"arms ran different work: a={a_key} b={b_key}")
    a, a_meta, a_tot = _reduce(a_caps)
    b, b_meta, b_tot = _reduce(b_caps)
    total_a = a_tot["TOTAL"][0]

    def row(wid, ua, ub, ca, cb, sa, sb, w):
        status = "matched"
        if ua is None:
            status = "only_b"
        elif ub is None:
            status = "only_a"
        elif ca is not None and ca != cb:
            status = "calls_differ"
        speed = round(ua / ub, 4) if status == "matched" and ub else None
        return {
            "framework": a_key["framework"], "model": a_key["model"], "workload_id": wid,
            "op": w.framework_meta.get("framework_op", "") if w else "",
            "kernel": w.naming.name if w else "", "route": w.naming.route if w else "",
            "status": status, "calls_a": ca, "calls_b": cb,
            "us_a": None if ua is None else round(ua, 3), "us_b": None if ub is None else round(ub, 3),
            "per_call_a": round(ua / ca, 4) if ua is not None and ca else None,
            "per_call_b": round(ub / cb, 4) if ub is not None and cb else None,
            "speedup": speed,
            "delta_us": round(ua - ub, 3) if ua is not None and ub is not None else None,
            "share_a": round(ua / total_a, 6) if ua is not None and total_a else None,
            "spread_a": round(sa, 4) if sa is not None else None,
            "spread_b": round(sb, 4) if sb is not None else None,
        }  # fmt: skip

    ta, tb = a_tot["TOTAL"], b_tot["TOTAL"]
    rows = [row("TOTAL", ta[0], tb[0], None, None, ta[2], tb[2], None)]
    rows[0].update(op="", kernel="", route="", status="total")
    rows[0]["speedup"] = round(ta[0] / tb[0], 4) if tb[0] else None
    rows[0]["share_a"] = 1.0
    body = []
    for wid in sorted(set(a) | set(b)):
        sa, sb = a.get(wid), b.get(wid)
        body.append(
            row(wid, sa and sa[0], sb and sb[0], sa and sa[1], sb and sb[1],
                sa and sa[2], sb and sb[2], a_meta.get(wid) or b_meta.get(wid))
        )  # fmt: skip
    body.sort(key=lambda r: (-(r["us_a"] or 0.0), -(r["us_b"] or 0.0), r["workload_id"]))
    return rows + body


def write(rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "compare.json").write_text(
        json.dumps({"columns": COLUMNS, "rows": rows}, indent=1) + "\n"
    )
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    shown = [r for r in rows if r["status"] != "matched" or (r["share_a"] or 0) >= 0.001]
    md = [
        f"# Compare: {rows[0]['model']} under {rows[0]['framework']}",
        "",
        "Arm a is the baseline. speedup = us_a / us_b; workloads under 0.1% of a are in compare.csv only.",
        "",
        "| workload | kernel | route | status | us_a | us_b | speedup | share_a |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in shown:
        md.append(
            f"| `{r['workload_id']}` | {r['kernel'] or '-'} | {r['route'] or '-'} | {r['status']} | "
            f"{fmt(r['us_a'])} | {fmt(r['us_b'])} | {fmt(r['speedup'])} | {fmt(r['share_a'])} |"
        )
    (out_dir / "compare.md").write_text("\n".join(md) + "\n")
    return out_dir / "compare.csv"
