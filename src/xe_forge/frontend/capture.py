"""``raw/`` -> ``capture.json``: what ``xe-forge-skill capture parse`` builds from a launcher's run.

A torch framework's ops come from its torch.profiler trace; any other framework's from the
op map ``capture parse`` already checked against unitrace (``summary``).
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from xe_forge.frontend import canonical, ir, naming
from xe_forge.frontend.torch_trace import parse_traces


def build(raw_dir: Path, out: Path, names: dict | None = None, summary=None) -> ir.CaptureRun:
    """``raw/`` (run.json + traces) -> capture.json, with ops named from ``names``."""
    run = json.loads((raw_dir / "run.json").read_text())
    traces = sorted(raw_dir.glob("trace*.json*"))
    ranks: dict[int, list[Path]] = {}
    for t in traces:
        found = re.search(r"rank(\d+)", t.name)
        ranks.setdefault(int(found.group(1)) if found else 0, []).append(t)
    if summary is None and traces:
        # TP ranks run the same shapes: rank 0 is the workload, per-rank totals show imbalance.
        summary = parse_traces(ranks[min(ranks)])
        if len(ranks) > 1:
            run["ranks"] = {
                r: round(parse_traces(ts).total_us, 1) for r, ts in sorted(ranks.items())
            }
    elif summary is None:
        raise SystemExit(f"no trace in {raw_dir}")

    if not summary.launches:
        raise SystemExit(f"capture: no device kernel in {raw_dir}; nothing ran on the device")
    run["total_device_us"] = round(summary.total_us, 1)
    run["attributed_device_us"] = round(summary.attributed_us, 1)
    run["attributed_pct"] = round(summary.attributed_pct, 2)
    run["kernel_invocations"] = summary.launches
    run["raw"] = [
        os.path.relpath(p, out.parent) for p in sorted(raw_dir.iterdir()) if p.name != "run.json"
    ]
    _annotate_schemas(summary.ops, raw_dir / "schemas.json")
    key = naming.key
    entries = (names or {}).get("ops") or {}
    ops = []
    for op in summary.ops:
        op.framework_meta["name_key"] = key(op)
        if key(op) in entries:
            op.framework_meta["named"] = entries[key(op)]
        ops.extend(_split(op))
    summary.ops = ops
    if names is not None:
        run["model_facts"] = names.get("model") or {}
    if (raw_dir / "counters.json").is_file():
        run["counters"] = _counters(json.loads((raw_dir / "counters.json").read_text()))
    capture = ir.CaptureRun(run=run, ops=summary.ops)
    capture.workloads = canonical.workloads(capture.ops, summary.total_us, naming.name_op)
    ir.dump(capture, out)
    return capture


_SIMD = re.compile(r"\[SIMD\d+ \{\d+;\s*\d+;\s*\d+\} \{\d+;\s*\d+;\s*\d+\}\]$")


def _counters(summary: dict) -> dict:
    """unitrace ComputeBasic records -> {kernel: {gb_per_s, util_pct, us}}, time-weighted.

    compute_metrics reports an incomplete counter as None (unknown); a rate is averaged over
    the records that measured it, and stays None when none did -- never 0.
    """
    acc: dict = {}
    for r in summary.get("records") or []:
        time_ns = r.get("time_ns")
        if not time_ns:
            continue
        name = _SIMD.sub("", r["kernel"])
        a = acc.setdefault(
            name, {"ns": 0.0, "bytes": 0.0, "bytes_ns": 0.0, "util": 0.0, "util_ns": 0.0}
        )
        a["ns"] += time_ns
        moved = [((r.get("memory") or {}).get(d) or {}).get("bytes") for d in ("read", "write")]
        if None not in moved:
            a["bytes"] += sum(moved)
            a["bytes_ns"] += time_ns
        utils = [v for k, v in (r.get("counters") or {}).items()
                 if k.endswith("_UTILIZATION[%]") and v is not None]  # fmt: skip
        if utils:
            a["util"] += max(utils) * time_ns
            a["util_ns"] += time_ns
    return {
        k: {"gb_per_s": round(a["bytes"] / a["bytes_ns"], 3) if a["bytes_ns"] else None,
            "util_pct": round(a["util"] / a["util_ns"], 2) if a["util_ns"] else None,
            "us": round(a["ns"] / 1e3, 3)}
        for k, a in acc.items()
    }  # fmt: skip


def _split(op) -> list:
    """Split an op into one row per kernel group in ``named["kernels"]``."""
    named = op.framework_meta.get("named") or {}
    groups = named.get("kernels")
    if not isinstance(groups, dict) or not groups:
        return [op]
    by: dict = {}
    for sym, us in op.kernel_symbols.items():
        sub = next((g for g in groups if g in sym), None)
        by.setdefault(sub, {})[sym] = us
    parts = []
    for sub, syms in by.items():
        part = copy.deepcopy(op)
        part.kernel_symbols = syms
        part.device_us = round(sum(syms.values()), 3)
        entry = {k: v for k, v in named.items() if k != "kernels"}
        if sub is not None and isinstance(groups[sub], dict):
            entry.update(groups[sub])
        part.framework_meta["named"] = entry
        parts.append(part)
    return parts


def _schema_args(op_name: str, nargs: int) -> list[str] | None:
    """Argument names of the overload of ``ns::op`` taking ``nargs`` arguments."""
    try:
        import torch

        ns, name = op_name.split("::", 1)
        packet = getattr(getattr(torch.ops, ns), name)
        for overload in packet.overloads():
            args = getattr(packet, overload)._schema.arguments
            if len(args) == nargs:
                return [a.name for a in args]
    except Exception:
        return None
    return None


def _annotate_schemas(ops: list[ir.CapturedOp], cache: Path) -> None:
    """Name each op's arguments from its schema, cached in ``cache`` for a re-parse elsewhere."""
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    for op in ops:
        if "::" not in op.framework_op or op.framework_op.startswith("kernel::"):
            continue
        key = f"{op.framework_op}/{len(op.args)}"
        if key not in known:
            names = _schema_args(op.framework_op, len(op.args))
            if names is None:
                continue
            known[key] = names
        op.framework_meta["arg_names"] = known[key]
    if known:
        cache.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
