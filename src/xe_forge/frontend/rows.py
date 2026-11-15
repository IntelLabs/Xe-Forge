"""Several rows of one run (``xe-forge run --rows rows.yaml``): the capture agent's launcher
rerun at each row's batch and lengths, and the rows' captures merged into one, each row's
ops scaled to its weight. The merged ops are grouped as usual, so the same op at another
row's shape is another shape of its workload: the spec holds every row's shapes.

rows.yaml: {"rows": [{"batch", "input_len", "output_len", "weight"}, ...]}, weights summing
to 1, each the row's share of the served GPU time.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from xe_forge.frontend import canonical, ir, naming

KEYS = ("batch", "input_len", "output_len")


def load(path: str | Path) -> list[dict]:
    """The rows, the anchor first: the largest batch x input, the one the capture agent runs."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty 'rows' list")
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or set(r) != {*KEYS, "weight"}:
            raise ValueError(f"{path}: row {i} must have exactly {', '.join(KEYS)} and weight")
        if not all(type(r[k]) is int and r[k] > 0 for k in KEYS):
            raise ValueError(f"{path}: row {i}: {', '.join(KEYS)} must be positive integers")
        w = r["weight"]
        if not (isinstance(w, int | float) and not isinstance(w, bool) and w > 0):
            raise ValueError(f"{path}: row {i}: weight must be a positive number")
    if len({tuple(r[k] for k in KEYS) for r in rows}) < len(rows):
        raise ValueError(f"{path}: two rows run the same batch and lengths")
    total = sum(r["weight"] for r in rows)
    if abs(total - 1) > 1e-3:
        raise ValueError(f"{path}: weights sum to {total:.4f}, not 1")
    return sorted(rows, key=lambda r: -r["batch"] * r["input_len"])


def ran(cap: ir.CaptureRun, row: dict) -> bool:
    """Whether the capture's launcher ran at the row's batch and lengths."""
    rt = cap.run.get("runtime") or {}
    return all(rt.get(k) == row[k] for k in KEYS)


def merge(caps: list[ir.CaptureRun], rows: list[dict]) -> ir.CaptureRun:
    """One capture of ``rows`` (anchor first, its capture first): row r's calls and times
    scaled by w_r x T / T_r (T the anchor's total device time), which keeps each op's time
    per call; op ids prefixed ``rNN:``; each kernel's counters those of every row, by time."""
    from xe_forge.frontend.analyze import combine

    total = caps[0].run["total_device_us"]
    ops: list[ir.CapturedOp] = []
    attributed = 0.0
    counters: dict[str, list[dict]] = {}
    for r, (cap, row) in enumerate(zip(caps, rows, strict=True), 1):
        scale = row["weight"] * total / (cap.run["total_device_us"] or 1.0)
        attributed += scale * cap.run.get("attributed_device_us", 0.0)
        for name, c in (cap.run.get("counters") or {}).items():
            counters.setdefault(name, []).append({**c, "us": c["us"] * scale})
        for op in cap.ops:
            op = copy.deepcopy(op)
            op.id = f"r{r:02d}:{op.id}"
            op.calls = op.calls * scale  # unrounded: time per call stays exact
            op.device_us = round(op.device_us * scale, 3)
            op.kernel_symbols = {k: round(us * scale, 3) for k, us in op.kernel_symbols.items()}
            op.framework_meta.update(row=r, row_scale=round(scale, 6))
            ops.append(op)
    run = copy.deepcopy(caps[0].run)
    run["runtime"]["rows"] = rows
    run["total_device_us"] = round(total, 1)
    run["attributed_device_us"] = round(attributed, 1)
    run["counters"] = {k: c for k, v in counters.items() if (c := combine(v))}
    run["attributed_pct"] = round(100 * attributed / total, 2) if total else 0.0
    merged = ir.CaptureRun(run=run, ops=ops)
    merged.workloads = canonical.workloads(ops, total, naming.name_op)
    from xe_forge.skills.capture import precision_record

    run["precision"] = precision_record(merged)
    return merged
