"""Several rows of one run (``x-forge run --rows rows.yaml``): the capture agent's launcher
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

from x_forge.frontend import canonical, ir, naming

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


def env(row: dict) -> dict[str, str]:
    """The launcher's environment that runs it at ``row``'s batch and lengths."""
    return {f"XF_{k.upper()}": str(row[k]) for k in KEYS}


def scales(caps: list[ir.CaptureRun], rows: list[dict]) -> list[float]:
    """Each row's scale in :func:`merge`: w_r x T / T_r, T the anchor's total device time."""
    total = caps[0].run["total_device_us"]
    return [r["weight"] * total / (c.run["total_device_us"] or 1.0)
            for c, r in zip(caps, rows, strict=True)]  # fmt: skip


def of(run_dir: Path) -> tuple[list[dict], list[float]] | None:
    """The run's rows and the scales its merged capture gave them, or None for a run of one
    row. A later measurement merges its own captures with these scales, so a row that got
    faster moves the total, and both arms of a comparison count the same calls."""
    merged = run_dir / "rows" / "capture.json"
    if not merged.is_file():
        return None
    rows = ir.load(merged).run["runtime"]["rows"]
    paths = [run_dir / "capture.json"]
    paths += [run_dir / "rows" / f"{i:02d}" / "capture.json" for i in range(2, len(rows) + 1)]
    if missing := [str(p) for p in paths if not p.is_file()]:
        raise RuntimeError(f"the run's rows have no capture at {', '.join(missing)}")
    return rows, scales([ir.load(p) for p in paths], rows)


def schedule(repeats: int, n_rows: int):
    """(repeat, row index, arm) of a two-arm measurement over rows: repeats outermost, the two
    arms at one row back to back, the arm that runs first alternating with the repeat and the
    row, so drift during a long row falls on both. With one row this is ABBA."""
    for r in range(1, repeats + 1):
        for i in range(n_rows):
            yield from ((r, i, arm) for arm in (("A", "B") if (r + i) % 2 else ("B", "A")))


def _signature(op: ir.CapturedOp) -> tuple:
    """What an op is, independent of where a parse ranked it: its op, name, inputs and calls."""
    return (str(op.framework_op), str(op.framework_meta.get("name_key", "")), repr(op.args),
            repr(op.scalars), op.call_site, op.calls)  # fmt: skip


def _host(caps: list[ir.CaptureRun], scales: list[float]) -> dict:
    """The anchor's host record with each wall-clock time the rows' sum at ``scales``, as their
    device times are; a time some row did not measure is unknown, and the record says which."""
    host = copy.deepcopy(caps[0].run["host"])
    for key in ("window_us", "node_us", "plain_us"):
        got = [(c.run.get("host") or {}).get(key) for c in caps]
        if any(g is None for g in got):
            host[key] = None
            missing = [str(r) for r, g in enumerate(got, 1) if g is None]
            host[f"{key}_why"] = f"row {', '.join(missing)} did not measure it"
        else:
            host[key] = round(sum(s * g for s, g in zip(scales, got, strict=True)), 3)
    return host


def merge(caps: list[ir.CaptureRun], rows: list[dict],
          given: list[float] | None = None) -> ir.CaptureRun:  # fmt: skip
    """One capture of ``rows`` (anchor first, its capture first): row r's calls and times
    scaled by w_r x T / T_r (T the anchor's total device time), which keeps each op's time
    per call; op ids prefixed ``rNN:``; each kernel's counters those of every row, by time,
    and its unitrace time the sum of every row's, scaled the same. ``given``: the scales of
    an earlier merge (:func:`of`) instead, and the total then the rows' scaled totals."""
    from x_forge.frontend.analyze import RATES, combine

    total = caps[0].run["total_device_us"]
    if given is not None:
        total = sum(s * c.run["total_device_us"] for s, c in zip(given, caps, strict=True))
    ops: list[ir.CapturedOp] = []
    attributed = 0.0
    counters: dict[str, list[dict]] = {}
    traced: dict[str, dict] = {}
    used: list[float] = []
    for r, (cap, row) in enumerate(zip(caps, rows, strict=True), 1):
        scale = (given[r - 1] if given is not None
                 else row["weight"] * total / (cap.run["total_device_us"] or 1.0))  # fmt: skip
        used.append(scale)
        attributed += scale * cap.run.get("attributed_device_us", 0.0)
        for name, c in (cap.run.get("counters") or {}).items():
            counters.setdefault(name, []).append(
                {**c, **{t: c[t] * scale for t in ("us", *RATES.values()) if c.get(t) is not None}}
            )
        for name, k in (cap.run.get("unitrace_kernels") or {}).items():
            seen = traced.setdefault(name, {**k, "us": 0.0})
            seen["us"] += k["us"] * scale
        # Given scales compare two measurements: their ops in an order the work decides, not
        # the parse's ranking by time, so the same work sums to the same calls to the bit.
        for op in sorted(cap.ops, key=_signature) if given is not None else cap.ops:
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
    if traced:
        run["unitrace_kernels"] = traced
    if run.get("host"):
        run["host"] = _host(caps, used)
    run["attributed_pct"] = round(100 * attributed / total, 2) if total else 0.0
    merged = ir.CaptureRun(run=run, ops=ops)
    merged.workloads = canonical.workloads(ops, total, naming.name_op)
    from x_forge.skills.capture import precision_record

    run["precision"] = precision_record(merged)
    return merged
