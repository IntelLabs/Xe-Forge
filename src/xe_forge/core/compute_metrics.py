"""Streaming, evidence-only summaries of unitrace ComputeBasic samples."""

import csv
import math


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def metric_rows(lines):
    """Yield ``(device, row)`` for each data row of a unitrace metrics table.

    Reads both layouts unitrace writes: one header with the kernel on every
    row, and per-launch blocks where a unit-less ``Kernel, ...`` header and
    one identity row precede the block's own header and samples. A row with
    no numeric field is a header. Unknown layouts yield nothing.
    """
    device, header, identity, pending = "unknown", None, {}, None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("==="):
            device, header, identity, pending = stripped, None, {}, None
            continue
        if not stripped:
            continue
        values = [v.strip() for v in next(csv.reader([stripped], skipinitialspace=True))]
        if pending is not None:
            identity, pending = dict(zip(pending, values, strict=False)), None
        elif not any(_is_number(v) for v in values):
            if "Kernel" in values and not any("[" in v for v in values):
                pending = values
            else:
                header = values
        elif header is not None and len(values) == len(header):
            yield device, {**identity, **dict(zip(header, values, strict=True))}


def summarize_compute_metrics(lines) -> dict:
    """Aggregate within each device/kernel; keep missing counters unknown."""
    groups = {}
    for device, row in metric_rows(lines):
        if "GlobalInstanceId" not in row:
            continue
        time_column = next(
            (
                column
                for column in ("GpuTime[ns]", "GPU_TIME[ns]", "Time[ns]", "GpuDuration[ns]")
                if column in row
            ),
            None,
        )
        name = row.get("Kernel")
        if not name:
            continue
        key = (device, name)
        group = groups.setdefault(
            key,
            {
                "device": device,
                "kernel": name,
                "samples": 0,
                "launches": set(),
                "missing_launch_ids": 0,
                "counters": {},
                "time_ns": 0.0,
                "valid_time_samples": 0,
            },
        )
        group["samples"] += 1
        if row["GlobalInstanceId"]:
            group["launches"].add(row["GlobalInstanceId"])
        else:
            group["missing_launch_ids"] += 1
        duration = _number(row.get(time_column))
        if duration is not None and duration > 0:
            group["time_ns"] += duration
            group["valid_time_samples"] += 1
        for column, raw in row.items():
            if not column.endswith(("[bytes]", "[events]", "[%]")):
                continue
            counter = group["counters"].setdefault(
                column,
                {
                    "sum": 0.0,
                    "valid_samples": 0,
                    "weighted_sum": 0.0,
                    "weight_ns": 0.0,
                },
            )
            value = _number(raw)
            if value is None:
                continue
            counter["sum"] += value
            counter["valid_samples"] += 1
            if column.endswith("[%]") and duration is not None and duration > 0:
                counter["weighted_sum"] += value * duration
                counter["weight_ns"] += duration

    records = []
    for group in groups.values():
        samples = group["samples"]
        launch_ids = group.pop("launches")
        launches = len(launch_ids) if not group["missing_launch_ids"] else None
        time_ns = group["time_ns"] if group.pop("valid_time_samples") == samples else None
        counters = {}
        for column, aggregate in group["counters"].items():
            complete = aggregate["valid_samples"] == samples
            if column.endswith("[%]"):
                value = (
                    aggregate["weighted_sum"] / time_ns
                    if complete and time_ns and aggregate["weight_ns"] == time_ns
                    else None
                )
            else:
                value = aggregate["sum"] if complete else None
            counters[column] = value
        group.update({"counters": counters, "launches": launches, "time_ns": time_ns})
        group["sampled_us_per_launch"] = time_ns / launches / 1000 if time_ns and launches else None
        group["memory"] = {}
        for direction in ("READ", "WRITE"):
            amount = counters.get(f"GPU_MEMORY_BYTE_{direction}[bytes]")
            group["memory"][direction.lower()] = {
                "bytes": amount,
                "bytes_per_launch": amount / launches if amount is not None and launches else None,
                "gb_per_s": amount / time_ns if amount is not None and time_ns else None,
            }
        records.append(group)
    return {
        "group": "ComputeBasic",
        "records": records,
        "limitations": [
            "Time and per-launch amounts cover sampled intervals, not guaranteed complete launches.",
            "No launches discarded by sample-count heuristics; partial captures require inspection.",
            "Percentages are duration-weighted when all durations and values are present.",
            "Missing or invalid counters are unknown, not zero. No stall-cause or ISA attribution.",
        ],
    }
