"""Streaming, evidence-only summaries of unitrace ComputeBasic samples."""

import csv
import math


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def metric_rows(lines):
    """Yield ``(device, row)`` for each data row of a unitrace metrics table.

    Reads both layouts unitrace writes: one header with the kernel on every
    row, and per-launch blocks where a unit-less ``Kernel, ...`` header and
    one identity row precede the block's own header and samples. Headers are
    known by position, not content: the first row after a device banner or an
    identity row is the header, and every later row is a sample, whatever its
    values (a missing one may be empty or text); under a flat table's header,
    a row naming a kernel ``Kernel`` is a sample too. A section changes layout
    only at a device banner. A row of a different length
    than the header is a sample with the values it cannot place unknown: a
    longer one has every value unknown, a shorter one its last value too.
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
            identity, pending, header = dict(zip(pending, values, strict=False)), None, None
        elif (
            # Under a flat table's header, which names the kernel itself, it is a sample.
            (header is None or identity or "Kernel" not in header)
            and "Kernel" in values
            and not any("[" in v for v in values)
        ):
            pending = values  # a launch block's unit-less identity header
        elif header is None:
            header = values
        elif values != header:  # a repeated header is not a sample
            if len(values) > len(header):  # which value is which column is unknown
                values = [""] * len(header)
            elif len(values) < len(header):  # cut short: its last value may be cut too
                values = values[:-1] + [""] * (len(header) - len(values) + 1)
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
