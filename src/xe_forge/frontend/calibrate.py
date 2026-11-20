"""Measure the card's read bandwidth and matmul throughput per dtype for roofline bounds.

An unmeasurable value is ``None`` with its reason in ``unmeasured``; never 0.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from xe_forge import frontend

CACHE = frontend.CACHE / "calibration"
DTYPES = ("bfloat16", "float16", "float32")
# Float formats whose every value is exactly a value of these: a matmul in them is never bound
# to be slower than one in those (convert, then multiply), whatever the card's own kernel does.
# Not int8: its int32 result is not one a 16-bit float matmul computes.
EXACT_IN = dict.fromkeys(("float8_e4m3fn", "float8_e5m2"), ("bfloat16", "float16"))
# Matmul dtypes measured by their own kernel, not torch.matmul.
SPECIAL = ("int8", "float8_e4m3fn")


def peaks(calib: dict | None) -> tuple[float | None, dict[str, float]]:
    """(read bandwidth GB/s, {dtype: matmul TFLOP/s}) of a calibration, unmeasured left out;
    a dtype held exactly in a faster one takes that one's peak."""
    calib = calib or {}
    tflops = {k: v for k, v in (calib.get("matmul_tflops") or {}).items() if v}
    for d, wider in EXACT_IN.items():
        best = max([tflops.get(d) or 0, *(tflops.get(w) or 0 for w in wider)])
        if best:
            tflops[d] = best
    return calib.get("bandwidth_gbs") or None, tflops


def peak_of(tflops: dict[str, float], dtype, ctx: dict | None = None) -> float | None:
    """``dtype``'s measured matmul peak, else None: its compute bound is unknown, and one the
    calibration lacks is noted in ``ctx["unpeaked"]`` for the agent reading the bounds."""
    if dtype in tflops:
        return tflops[dtype]
    if ctx is not None and dtype:
        ctx.setdefault("unpeaked", set()).add(dtype)
    return None


def unpeaked_note(ctx: dict) -> list[str]:
    return [f"{d} has no measured matmul peak in the calibration, so its compute bound is "
            "unknown: its GEMMs and costs are bounded by bandwidth alone, so their "
            "efficiency reads low when compute is what limits them"
            for d in sorted(ctx.get("unpeaked", ()))]  # fmt: skip


def measurable(name: str) -> bool:
    """Whether :func:`measure` has a matmul in dtype ``name``."""
    import torch

    dt = getattr(torch, name, None)
    return name in SPECIAL or (isinstance(dt, torch.dtype) and dt.is_floating_point)


def _identity(device: dict) -> tuple:
    return device.get("type"), device.get("name")


def other_device(calib: dict | None, run: dict) -> str | None:
    """A note when the calibration names another device than the capture, or None. A note,
    not a refusal: the capture's device is the launcher's own words."""

    def device(d) -> dict:
        return d if isinstance(d, dict) else {}

    mine, theirs = device((calib or {}).get("device")), device(run.get("device"))
    # Names only: one card is reported under different device types by different stacks.
    if mine.get("name") and theirs.get("name") and mine["name"] != theirs["name"]:
        return (f"the calibration was measured on {mine.get('type')} {mine['name']}, the "
                f"capture ran on {theirs.get('type')} {theirs['name']}: its bounds are that "
                "device's")  # fmt: skip
    return None


def complete(calib: dict, dtypes, measure_fn=None) -> dict:
    """``calib`` with a matmul peak measured for each of ``dtypes`` it has no entry for (a
    dtype held exactly in a measured one has its peak). ``measure_fn(dtypes)`` returns a
    :func:`measure` record, or None when the measurement did not finish.

    A dtype the record gives a reason for, or that has no matmul to measure, is kept as
    ``None`` with that reason in ``unmeasured``, so it is not tried again; one that failed
    for no reason of its own (no device, the measurement stopped) is left out, to be tried
    on the next run. A record of another device is not merged at all."""
    have, tried = peaks(calib)[1], calib.get("matmul_tflops") or {}
    missing = sorted({d for d in dtypes if d and d not in have and d not in tried})
    if not missing:
        return calib
    todo = [d for d in missing if measurable(d)]
    rec = (measure_fn or (lambda ds: measure(dtypes=tuple(ds))))(todo) if todo else {}
    if rec is None:
        print(f"calibrate: the measurement of {', '.join(todo)} did not finish; their "
              "compute bound stays unknown", file=sys.stderr)  # fmt: skip
        rec = {}
    if (
        rec.get("device")
        and calib.get("device")
        and _identity(rec["device"]) != _identity(calib["device"])
    ):
        print(f"calibrate: measured on {rec['device']}, not the calibration's "
              f"{calib['device']}; not merged, the compute bound of {', '.join(missing)} "
              "stays unknown", file=sys.stderr)  # fmt: skip
        return calib
    out = {**calib, "matmul_tflops": dict(tried), "unmeasured": dict(calib.get("unmeasured") or {})}
    why, changed = rec.get("unmeasured") or {}, False
    for d in missing:
        key = f"matmul_tflops.{d}"
        value = (rec.get("matmul_tflops") or {}).get(d)
        reason = why.get(key) or (None if d in todo else "no matmul in this dtype to measure")
        if value:
            out["matmul_tflops"][d] = value
            out["unmeasured"].pop(key, None)
        elif reason:
            out["matmul_tflops"][d], out["unmeasured"][key] = None, reason
        changed = changed or bool(value or reason)
    return out if changed else calib


def _time_us(fn, dev, iters: int) -> float:
    import torch

    mod = getattr(torch, dev)
    for _ in range(3):
        fn()
    mod.synchronize()
    times = []
    for _ in range(iters):
        start, end = mod.Event(enable_timing=True), mod.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        mod.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return min(times)  # a peak is the best the card did; a median moves with clocks


def measure(
    read_bytes: int = 1 << 30, n: int = 8192, iters: int = 20, dtypes: tuple | None = None
) -> dict:
    """``read_bytes`` must exceed the card's caches; ``n`` is the square matmul size."""
    import torch

    from xe_forge.frontend.profile import device, device_info, device_type

    dev = device_type()
    rec: dict = {
        "device": device_info(),
        "bandwidth_gbs": None,
        "matmul_tflops": {},
        "unmeasured": {},
    }
    if dev == "cpu":
        rec["unmeasured"]["all"] = "no GPU visible to torch"
        return rec
    index = torch.device(device()).index
    if index is not None:  # tensors below are allocated on the current device
        getattr(torch, dev).set_device(index)
    try:
        # Random data: a constant fill compresses and reads faster than real tensors.
        x = torch.rand(read_bytes // 2, dtype=torch.float16, device=dev)
        us = _time_us(lambda: x.sum(), dev, iters)
        rec["bandwidth_gbs"] = round(x.numel() * x.element_size() / us / 1e3, 1)
        del x
    except Exception as exc:
        rec["unmeasured"]["bandwidth_gbs"] = repr(exc)

    def fp(dt):
        a, b = (torch.randn(n, n, device=dev, dtype=dt) for _ in range(2))
        return lambda: torch.matmul(a, b)

    def int8():
        a, b = (torch.randint(-64, 64, (n, n), device=dev, dtype=torch.int8) for _ in range(2))
        return lambda: torch._int_mm(a, b)

    def fp8():
        one = torch.ones((), device=dev)
        a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()
        return lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    special = {"int8": int8, "float8_e4m3fn": fp8}
    # Any other floating dtype asked for is measured the same way as the defaults.
    names = [n for n in dtypes if n not in special] if dtypes else list(DTYPES)
    kernels = {**{name: (lambda name=name: fp(getattr(torch, name))) for name in names},
               **special}  # fmt: skip
    for name, make in kernels.items():
        if dtypes and name not in dtypes:
            continue
        if not measurable(name):
            rec["matmul_tflops"][name] = None
            rec["unmeasured"][f"matmul_tflops.{name}"] = "not a floating torch dtype"
            continue
        try:
            us = _time_us(make(), dev, max(1, iters // 2))
            rec["matmul_tflops"][name] = round(2 * n**3 / us / 1e6, 1)
        except Exception as exc:
            rec["matmul_tflops"][name] = None
            rec["unmeasured"][f"matmul_tflops.{name}"] = repr(exc)
    return rec


def default_path(rec: dict) -> Path:
    name = re.sub(
        r"[^A-Za-z0-9]+", "_", str(rec["device"].get("name", rec["device"]["type"]))
    ).strip("_")
    return CACHE / f"{name}.json"


def main(
    out: Path | None, read_bytes: int = 1 << 30, n: int = 8192, iters: int = 20, dtypes=None
) -> int:
    rec = measure(read_bytes, n, iters, dtypes)
    path = out or default_path(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print(
        f"calibrate: {rec['bandwidth_gbs']} GB/s, matmul {rec['matmul_tflops']} TFLOP/s -> {path}"
    )
    return 0
