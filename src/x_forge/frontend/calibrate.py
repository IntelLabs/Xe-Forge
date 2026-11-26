"""Measure the device's read bandwidth and matmul throughput per dtype for roofline bounds.

An unmeasurable value is ``None`` with its reason in ``unmeasured``; never 0.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from x_forge import frontend

CACHE = frontend.CACHE / "calibration"
DTYPES = ("bfloat16", "float16", "float32")
# Measured on a host only when asked for (``--dtypes``): timing one there can take longer
# than the rest of the calibration together.
HOST_ON_REQUEST = ("float8_e4m3fn",)
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


def _placement(device: dict) -> str | None:
    """A host placement's tag, or None for a device that has none."""
    from x_forge.core.topology import placement_tag

    p = device.get("placement")
    if not (isinstance(p, dict) and p.get("cpus")):
        return None
    # The thread count is part of what was measured, as much as the CPUs it ran on.
    return f"{placement_tag(p)}_threads{device.get('threads', 'unknown')}"


def _identity(device: dict) -> tuple:
    # On a host, the CPUs and memory nodes a run may use are part of what was measured.
    return device.get("type"), device.get("name"), _placement(device)


def _where(device: dict) -> str:
    p = device["placement"]
    return (f"CPUs {p['cpus']} (memory nodes {p.get('memory_nodes')}, policy "
            f"{p.get('memory_policy') or 'unknown'}) with {device.get('threads', 'unknown')} "
            "threads")  # fmt: skip


def _short(device: dict) -> str:
    """A device as one line: its type and name, and on a host its placement."""
    tag = _placement(device)
    return f"{device.get('type')} {device.get('name')}" + (f" ({tag})" if tag else "")


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
    if _placement(mine) and _placement(theirs) and _placement(mine) != _placement(theirs):
        return (
            f"the calibration was measured under {_where(mine)}, the capture ran under "
            f"{_where(theirs)}: its bounds are that placement's"
        )
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
        print(f"calibrate: measured on {_short(rec['device'])}, not the calibration's "
              f"{_short(calib['device'])}; not merged, the compute bound of {', '.join(missing)} "
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

    if dev == "cpu":  # host work: done when the call returns
        import time

        for _ in range(3):
            fn()
        times = []
        for _ in range(iters):
            t = time.perf_counter_ns()
            fn()
            times.append((time.perf_counter_ns() - t) / 1e3)
        return min(times)
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
    """``read_bytes`` must exceed the device's caches; ``n`` is the square matmul size.

    On a host, everything is measured under the placement the process inherited, and read
    bandwidth twice: from a buffer first written by one thread (its pages on that thread's
    memory node) and by all threads (pages spread over the nodes they run on)."""
    import torch

    from x_forge.frontend.profile import device, device_info, device_type

    dev = device_type()
    rec: dict = {
        "device": device_info(),
        "bandwidth_gbs": None,
        "matmul_tflops": {},
        "unmeasured": {},
    }
    index = torch.device(device()).index
    if dev != "cpu" and index is not None:  # tensors below are allocated on the current device
        getattr(torch, dev).set_device(index)

    def read_gbs(x) -> float:
        us = _time_us(lambda: x.sum(), dev, iters)
        return _significant(x.numel() * x.element_size() / us / 1e3)

    read_dtype = torch.float32  # one dtype on every device, so the bytes read mean the same
    # How bandwidth was read, so a record made another way can be told apart.
    rec["bandwidth_dtype"], rec["read_bytes"] = str(read_dtype).removeprefix("torch."), read_bytes
    width = torch.tensor([], dtype=read_dtype).element_size()
    llc = (rec["device"].get("placement") or {}).get("llc_bytes")
    if llc and read_bytes <= llc:
        rec["unmeasured"]["bandwidth_gbs"] = (
            f"read_bytes {read_bytes} fits in the {llc} bytes of last-level cache of the CPUs "
            "this run may use: it would time the cache"
        )
    else:

        def buffer(writers: int | None = None):
            # Random data: a constant fill compresses and reads faster than real tensors.
            if dev != "cpu":
                return torch.rand(read_bytes // width, dtype=read_dtype, device=dev)
            # On a host, pages are placed by the first write: zero_ runs on the thread pool,
            # so it places them as that many writers would; the random fill then reuses them.
            threads = torch.get_num_threads()
            try:
                torch.set_num_threads(writers or threads)
                x = torch.empty(read_bytes // width, dtype=read_dtype).zero_()
            finally:
                torch.set_num_threads(threads)
            return x.uniform_()

        try:
            x = buffer()
            rec["bandwidth_gbs"] = read_gbs(x)
            del x
        except Exception as exc:
            rec["unmeasured"]["bandwidth_gbs"] = repr(exc)
        if dev == "cpu":
            if torch.get_num_threads() == 1:
                rec["bandwidth_gbs_one_writer"] = None
                rec["unmeasured"]["bandwidth_gbs_one_writer"] = (
                    "one thread: the same buffer as bandwidth_gbs"
                )
            else:
                try:
                    x = buffer(writers=1)
                    rec["bandwidth_gbs_one_writer"] = read_gbs(x)
                    del x
                except Exception as exc:
                    rec["unmeasured"]["bandwidth_gbs_one_writer"] = repr(exc)

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
        if not dtypes and dev == "cpu" and name in HOST_ON_REQUEST:
            # Not a value: left out, so a run that needs it measures it (``--dtypes``).
            rec.setdefault("not_requested", {})[f"matmul_tflops.{name}"] = (
                f"not measured by default on a host; pass --dtypes {name} to measure it"
            )
            continue
        if not measurable(name):
            rec["matmul_tflops"][name] = None
            rec["unmeasured"][f"matmul_tflops.{name}"] = "not a floating torch dtype"
            continue
        try:
            us = _time_us(make(), dev, max(1, iters // 2))
            rec["matmul_tflops"][name] = _significant(2 * n**3 / us / 1e6)
        except Exception as exc:
            rec["matmul_tflops"][name] = None
            rec["unmeasured"][f"matmul_tflops.{name}"] = repr(exc)
    return rec


def _significant(value: float) -> float:
    """Three significant digits: a slow rate stays a small number, never rounds to 0."""
    return float(f"{value:.3g}")


def default_path(rec: dict) -> Path:
    name = re.sub(
        r"[^A-Za-z0-9]+", "_", str(rec["device"].get("name", rec["device"]["type"]))
    ).strip("_")
    tag = _placement(rec["device"])
    if not tag:
        return CACHE / f"{name}.json"
    # The full placement is in the record; a hash keeps a long CPU list within a file name.
    return CACHE / f"{name}_{hashlib.sha256(tag.encode()).hexdigest()[:12]}.json"


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
