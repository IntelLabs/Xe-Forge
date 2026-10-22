"""Measure the card's read bandwidth and matmul throughput per dtype for roofline bounds.

An unmeasurable value is ``None`` with its reason in ``unmeasured``; never 0.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from xe_forge import frontend

CACHE = frontend.CACHE / "calibration"
DTYPES = ("bfloat16", "float16")


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
    return statistics.median(times)


def measure(
    read_bytes: int = 1 << 30, n: int = 8192, iters: int = 20, dtypes: tuple | None = None
) -> dict:
    """``read_bytes`` must exceed the card's caches; ``n`` is the square matmul size."""
    import torch

    from xe_forge.frontend.adapters.pytorch import device_info, device_type

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

    kernels = {**{name: (lambda name=name: fp(getattr(torch, name))) for name in DTYPES},
               "int8": int8, "float8_e4m3fn": fp8}  # fmt: skip
    for name, make in kernels.items():
        if dtypes and name not in dtypes:
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


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


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
