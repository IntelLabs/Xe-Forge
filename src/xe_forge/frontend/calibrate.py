"""What this card can do: the two numbers a GEMM's roofline bound needs.

    xe-forge calibrate [-o calibration.json]          (on a GPU node, torch with XPU)

- ``bandwidth_gbs``: read bandwidth, a reduction over a tensor far larger than any cache.
- ``matmul_tflops[dtype]``: throughput of a large square ``torch.matmul`` -- oneDNN at its
  best case, so "at the bound" means "as good as the library gets on a friendly shape".

Measured, never written down: a value that cannot be measured is ``None`` (never 0), with
the reason in ``unmeasured``, and a reader treats ``None`` as unknown. The record is cached
under ``~/.cache/xe_forge/calibration/<device>.json``; ``xe-forge analyze --calibration``
reads it.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

CACHE = Path.home() / ".cache" / "xe_forge" / "calibration"
DTYPES = ("bfloat16", "float16")


def _time_us(fn, dev, iters: int = 20) -> float:
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


def measure() -> dict:
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
        # 1 GiB of random data: a constant fill would compress, and Intel GPUs compress
        # memory traffic losslessly, so it would read faster than any real tensor.
        x = torch.rand(1 << 29, dtype=torch.float16, device=dev)
        us = _time_us(lambda: x.sum(), dev)
        rec["bandwidth_gbs"] = round(x.numel() * x.element_size() / us / 1e3, 1)
        del x
    except Exception as exc:  # recorded, not swallowed: the reason says whether to retry
        rec["unmeasured"]["bandwidth_gbs"] = repr(exc)
    for name in DTYPES:
        try:
            n = 8192
            a = torch.randn(n, n, device=dev, dtype=getattr(torch, name))
            b = torch.randn(n, n, device=dev, dtype=getattr(torch, name))
            us = _time_us(lambda a=a, b=b: torch.matmul(a, b), dev, iters=10)
            rec["matmul_tflops"][name] = round(2 * n**3 / us / 1e6, 1)
        except Exception as exc:
            rec["matmul_tflops"][name] = None
            rec["unmeasured"][f"matmul_tflops.{name}"] = repr(exc)
    try:  # int8 activations (W4A8, W8A8 int) multiply on XMX int8
        a = torch.randint(-64, 64, (8192, 8192), device=dev, dtype=torch.int8)
        b = torch.randint(-64, 64, (8192, 8192), device=dev, dtype=torch.int8)
        us = _time_us(lambda: torch._int_mm(a, b), dev, iters=10)
        rec["matmul_tflops"]["int8"] = round(2 * 8192**3 / us / 1e6, 1)
    except Exception as exc:
        rec["matmul_tflops"]["int8"] = None
        rec["unmeasured"]["matmul_tflops.int8"] = repr(exc)
    # No FP8/FP4 entry on purpose: Xe2 XMX has no FP8/FP4 matrix math; those GEMMs dequantize
    # and multiply in bf16/fp16, which is the peak their bound uses.
    return rec


def default_path(rec: dict) -> Path:
    name = re.sub(
        r"[^A-Za-z0-9]+", "_", str(rec["device"].get("name", rec["device"]["type"]))
    ).strip("_")
    return CACHE / f"{name}.json"


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main(out: Path | None) -> int:
    rec = measure()
    path = out or default_path(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print(
        f"calibrate: {rec['bandwidth_gbs']} GB/s, matmul {rec['matmul_tflops']} TFLOP/s -> {path}"
    )
    return 0
