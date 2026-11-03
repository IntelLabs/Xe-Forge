"""Helpers a capture launcher imports: the device torch sees, and one torch.profiler region
written as ``trace.json.gz`` -- what ``xe-forge-skill capture parse`` reads for a torch
framework.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path


def device_type() -> str:
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info() -> dict:
    import torch

    dev = device_type()
    info = {"type": dev, "torch": torch.__version__}
    if dev == "xpu":
        info["name"] = torch.xpu.get_device_name(0)
    elif dev == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
    return info


def profile_region(fn, raw_dir: Path, with_stack: bool = True) -> Path:
    """Run ``fn`` under torch.profiler and write ``raw_dir/trace.json.gz``."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = device_type()
    activities = [ProfilerActivity.CPU]
    if dev == "xpu":
        activities.append(ProfilerActivity.XPU)
    elif dev == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=with_stack) as prof:
        fn()
        if dev != "cpu":
            getattr(torch, dev).synchronize()

    raw_dir.mkdir(parents=True, exist_ok=True)
    plain = raw_dir / "trace.json"
    prof.export_chrome_trace(str(plain))
    out = raw_dir / "trace.json.gz"
    with open(plain, "rb") as src, gzip.open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    plain.unlink()
    return out
