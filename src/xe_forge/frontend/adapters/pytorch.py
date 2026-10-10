"""Profile any Python callable, and the profiler region every PyTorch adapter shares.

    xe-forge capture --framework torch --target my_pkg.bench:make -o capture.json

``make()`` returns a zero-argument callable; it is called once to warm up and then
``--iters`` times under the profiler.
"""

from __future__ import annotations

import gzip
import importlib
import shutil
import sys
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


def add_arguments(parser) -> None:
    parser.add_argument("--target", help="module:factory returning a zero-arg callable")
    parser.add_argument("--iters", type=int, default=3)


def capture(args, raw_dir: Path) -> dict:
    if not args.target or ":" not in args.target:
        raise SystemExit("--framework torch needs --target module:factory")
    mod_name, attr = args.target.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    fn = getattr(importlib.import_module(mod_name), attr)()
    fn()

    def region():
        for _ in range(args.iters):
            fn()

    profile_region(region, raw_dir)
    return {
        "framework": "torch",
        "model": {"name": args.target},
        "runtime": {"iters": args.iters},
        "device": device_info(),
    }
