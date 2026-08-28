"""
A workload whose hot kernel can actually be replaced (plan §13).

This exists to prove reinsertion end to end. Everything else in Orbit can be exercised
against a trace fixture, but "did the optimized kernel actually replace the original in
a running process" cannot be faked — it needs a real dispatcher op, a real override,
and a real re-profile.

The kernel is an RMSNorm registered as a dispatcher op, `orbit_demo::rms_norm`, which
is what makes rung P1 available: an override registers an implementation for the
existing op on the device key and shadows the default, touching nothing here.

The baseline is deliberately naive — four passes over the data with three intermediate
allocations — so a fused replacement has real headroom rather than a rounding error to
chase. That is honest: a demo where the "optimization" is noise would prove nothing
about the measurement chain.

Run it directly:

    python -m examples.kernel_replacement.workload --iters 200

It prints `ORBIT_KERNEL=<name>` for whichever implementation actually dispatched. That
line is what the dispatch assertion reads (§13): the new kernel must appear *and* the
old one must not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# The dispatch log lives in its own module so the override and the workload append to
# the same list even when the workload is running as __main__ (see dispatch_log.py).
from examples.kernel_replacement import dispatch_log

NAMESPACE = "orbit_demo"
OP_NAME = "rms_norm"
BASELINE_KERNEL = "orbit_demo_rms_norm_naive"


def _define_op() -> None:
    """Register the op and its baseline implementation.

    Registration is idempotent across processes but not within one: defining twice in
    the same interpreter raises. The workload runs as a subprocess precisely so each
    measurement starts from a clean dispatcher.
    """
    library = torch.library.Library(NAMESPACE, "DEF")
    library.define("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor")

    impl = torch.library.Library(NAMESPACE, "IMPL")
    impl.impl(OP_NAME, _naive_rms_norm, "CPU")
    impl.impl(OP_NAME, _naive_rms_norm, "XPU")

    # Keep the handles alive: a garbage-collected Library deregisters its ops.
    globals()["_LIBRARY"] = library
    globals()["_IMPL"] = impl


def _naive_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Four passes, three intermediates. The thing worth replacing."""
    dispatch_log.record(BASELINE_KERNEL)
    squared = x * x
    mean = squared.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean + eps)
    normalized = x / rms
    return normalized * weight


def build_inputs(
    batch: int = 8, seq: int = 512, hidden: int = 1024, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(batch, seq, hidden, generator=generator, device=device)
    weight = torch.randn(hidden, generator=generator, device=device)
    return x, weight


def run(iters: int = 100, device: str = "cpu", warmup: int = 10) -> dict[str, object]:
    """Drive the op and report timing plus which kernel dispatched."""
    x, weight = build_inputs(device=device)
    op = getattr(torch.ops, NAMESPACE).rms_norm

    for _ in range(warmup):
        op(x, weight, 1e-6)
    dispatch_log.clear()

    start = time.perf_counter()
    for _ in range(iters):
        out = op(x, weight, 1e-6)
    if device != "cpu":
        getattr(torch, device).synchronize()
    elapsed = time.perf_counter() - start

    # The checksum is the correctness anchor: a replacement that changes it is wrong,
    # however fast it is.
    checksum = float(out.double().sum().item())
    kernels = dispatch_log.observed()

    return {
        "iters": iters,
        "elapsed_s": elapsed,
        "per_iter_ms": elapsed / iters * 1000.0,
        "checksum": checksum,
        "kernels": kernels,
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replaceable-kernel demo workload")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default=os.environ.get("ORBIT_DEMO_DEVICE", "cpu"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    _define_op()

    # The override, when present, is imported by the *workload* rather than by whatever
    # is measuring it — that is the whole point of P1, and it is why applying a patch
    # never contaminates the baseline process.
    override = os.environ.get("ORBIT_OVERRIDE_MODULE")
    if override:
        __import__(override)

    result = run(iters=args.iters, device=args.device, warmup=args.warmup)

    if args.json:
        print(json.dumps(result))
    else:
        for kernel in result["kernels"]:
            print(f"ORBIT_KERNEL={kernel}")
        print(f"per_iter_ms={result['per_iter_ms']:.4f}")
        print(f"checksum={result['checksum']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
