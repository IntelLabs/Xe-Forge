"""``orbit_mini`` — the Xe-Orbit reference micro-workload (plan §15).

Nothing in the Orbit pipeline can be developed against a full vLLM run: a closed
loop that takes twenty minutes cannot be iterated on and cannot run in CI. This
package is the small workload that exercises the whole thing in seconds — a
two-layer Qwen-shaped decoder block at toy dimensions, and **deliberately
adversarial for extraction**, because a test workload that extracts cleanly
tests nothing (§15.2).

Entry points::

    from examples.orbit_mini import build_model, get_example_inputs, run, main

    python -m examples.orbit_mini
    python -m examples.orbit_mini --steps 5

Runs on CPU with plain PyTorch and no Triton. The Triton and SYCL paths are in
the source — that is the point, the file structure has to exercise the §11
language taxonomy — but every one of them is behind an availability check with a
pure-torch fallback, so the CPU-only CI tier T0 (§16.6) stays green.

The seven traps, and where they live:

===================================  =========================================
Split device-helper closure          ``kernels/helpers_{a,b,c}.py``,
(3 modules, one via a re-export)     reached through ``kernels/device_ops.py``
Autotune config list to pin          ``kernels/rmsnorm.py:RMSNORM_CONFIGS``
Heuristics over module state         ``kernels/rmsnorm.py:_num_stages_hint``
Data dependency keyed by device      ``kernels/tuned_configs.json``
Non-contiguous input                 :func:`get_example_inputs`
Hand-written SYCL dispatcher op      ``sycl/`` + ``kernels/sycl_op.py``
Opaque library call (E4/NO_ACTION)   ``kernels/opaque_gemm.py``
Three-kernel fusable region          ``kernels/region.py``
===================================  =========================================
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from typing import Any

import torch

from .kernels import opaque_gemm, region, rmsnorm, swiglu, sycl_op, triton_compat, tuned
from .model import OrbitMiniConfig, OrbitMiniModel

__all__ = [
    "OrbitMiniConfig",
    "OrbitMiniModel",
    "build_model",
    "get_example_inputs",
    "main",
    "run",
]

#: Seed for every stochastic thing in the workload. §17 wants reproducibility to
#: be a property of the fixture, not something a caller remembers to arrange.
SEED: int = 20260101


def build_model(
    config: OrbitMiniConfig | None = None,
    device: torch.device | str = "cpu",
    *,
    eval_mode: bool = True,
) -> OrbitMiniModel:
    """Build the two-layer decoder block.

    ``eval_mode`` defaults to True: a workload measured in training mode is not
    the workload being served, and the difference shows up in the trace.
    """
    torch.manual_seed(SEED)
    model = OrbitMiniModel(config)
    model = model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model


def get_example_inputs(
    config: OrbitMiniConfig | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the hidden-states input — **deliberately non-contiguous**.

    §15.2: "one input deliberately non-contiguous, so synthetic-input
    reconstruction fails visibly."

    The tensor is allocated as ``(batch, hidden, seq)`` and handed back
    transposed, so its shape is the ordinary ``(batch, seq, hidden)`` a caller
    expects while its strides are not. Anything that reconstructs this input
    from a shape and a dtype — which is what synthetic-input capture does when
    it has not stored the strides — produces a contiguous tensor, a different
    launch record (§12.4), and, on a stride-sensitive kernel, different numbers.

    The non-contiguity holds on CPU, on XPU and on CUDA; it is a property of the
    view, not of the backend.
    """
    cfg = config or OrbitMiniConfig()
    dev = torch.device(device)
    torch.manual_seed(SEED)

    # Allocate transposed, then flip the last two axes back.
    staged = torch.randn(cfg.batch_size, cfg.hidden_size, cfg.seq_len, dtype=cfg.dtype, device=dev)
    hidden_states = staged.transpose(1, 2)

    # A raise rather than an assert: `python -O` strips asserts, and a trap that
    # disarms itself under an optimisation flag is worse than no trap.
    if hidden_states.shape != (cfg.batch_size, cfg.seq_len, cfg.hidden_size):
        raise RuntimeError(f"unexpected input shape {tuple(hidden_states.shape)}")
    if hidden_states.is_contiguous():
        raise RuntimeError(
            "orbit_mini's input must be non-contiguous; something normalised it "
            "away and the synthetic-input trap is no longer armed (plan §15.2)."
        )
    return hidden_states


def environment_report() -> dict[str, Any]:
    """What §12.9 says a bundle has to pin. Collected here so the rig can diff it."""
    device_name = "cpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": getattr(triton_compat.triton, "__version__", None)
        if triton_compat.HAS_TRITON
        else None,
        "has_triton": triton_compat.HAS_TRITON,
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": device_name,
        "sycl_op": sycl_op.status(),
    }


def run(steps: int = 3, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Run the workload end to end and return a structured summary.

    This is the shape ``xe-orbit selftest`` drives: a fixed number of forward
    passes, a deterministic checksum per step, and the launch metadata the
    extraction stages have to reproduce. Returning a dict rather than printing
    is what lets a test assert on it (§16.2 rule 2 — no stage parses console
    output when a structured artifact exists).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    cfg = OrbitMiniConfig()
    dev = torch.device(device)
    model = build_model(cfg, dev)
    hidden_states = get_example_inputs(cfg, dev)
    opaque_gemm.reset_call_sites()

    tuned_entry = tuned.lookup(dev)

    checksums: list[float] = []
    durations_ms: list[float] = []
    output: torch.Tensor | None = None

    with torch.inference_mode():
        for _ in range(steps):
            started = time.perf_counter()
            output = model(hidden_states)
            durations_ms.append((time.perf_counter() - started) * 1e3)
            checksums.append(float(output.to(torch.float64).sum().item()))

    assert output is not None

    # Run the fusable region standalone as well (§12.11): the region driver has
    # to work outside the model, or a fused candidate has nothing to be compared
    # against as a unit.
    layer = model.layers[0]
    with torch.inference_mode():
        region_out, intermediates = region.run_region(
            hidden=hidden_states,
            residual=hidden_states,
            o_proj_weight=layer.attention.o_weight,
            norm_weight=layer.post_attention_norm.weight,
            gate_weight=layer.mlp.gate_weight,
            up_weight=layer.mlp.up_weight,
        )

    return {
        "workload": "orbit_mini",
        "steps": steps,
        "device": str(dev),
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "ffn_size": cfg.ffn_size,
            "num_layers": cfg.num_layers,
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
        },
        "environment": environment_report(),
        "input": {
            "shape": tuple(hidden_states.shape),
            "strides": tuple(hidden_states.stride()),
            "contiguous": bool(hidden_states.is_contiguous()),
        },
        "output": {
            "shape": tuple(output.shape),
            "checksum": checksums[-1],
            "deterministic": len(set(checksums)) == 1,
        },
        "timing_ms": {
            "per_step": durations_ms,
            "mean": sum(durations_ms) / len(durations_ms),
        },
        "tuned_config": {
            "path": str(tuned.TUNED_CONFIG_PATH),
            "entry": tuned_entry.describe(),
        },
        "launches": {
            "rmsnorm": dict(rmsnorm.LAST_LAUNCH),
            "swiglu": dict(swiglu.LAST_LAUNCH),
        },
        "opaque_call_sites": {
            "count": len(opaque_gemm.CALL_SITES),
            "level": opaque_gemm.EXTRACTION_LEVEL,
            "permitted_actions": list(opaque_gemm.PERMITTED_ACTIONS),
        },
        "region": {
            "region_id": region.MLP_REGION.region_id,
            "members": [m.name for m in region.MLP_REGION.members],
            "edges": [list(e) for e in region.MLP_REGION.edges],
            "intermediates": list(region.MLP_REGION.intermediates),
            "output_shape": tuple(region_out.shape),
            "materialised": sorted(intermediates),
        },
    }


def _print_summary(summary: dict[str, Any]) -> None:
    env = summary["environment"]
    cfg = summary["config"]
    launch = summary["launches"]["rmsnorm"]

    print("orbit_mini — Xe-Orbit reference micro-workload (plan §15)")
    print("-" * 66)
    print(f"  device            {summary['device']}  ({env['device_name']})")
    print(f"  torch             {env['torch']}   python {env['python']}")
    triton_line = (
        f"yes, {env['triton']}" if env["has_triton"] else "not installed -> torch fallback"
    )
    print(f"  triton            {triton_line}")
    print(f"  {env['sycl_op']}")
    print()
    print(
        f"  model             {cfg['num_layers']} layers, hidden={cfg['hidden_size']}, "
        f"heads={cfg['num_heads']}/{cfg['num_kv_heads']} kv, head_dim={cfg['head_dim']}, "
        f"ffn={cfg['ffn_size']}"
    )
    print(f"  batch x seq       {cfg['batch_size']} x {cfg['seq_len']}")
    print()
    print("  ADVERSARIAL STRUCTURE")
    inp = summary["input"]
    print(
        f"    input           shape={inp['shape']} strides={inp['strides']} "
        f"contiguous={inp['contiguous']}   <- trap armed"
    )
    print(f"    data dep        {summary['tuned_config']['path']}")
    print(f"                    {summary['tuned_config']['entry']}")
    print(f"    autotune pin    {launch.get('config')}  (backend: {launch.get('backend')})")
    print(f"    heuristics      {launch.get('constexprs')}")
    print(
        f"    opaque calls    {summary['opaque_call_sites']['count']} sites at "
        f"{summary['opaque_call_sites']['level']}, actions "
        f"{summary['opaque_call_sites']['permitted_actions']}"
    )
    reg = summary["region"]
    print(f"    fusable region  {' -> '.join(reg['members'])}")
    print(f"                    intermediates {reg['intermediates']}")
    print()
    out = summary["output"]
    timing = summary["timing_ms"]
    print(
        f"  {summary['steps']} steps       output {out['shape']}  "
        f"checksum {out['checksum']:.6f}  deterministic={out['deterministic']}"
    )
    print(
        f"                    {timing['mean']:.2f} ms/step "
        f"({', '.join(f'{t:.2f}' for t in timing['per_step'])})"
    )
    print("-" * 66)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. ``python -m examples.orbit_mini``."""
    parser = argparse.ArgumentParser(
        prog="orbit_mini",
        description="Xe-Orbit reference micro-workload (plan §15).",
    )
    parser.add_argument("--steps", type=int, default=3, help="forward passes to run")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    summary = run(steps=args.steps, device=args.device)

    if args.json:
        import json

        print(json.dumps(summary, indent=2, default=str))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
