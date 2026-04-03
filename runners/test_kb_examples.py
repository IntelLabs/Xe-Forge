#!/usr/bin/env python3
"""
Test runner for knowledge base example pairs.

Usage:
    python test_kb_examples.py                          # test all pairs
    python test_kb_examples.py gemm_activation          # test one pair by id
    python test_kb_examples.py --single gemmelem.py     # test a single file

Each test:
  1. Imports the Model class
  2. Runs forward() with random inputs
  3. Checks output shape, dtype, no NaN/Inf
  4. For pairs: checks unoptimized vs optimized outputs match (within tolerance)
  5. Reports pass/fail and execution time
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

# ── config ──────────────────────────────────────────────────────────────────
# Resolve examples dir relative to this script's location.
# Works whether the script lives in the repo root or in knowledge_base/
_here = Path(__file__).resolve().parent
if (_here / "examples").exists():
    _DEFAULT_EXAMPLES_DIR = _here / "examples"
elif (_here / "knowledge_base" / "examples").exists():
    _DEFAULT_EXAMPLES_DIR = _here / "knowledge_base" / "examples"
else:
    _DEFAULT_EXAMPLES_DIR = _here / "examples"
RTOL = 1e-2
ATOL = 1e-2

# Shape overrides for kernels that need specific dimensions
# (e.g. matmul_max needs M divisible by reduction factor)
SHAPE_OVERRIDES: dict[str, dict] = {
    "matmul_max.py": {"batch_size": 128, "in_features": 512, "out_features": 256},
    "matmul_maxpool_sum_scale": {"batch_size": 128},
}

PAIRS = [
    {
        "id": "gemm_activation",
        "unoptimized": "gemm_activation_unoptimized.py",
        "optimized": "gemm_activation_optimized.py",
    },
    {
        "id": "matmul_at",
        "unoptimized": "matmul_at_unoptimized.py",
        "optimized": "matmul_at_optimized.py",
    },
]

# Per-file module-level attribute overrides for files without get_init_inputs
MODULE_DEFAULTS: dict[str, dict] = {
    "gemmelem.py": {"in_features": 512, "out_features": 256},
}

SINGLES = [
    "gemmelem.py",
    "MMTanh.py",
    "matt_elem.py",
    "matmul_max.py",
    "TritonSigmGemm.py",
    "gemm_sigm.py",
    "gemm_scale.py",
    "gemmsoft.py",
    "MMSoftx.py",
    "gemm_sum.py",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def load_model_from_file(path: Path):
    """Import a kernel file and return its Model class + get_inputs/get_init_inputs."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_model_and_inputs(mod, device="xpu", fname=None):
    """Instantiate Model and generate inputs from the module's get_* helpers."""
    init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs() if hasattr(mod, "get_inputs") else []

    # If no init args found but Model requires them, try module-level defaults
    if not init_args:
        import inspect

        sig = inspect.signature(mod.Model.__init__)
        required = [
            p for p in list(sig.parameters.values())[1:] if p.default is inspect.Parameter.empty
        ]
        if required:
            # Use module-level variables as fallback (in_features, out_features, etc.)
            overrides = MODULE_DEFAULTS.get(fname or "", {})
            fallbacks = {
                k: getattr(mod, k)
                for k in [
                    "in_features",
                    "out_features",
                    "hidden_size",
                    "input_size",
                    "constant",
                    "kernel_size",
                    "scale_factor",
                    "scaling_factor",
                ]
                if hasattr(mod, k)
            }
            fallbacks.update(overrides)
            init_args = [fallbacks.get(p.name, 512) for p in required]

    model = mod.Model(*init_args).to(device)
    inputs_dev = [t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs]
    return model, inputs_dev


def run_forward(model, inputs, warmup=3, runs=1):
    """Run forward pass, return output and median wall time in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            out = model(*inputs)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            out = model(*inputs)
        torch.xpu.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / runs * 1000

    return out, elapsed_ms


def check_tensor(t, name):
    """Return list of issues with a tensor."""
    issues = []
    if torch.isnan(t).any():
        issues.append(f"{name}: contains NaN")
    if torch.isinf(t).any():
        issues.append(f"{name}: contains Inf")
    return issues


def test_single(path: Path, label: str | None = None) -> bool:
    label = label or path.name
    print(f"\n{'─' * 60}")
    print(f"Testing: {label}")
    try:
        mod = load_model_from_file(path)
        model, inputs = make_model_and_inputs(mod, fname=path.name)
        out, ms = run_forward(model, inputs)
        issues = check_tensor(out, "output")
        if issues:
            for iss in issues:
                print(f"  FAIL  {iss}")
            return False
        print(f"  PASS  shape={tuple(out.shape)} dtype={out.dtype} time={ms:.2f}ms")
        return True
    except Exception as e:
        print(f"  ERROR {e}")
        return False


def test_pair(pair: dict, examples_dir: Path | None = None) -> bool:
    pid = pair["id"]
    base = examples_dir if examples_dir else _DEFAULT_EXAMPLES_DIR
    unopt_path = base / pair["unoptimized"]
    opt_path = base / pair["optimized"]

    print(f"\n{'═' * 60}")
    print(f"Pair: {pid}")

    try:
        unopt_mod = load_model_from_file(unopt_path)
        opt_mod = load_model_from_file(opt_path)

        # Build reference model + shared inputs from unoptimized module
        ref_model, ref_inputs = make_model_and_inputs(unopt_mod, fname=unopt_path.name)

        # Build optimized model with same constructor args
        init_args = opt_mod.get_init_inputs() if hasattr(opt_mod, "get_init_inputs") else []
        opt_model = opt_mod.Model(*init_args).to("xpu")

        # Copy weights/bias so both models use identical parameters
        opt_model.load_state_dict(ref_model.state_dict())

        # Shared input tensor(s)
        shared_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in ref_inputs]

        results = {}
        outputs = {}

        for label, model in [("unoptimized", ref_model), ("optimized", opt_model)]:
            print(f"  [{label}] {(unopt_path if label == 'unoptimized' else opt_path).name}")
            try:
                out, ms = run_forward(model, shared_inputs)
                issues = check_tensor(out, label)
                if issues:
                    for iss in issues:
                        print(f"    FAIL  {iss}")
                    results[label] = False
                else:
                    print(f"    OK    shape={tuple(out.shape)} dtype={out.dtype} time={ms:.2f}ms")
                    results[label] = True
                    outputs[label] = out.float()
            except Exception as e:
                print(f"    ERROR {e}")
                results[label] = False

        if results.get("unoptimized") and results.get("optimized"):
            ref = outputs["unoptimized"]
            opt_out = outputs["optimized"]

            if ref.shape != opt_out.shape:
                print(f"  MISMATCH  shape {ref.shape} vs {opt_out.shape}")
                return False

            if torch.allclose(ref, opt_out, rtol=RTOL, atol=ATOL):
                max_diff = (ref - opt_out).abs().max().item()
                print(f"  MATCH     max_diff={max_diff:.2e} (rtol={RTOL} atol={ATOL})")
                return True
            else:
                max_diff = (ref - opt_out).abs().max().item()
                print(f"  MISMATCH  max_diff={max_diff:.2e} exceeds tolerance")
                return False

        return all(results.values())

    except Exception as e:
        print(f"  ERROR {e}")
        return False


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Test KB example kernels")
    parser.add_argument("ids", nargs="*", help="Pair IDs or filenames to test (default: all)")
    parser.add_argument("--single", help="Test a single file directly")
    parser.add_argument("--examples-dir", default=None, help="Path to examples directory")
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir).resolve() if args.examples_dir else _DEFAULT_EXAMPLES_DIR

    if not torch.xpu.is_available():
        print("ERROR: XPU not available")
        sys.exit(1)

    passed, failed = 0, 0

    if args.single:
        ok = test_single(examples_dir / args.single)
        sys.exit(0 if ok else 1)

    filter_ids = set(args.ids) if args.ids else None

    # Test pairs
    for pair in PAIRS:
        if filter_ids and pair["id"] not in filter_ids:
            continue
        ok = test_pair(pair, examples_dir=examples_dir)
        if ok:
            passed += 1
        else:
            failed += 1

    # Test singles
    for fname in SINGLES:
        if filter_ids and fname not in filter_ids and fname.replace(".py", "") not in filter_ids:
            continue
        ok = test_single(examples_dir / fname)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
