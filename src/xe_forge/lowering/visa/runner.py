"""GPU-side work for vISA lowering, run in a child process per job.

``python -m xe_forge.lowering.visa.runner <job.json>`` reads a job, writes
``result.json`` next to it, and exits. Every job gets a fresh process: IGC reads
its flags once per process, a hung kernel can only be ended from outside, and a
crashed device context must not outlive the attempt that caused it.

Modes:

``capture``
    Run ``Model.forward`` once with launches recorded; build the ABI stub for the
    launch to lower and write its SPIR-V. The reference kernel compiles here
    (into this job's private Triton cache, which the parent hands to the
    artifact guard and never to the model).

``evaluate``
    For each verification case: run the original module (reference) and the same
    module with the target kernel's launches redirected to the stub, whose native
    code is the finalized vISA. Compare every output and every input the
    reference mutated, check guard regions, then optionally time both.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_TL_TO_TORCH = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp64": "float64",
    "i8": "int8",
    "u8": "uint8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "i1": "bool",
}


class SpecializationChanged(RuntimeError):
    """A case launched the kernel with other constexprs than the contract's."""


def _device() -> str:
    return os.environ.get("XE_FORGE_LOWER_DEVICE", "xpu")


def _sync():
    import torch

    getattr(torch, _device().split(":")[0]).synchronize()


def build_model(module, spec, variant: str, seed: int):
    import torch

    torch.manual_seed(1234)
    init = spec.get_init_args(variant) if spec is not None else []
    if not init and hasattr(module, "get_init_inputs"):
        init = list(module.get_init_inputs())
    model = module.Model(*init)
    if hasattr(model, "to"):
        model = model.to(_device())
    torch.manual_seed(seed)
    if spec is not None and spec.inputs:
        inputs = spec.create_inputs(variant, device=_device())
    else:
        inputs = [x.to(_device()) if hasattr(x, "to") else x for x in module.get_inputs()]
    return model, inputs


def _clone(inputs):
    return [x.clone() if hasattr(x, "clone") else x for x in inputs]


def _outputs(out) -> list:
    if isinstance(out, (tuple, list)):
        return [o for o in out if hasattr(o, "shape")]
    return [out] if hasattr(out, "shape") else []


# -- capture ------------------------------------------------------------------


def run_capture(job: dict) -> dict:
    import torch

    with torch.no_grad():
        return _run_capture(job)


def _run_capture(job: dict) -> dict:
    from xe_forge.core.spec_loader import load_spec
    from xe_forge.lowering.triton_analyzer import (
        capture_launches,
        jit_functions,
        kernel_source,
        load_module,
        select_launch,
    )
    from xe_forge.lowering.visa.compiler import stub_source

    module = load_module(job["module_path"])
    spec = load_spec(job["spec_path"]) if job.get("spec_path") else None
    variant = spec.resolve_variant(job.get("variant")) if spec is not None else None
    model, inputs = build_model(module, spec, variant, seed=0)
    records = []
    with capture_launches(records):
        model(*inputs)
        _sync()
    record = select_launch(records, job.get("kernel"))
    source = kernel_source(jit_functions(module)[record.kernel])

    workdir = Path(job["workdir"])
    stub_dir = workdir / "stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_py = stub_dir / f"{record.kernel}_abi_stub.py"
    stub_py.write_text(stub_source(record))
    spv = build_stub_spv(stub_py, record, stub_dir / "stub.spv")
    return {
        "launch": record.to_json(),
        "kernel_source": source,
        "variant": variant,
        "dims": spec.get_dims(variant) if spec is not None else {},
        "stub_py": str(stub_py),
        "stub_spv": str(spv),
        "all_kernels": sorted({r.kernel for r in records}),
    }


def build_stub_spv(stub_py: Path, record, out: Path, config=None) -> Path:
    import torch

    from xe_forge.lowering.triton_analyzer import load_module

    stub = getattr(load_module(stub_py), record.kernel)
    args = []
    for a in record.args:
        if a.kind == "pointer":
            args.append(torch.empty(64, dtype=getattr(torch, _TL_TO_TORCH.get(a.dtype, "float32")), device=_device()))
        elif a.kind == "scalar":
            # A None argument is compile-time to Triton: pass None so the signature matches.
            args.append(None if a.dtype == "none" else record.scalar_values.get(a.name, 0))
    constexprs = {a.name: a.value for a in record.args if a.kind == "constexpr"}
    nw = config.num_warps if config else record.num_warps
    tpw = config.threads_per_warp if config else record.threads_per_warp
    compiled = stub.warmup(*args, grid=(1,), num_warps=nw, warp_size=tpw, **constexprs)
    out.write_bytes(compiled.asm["spv"])
    return out


def run_stub(job: dict) -> dict:
    """Build the ABI stub for a launch configuration the kernel chose."""
    from xe_forge.lowering.triton_analyzer import LaunchRecord
    from xe_forge.lowering.visa.compiler import stub_source
    from xe_forge.lowering.visa.launch import LaunchConfig

    record = LaunchRecord.from_json(job["launch"])
    config = LaunchConfig.from_json(job["config"])
    stub_dir = Path(job["workdir"])
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_py = stub_dir / f"{record.kernel}_abi_stub_{config.key()}.py"
    stub_py.write_text(stub_source(record, config))
    spv = build_stub_spv(stub_py, record, stub_dir / "stub.spv", config)
    return {"stub_py": str(stub_py), "stub_spv": str(spv)}


# -- evaluate -----------------------------------------------------------------


@contextmanager
def redirect_to_visa(kernel_name: str, stub_fn, zebin: bytes, expected_constexprs: dict, num_warps: int, report: dict,
                     config=None):
    """Inside the block, launches of ``kernel_name`` run the finalized vISA instead.

    Every tensor argument is placed between guard regions for the launch and
    copied back afterwards, so the caller sees normal semantics and any write
    outside the tensor's own span is caught.
    """
    from triton import knobs
    from triton.runtime.jit import JITFunction

    from xe_forge.lowering.visa.verifier import guard

    digest = hashlib.sha256(zebin + repr(config).encode()).hexdigest()
    slm = config.slm_bytes if config else 0

    def hook(self=None, stages=None, options=None, language=None, capability=None):
        if stages is None:
            return "xe_forge_visa", digest
        if "zebin" in stages:
            def make_zebin(src, metadata):
                metadata["binary_ext"] = "zebin"
                metadata["generate_native_code"] = True
                if slm:
                    # The launcher binds this many bytes of shared local memory.
                    metadata["shared"] = max(int(metadata.get("shared") or 0), slm)
                return zebin

            stages["zebin"] = make_zebin

    original_run = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        if self is stub_fn or getattr(self, "__name__", None) != kernel_name or warmup:
            return original_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        bound = dict(zip(self.arg_names, args))
        bound.update(kwargs)
        got = {k: bound.get(k, p.default if p.has_default else None) for k, p in ((p.name, p) for p in self.params) if p.is_constexpr}
        got = {k: getattr(v, "value", v) for k, v in got.items()}
        if got != expected_constexprs or kwargs.get("num_warps", 4) != num_warps:
            raise SpecializationChanged(f"constexprs {got} / num_warps {kwargs.get('num_warps', 4)}")
        guarded = {}
        new_args = []
        for a in args:
            if hasattr(a, "data_ptr") and hasattr(a, "stride"):
                key = (a.data_ptr(), tuple(a.shape), tuple(a.stride()))
                if key not in guarded:
                    guarded[key] = guard(a)
                new_args.append(guarded[key].view)
            else:
                new_args.append(a)
        new_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "data_ptr") and hasattr(v, "stride"):
                key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()))
                if key not in guarded:
                    guarded[key] = guard(v)
                new_kwargs[k] = guarded[key].view
            else:
                new_kwargs[k] = v
        if callable(grid):
            meta = dict(zip(self.arg_names, args))
            meta.update(kwargs)
            grid = grid(meta)
        if config is not None:
            new_kwargs["num_warps"] = config.num_warps
            new_kwargs["warp_size"] = config.threads_per_warp
            if config.grid:
                from xe_forge.lowering.visa.launch import eval_grid

                g = tuple(grid) + (1,) * (3 - len(tuple(grid)))
                names = {k: v for k, v in bound.items() if isinstance(v, (int, bool)) or hasattr(v, "value")}
                names = {k: int(getattr(v, "value", v)) for k, v in names.items()}
                names.update(grid0=g[0], grid1=g[1], grid2=g[2])
                grid = eval_grid(config.grid, names)
        previous = knobs.intel.gen_native_code
        knobs.intel.gen_native_code = True
        try:
            kernel = original_run(stub_fn, *new_args, grid=grid, warmup=False, **new_kwargs)
        finally:
            knobs.intel.gen_native_code = previous
        _sync()
        report["launches"] = report.get("launches", 0) + 1
        for g in guarded.values():
            if not g.guards_intact():
                report.setdefault("guard_violations", []).append(f"{list(g.original.shape)} {g.original.dtype}")
            g.write_back()
        return kernel

    knobs.runtime.add_stages_inspection_hook = hook
    JITFunction.run = run
    try:
        yield
    finally:
        JITFunction.run = original_run
        knobs.runtime.add_stages_inspection_hook = None


def run_evaluate(job: dict) -> dict:
    import torch

    # Verification never needs autograd, and with it on, writing results back into a
    # tensor argument that is an nn.Parameter (a weight passed straight to the kernel)
    # is an illegal in-place operation on a leaf.
    with torch.no_grad():
        return _run_evaluate(job)


def _run_evaluate(job: dict) -> dict:
    import torch

    from xe_forge.core.spec_loader import load_spec
    from xe_forge.lowering.triton_analyzer import LaunchRecord, load_module
    from xe_forge.lowering.visa.verifier import compare, merge, spec_with_dims, tolerance_for

    record = LaunchRecord.from_json(job["launch"])
    zebin = Path(job["zebin_path"]).read_bytes()
    module = load_module(job["module_path"])
    stub_fn = getattr(load_module(job["stub_py"]), record.kernel)
    spec = load_spec(job["spec_path"]) if job.get("spec_path") else None
    variant = job.get("variant")
    from xe_forge.lowering.visa.launch import LaunchConfig

    config = LaunchConfig.from_json(job["config"]) if job.get("config") else None
    tol = job.get("tolerance") or {}
    transcendental = bool(job.get("transcendental"))

    case_results = []
    skipped = []
    report: dict[str, Any] = {}
    runtime_error = None
    for case in job["cases"]:
        label = f"{case['label']} seed={case['seed']}"
        case_spec = spec_with_dims(spec, variant, case["dims"]) if spec is not None else None
        try:
            model, inputs = build_model(module, case_spec, variant, case["seed"])
            ref_inputs = _clone(inputs)
            ref_out = _outputs(model(*ref_inputs))
            _sync()
            cand_inputs = _clone(inputs)
            report_case: dict[str, Any] = {}
            with redirect_to_visa(record.kernel, stub_fn, zebin, record.constexprs, record.num_warps, report_case, config):
                cand_out = _outputs(model(*cand_inputs))
                _sync()
        except SpecializationChanged as e:
            skipped.append({"case": label, "reason": str(e)})
            continue
        except Exception as e:  # launch or runtime failure of the candidate
            runtime_error = f"{type(e).__name__}: {e}"
            report["runtime_traceback"] = traceback.format_exc(limit=4)
            report["failing_case"] = {"case": label, "dims": case["dims"]}
            break
        if report_case.get("launches", 0) == 0:
            runtime_error = "the target kernel was never launched through the vISA path"
            break
        # Outputs, then every input the reference mutated (in-place kernels).
        pairs = list(zip(ref_out, cand_out)) if len(ref_out) == len(cand_out) else []
        for orig, r, c in zip(inputs, ref_inputs, cand_inputs):
            if hasattr(orig, "shape") and not torch.equal(orig, r):
                pairs.append((r, c))
        results = []
        for r, c in pairs:
            rtol, atol = tolerance_for(r.dtype, transcendental=transcendental, rtol=tol.get("rtol"), atol=tol.get("atol"))
            results.append(compare(r, c, rtol=rtol, atol=atol))
        if not pairs:
            from xe_forge.lowering.visa.diagnostics import CorrectnessResult

            results.append(CorrectnessResult(False, error="the module produced no comparable output"))
        unexpected = [
            j for j, (orig, r, c) in enumerate(zip(inputs, ref_inputs, cand_inputs))
            if hasattr(orig, "shape") and torch.equal(orig, r) and not torch.equal(r, c)
        ]
        case_result = merge([(label, x) for x in results])
        case_result.shapes_checked = 1
        if report_case.get("guard_violations"):
            case_result.success = False
            case_result.guard_violations = report_case["guard_violations"]
        if unexpected:
            case_result.success = False
            case_result.inputs_modified = [f"input {j}" for j in unexpected]
        if not case_result.success and case_result.failing_case is None:
            case_result.failing_case = {"case": label, "dims": case["dims"]}
        elif not case_result.success:
            case_result.failing_case = {**case_result.failing_case, "dims": case["dims"]}
        case_results.append((label, case_result))

    from dataclasses import asdict

    out: dict[str, Any] = {"skipped": skipped, **report}
    if runtime_error is not None:
        out["runtime"] = {"success": False, "error": runtime_error}
        return out
    out["runtime"] = {"success": True}
    merged = merge(case_results)
    merged.shapes_skipped = len(skipped)
    out["correctness"] = asdict(merged)
    if merged.success and job.get("measure_perf"):
        out["performance"] = measure(module, spec, variant, record, stub_fn, zebin, job, config)
    return out


def measure(module, spec, variant, record, stub_fn, zebin, job, config=None) -> dict:
    """Median per-call device time of ``Model.forward``: reference vs vISA, interleaved."""
    import torch

    dev = getattr(torch, _device().split(":")[0])
    model, inputs = build_model(module, spec, variant, seed=0)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=_device())
    iters = int(job.get("perf_iters", 50))
    rounds = int(job.get("perf_rounds", 5))

    def time_calls(n: int) -> list[float]:
        times = []
        for _ in range(n):
            flush.zero_()
            start, end = dev.Event(enable_timing=True), dev.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return times

    report: dict = {}
    base, cand = [], []
    time_calls(5)
    for _ in range(rounds):
        base += time_calls(iters)
        with redirect_to_visa(record.kernel, stub_fn, zebin, record.constexprs, record.num_warps, report, config):
            if not cand:
                time_calls(5)
            cand += time_calls(iters)
    base.sort()
    cand.sort()
    return {
        "baseline_us": base[len(base) // 2],
        "candidate_us": cand[len(cand) // 2],
        "timer": f"{_device()}-events-median-l2flush-{rounds}x{iters}-interleaved",
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job_path = Path(argv[0])
    job = json.loads(job_path.read_text())
    result_path = job_path.with_name("result.json")
    os.environ.setdefault("TRITON_CACHE_DIR", job.get("triton_cache") or tempfile.mkdtemp(prefix="xf_visa_tc_"))
    try:
        if job["mode"] == "capture":
            result = run_capture(job)
        elif job["mode"] == "stub":
            result = run_stub(job)
        elif job["mode"] == "evaluate":
            result = run_evaluate(job)
        else:
            raise ValueError(f"unknown mode {job['mode']!r}")
        result["ok"] = True
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    result_path.write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
