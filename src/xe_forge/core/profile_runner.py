"""Workload preparation and collection boundaries for profiler subprocesses."""

import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch

from xe_forge.core.executor import KernelBenchExecutor
from xe_forge.core.reference_workload import PreparedCall, ReferenceWorkload
from xe_forge.core.resource_feedback import capture_build_provenance
from xe_forge.core.spec_loader import load_spec


def prepare_call(kernel_file, spec_path=None, variant=None, reference_path=None, device="xpu"):
    executor = KernelBenchExecutor(device=device)
    if reference_path is not None:
        spec_workload = None
        if spec_path is not None:
            spec = load_spec(spec_path)
            variant = spec.resolve_variant(variant)
            spec_workload = spec.get_reference_workload(variant)
            for name, value in (("rtol", spec.get_rtol(variant)), ("atol", spec.get_atol(variant))):
                if value is not None:
                    setattr(executor, name, value)
        elif variant is not None:
            raise ValueError("Profiling --variant requires --spec")
        workload = ReferenceWorkload.prepare(
            executor._compile_module,
            Path(reference_path).read_text(),
            None,
            Path(kernel_file).read_text(),
            device,
            executor.rtol,
            executor.atol,
            spec_workload,
        )
        workload.validate()
        return workload.prepare_call(workload.optimized)

    options = {}
    if spec_path is not None:
        spec = load_spec(spec_path)
        variant = spec.resolve_variant(variant)
        if spec.get_variant(variant) is None:
            raise ValueError(f"Unknown profiling variant: {variant}")
        options = {
            "input_shapes": spec.get_input_shapes(variant),
            "dtype": spec.get_dtype(variant),
            "input_dtypes": spec.get_input_dtypes(variant),
            "init_args": spec.get_init_args(variant),
        }
    elif variant is not None:
        raise ValueError("Profiling --variant requires --spec")
    module = executor._compile_module(Path(kernel_file).read_text())
    if module is None:
        raise ValueError("Could not load profiling kernel")
    if spec_path is None:
        options["inputs"] = module.get_inputs()
    fn, model, inputs = executor.prepare_workload(module, **options)
    return PreparedCall(model if model is not None else fn, inputs)


def run_profile(
    kernel_file,
    *,
    tool,
    warmup,
    iters,
    spec_path=None,
    variant=None,
    reference_path=None,
    device="xpu",
    vtune_bin="vtune",
    result_dir=None,
):
    if warmup < 0 or iters < 1:
        raise ValueError("Profiling requires warmup >= 0 and iters >= 1")
    if tool not in ("unitrace", "vtune"):
        raise ValueError(f"Unknown profiler: {tool}")

    def collection(enabled):
        if tool == "unitrace":
            os.environ["PTI_ENABLE_COLLECTION"] = "1" if enabled else "0"
        else:
            subprocess.run(
                [vtune_bin, "-command", "resume" if enabled else "pause", "-r", str(result_dir)],
                capture_output=True,
                timeout=30,
                check=True,
            )

    def synchronize():
        if device != "cpu":
            getattr(torch, device).synchronize()

    provenance = (
        capture_build_provenance(Path(kernel_file).parent) if tool == "unitrace" else nullcontext()
    )
    previous_collection = os.environ.get("PTI_ENABLE_COLLECTION")
    try:
        collection(False)
        with torch.no_grad():
            with provenance:
                call = prepare_call(kernel_file, spec_path, variant, reference_path, device)
            for _ in range(warmup):
                call(*call.inputs)
                synchronize()
            for _ in range(iters):
                call.reset()
                synchronize()
                collection(True)
                try:
                    call.forward(*call.inputs)
                finally:
                    try:
                        synchronize()
                    finally:
                        collection(False)
            if tool == "unitrace" and device == "xpu":
                event = torch.xpu.Event()
                event.record()
                event.synchronize()
        return call
    finally:
        if tool == "unitrace":
            if previous_collection is None:
                os.environ.pop("PTI_ENABLE_COLLECTION", None)
            else:
                os.environ["PTI_ENABLE_COLLECTION"] = previous_collection
