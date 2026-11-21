"""Workload preparation and collection boundaries for profiler subprocesses."""

import json
import os
import subprocess
import time
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


# How long the runner waits for perf to acknowledge a start or stop before giving up: a
# liveness bound (perf has exited or stopped reading), not a measurement.
PERF_ACK_TIMEOUT_S = 60


class Collection:
    """Starts and stops one collector around the timed calls. unitrace reads an environment
    variable; VTune takes a command against its result; perf reads ``enable``/``disable``
    from a control descriptor and acknowledges each on another. ``none`` controls nothing,
    for a collector started around the whole run by hand (perf c2c, perf mem)."""

    def __init__(self, tool, vtune_bin="vtune", result_dir=None, control_fds=None):
        if tool not in ("unitrace", "vtune", "perf", "none"):
            raise ValueError(f"Unknown profiler: {tool}")
        if tool == "perf" and not control_fds:
            raise ValueError("perf collection needs its control and ack descriptors")
        self.tool, self.vtune_bin, self.result_dir = tool, vtune_bin, result_dir
        self.control_fds = tuple(control_fds or ())[:2]
        # perf's own ends came along through perf; holding them would hide its exit.
        for fd in tuple(control_fds or ())[2:]:
            try:
                os.close(fd)
            except OSError:
                pass

    def set(self, enabled: bool) -> None:
        if self.tool == "unitrace":
            os.environ["PTI_ENABLE_COLLECTION"] = "1" if enabled else "0"
        elif self.tool == "vtune":
            subprocess.run(
                [self.vtune_bin, "-command", "resume" if enabled else "pause", "-r",
                 str(self.result_dir)],
                capture_output=True,
                timeout=30,
                check=True,
            )  # fmt: skip
        elif self.tool == "perf":
            import select

            control, ack = self.control_fds
            os.write(control, b"enable\n" if enabled else b"disable\n")
            ready, _, _ = select.select([ack], [], [], PERF_ACK_TIMEOUT_S)
            if not ready or not os.read(ack, 16).startswith(b"ack"):
                raise RuntimeError("perf did not acknowledge a collection command")


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
    window="each",
    window_file=None,
    control_fds=None,
):
    """Run the timed calls inside the collection: each forward in its own window
    (``window="each"``), or one window over all timed calls, their buffer resets
    included (``"all"``), for a collector whose start/stop costs more than a forward.
    With *window_file*, what the window held is written there as JSON (see
    :func:`_timed_calls`)."""
    if warmup < 0 or iters < 1:
        raise ValueError("Profiling requires warmup >= 0 and iters >= 1")
    if window not in ("each", "all"):
        raise ValueError(f"Unknown collection window: {window}")
    collection = Collection(tool, vtune_bin, result_dir, control_fds)

    def synchronize():
        if device != "cpu":
            getattr(torch, device).synchronize()

    provenance = (
        capture_build_provenance(Path(kernel_file).parent) if tool == "unitrace" else nullcontext()
    )
    previous_collection = os.environ.get("PTI_ENABLE_COLLECTION")
    try:
        if tool == "unitrace":  # VTune and perf collections start disabled
            collection.set(False)
        with torch.no_grad():
            with provenance:
                call = prepare_call(kernel_file, spec_path, variant, reference_path, device)
            for _ in range(warmup):
                call(*call.inputs)
                synchronize()
            record = _timed_calls(call, iters, window, collection.set, synchronize)
            if window_file is not None:
                Path(window_file).write_text(json.dumps(record))
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


def _timed_calls(call, iters, window, collect, synchronize) -> dict:
    """Time the forwards and what the collection held around them. ``window_s`` runs from
    the collector's start command returning to its stop command starting; the collector
    may count from before the start returns to after the stop returns, so
    ``window_max_s`` is that outer span. Both are sums over windows."""
    forward = inner = outer = 0

    def start():
        t0 = time.perf_counter_ns()
        collect(True)
        return t0, time.perf_counter_ns()

    def stop(t0, t1):
        nonlocal inner, outer
        t2 = time.perf_counter_ns()
        collect(False)
        inner += t2 - t1
        outer += time.perf_counter_ns() - t0

    opened = start() if window == "all" else None
    try:
        for _ in range(iters):
            call.reset()
            synchronize()
            each = start() if window == "each" else None
            t = time.perf_counter_ns()
            try:
                call.forward(*call.inputs)
            finally:
                try:
                    synchronize()
                finally:
                    forward += time.perf_counter_ns() - t
                    if each is not None:
                        stop(*each)
    finally:
        if opened is not None:
            stop(*opened)
    return {"window": window, "iters": iters, "forward_s": forward / 1e9,
            "window_s": inner / 1e9, "window_max_s": outer / 1e9,
            "threads": torch.get_num_threads()}  # fmt: skip
