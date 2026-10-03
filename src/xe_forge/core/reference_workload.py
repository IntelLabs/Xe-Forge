"""Reference-owned model preparation, isolated inputs and state validation."""

import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from ai_bench.harness.runner.benchmark_compare import set_all_seeds

from xe_forge.core.integrity import check_trial, record_oracle


@dataclass
class PreparedCall:
    model: Callable
    inputs: Any
    initial_buffers: dict[str, torch.Tensor] | None = None

    def reset(self) -> None:
        if self.initial_buffers is not None:
            _restore_buffers(self.model, self.initial_buffers)

    def forward(self, *arguments):
        return self.model(*arguments)

    def __call__(self, *arguments):
        self.reset()
        return self.forward(*arguments)


@dataclass
class ReferenceWorkload:
    reference: torch.nn.Module
    original: torch.nn.Module
    optimized: torch.nn.Module
    inputs: Any
    rtol: float
    atol: float
    device: str = "cpu"
    get_inputs: Callable | None = None
    workload: dict | None = None

    @classmethod
    def prepare(
        cls,
        compile_module: Callable,
        reference_code: str,
        original_code: str | None,
        optimized_code: str,
        device: str,
        rtol: float,
        atol: float,
        workload: dict | None = None,
    ) -> "ReferenceWorkload":
        reference_module = compile_module(reference_code)
        if reference_module is None or not callable(getattr(reference_module, "get_inputs", None)):
            raise ValueError("Reference must define Model and get_inputs()")
        # A spec variant's dims reach the reference as keyword arguments; one that
        # neither function takes would describe a workload that is not the one run.
        workload = workload or {}
        get_init_inputs = getattr(reference_module, "get_init_inputs", None)
        builders = [f for f in (get_init_inputs, reference_module.get_inputs) if callable(f)]
        unused = set(workload).difference(*(_parameters(f) for f in builders))
        if unused:
            raise ValueError(
                f"spec dim {', '.join(sorted(unused))} is not a parameter of the "
                "reference's get_inputs/get_init_inputs"
            )
        set_all_seeds(42)
        init_args = _call_with(get_init_inputs, workload) if callable(get_init_inputs) else []
        models = []
        sources = (
            (reference_code, optimized_code)
            if original_code is None
            else (reference_code, original_code, optimized_code)
        )
        for index, code in enumerate(sources):
            module = reference_module if index == 0 else compile_module(code)
            if module is None or not hasattr(module, "Model"):
                raise ValueError("Every implementation must define Model")
            set_all_seeds(42)
            model = module.Model(*deepcopy(init_args))
            models.append(model.cpu().eval())

        reference, optimized = models[0], models[-1]
        original = reference if original_code is None else models[1]
        for model in models[1:]:
            model.load_state_dict(reference.state_dict(), strict=True)

        set_all_seeds(123)
        inputs = _snapshot(_call_with(reference_module.get_inputs, workload))
        return cls(
            reference,
            original,
            optimized,
            inputs,
            rtol,
            atol,
            device,
            get_inputs=reference_module.get_inputs,
            workload=workload,
        )

    def copy_inputs(self) -> Any:
        return deepcopy(self.inputs)

    def measure(self, model: torch.nn.Module, timer: Callable) -> float:
        try:
            call = self.prepare_call(model)
            return timer(call, tuple(call.inputs))
        finally:
            model.cpu()

    def prepare_call(self, model: torch.nn.Module) -> PreparedCall:
        """Prepare fixed inputs and explicit state restoration for repeated calls."""
        model.to(self.device)
        inputs = _move_inputs(self.copy_inputs(), self.device)
        initial_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        return PreparedCall(model, inputs, initial_buffers)

    def check_integrity(self) -> list[str]:
        """The trial's integrity failures against the reference; see :mod:`integrity`."""

        def make_inputs(seed: int):
            set_all_seeds(seed)
            return _move_inputs(_call_with(self.get_inputs, self.workload or {}), self.device)

        # One model on the device at a time, and each left with the buffers it started
        # with, as validate() does: timing snapshots them as the initial state.
        call = self.prepare_call(self.reference)
        try:
            record = record_oracle(call, make_inputs)
        finally:
            call.reset()
            self.reference.cpu()
        call = self.prepare_call(self.optimized)
        try:
            return check_trial(record, call, make_inputs, self._values_match, self.device)
        finally:
            call.reset()
            self.optimized.cpu()

    def validate(self) -> None:
        models = (
            (("reference", self.reference), ("optimized", self.optimized))
            if self.original is self.reference
            else (
                ("reference", self.reference),
                ("baseline", self.original),
                ("optimized", self.optimized),
            )
        )
        for name, model in models[1:]:
            self._assert_values_match(
                dict(self.reference.named_buffers()),
                dict(model.named_buffers()),
                f"Initialized buffers differ from the reference ({name})",
            )

        reference_steps = []
        for model_index, (name, model) in enumerate(models):
            initial_buffers = _snapshot(dict(model.named_buffers()))
            failed = False
            try:
                model.to(self.device)
                for iteration in range(2):
                    _restore_buffers(model, initial_buffers)
                    arguments = _move_inputs(self.copy_inputs(), self.device)
                    output = model(*arguments)
                    output_devices = _tensor_devices(output)
                    output_snapshot = _snapshot(output)
                    del output
                    if not self._values_match(self.inputs, _snapshot(arguments)):
                        raise ValueError("Mutating inputs requires a host benchmark with resets")
                    del arguments
                    buffers = _snapshot(dict(model.named_buffers()))
                    if model_index == 0:
                        reference_steps.append((output_snapshot, output_devices, buffers))
                    else:
                        expected_output, expected_devices, expected_buffers = reference_steps[
                            iteration
                        ]
                        context = f"{name}, call {iteration + 1}"
                        if output_devices != expected_devices:
                            raise ValueError(
                                f"Output differs from the semantic reference ({context}): "
                                f"devices expected={expected_devices}, actual={output_devices}"
                            )
                        self._assert_values_match(
                            expected_output,
                            output_snapshot,
                            f"Output differs from the semantic reference ({context})",
                        )
                        self._assert_values_match(
                            expected_buffers,
                            buffers,
                            f"Updated buffers differ from the reference ({context})",
                        )
            except Exception as error:
                failed = True
                raise ValueError(f"Validation failed for {name}: {error}") from error
            finally:
                try:
                    model.cpu()
                    _restore_buffers(model, initial_buffers)
                except Exception:
                    if not failed:
                        raise

    def _assert_values_match(self, reference, candidate, message):
        if self._values_match(reference, candidate):
            return
        try:
            torch.testing.assert_close(
                _upcast_fp8(candidate), _upcast_fp8(reference), rtol=self.rtol, atol=self.atol
            )
        except (AssertionError, TypeError, ValueError) as error:
            raise ValueError(f"{message}: {error}") from error
        raise ValueError(f"{message}: value types or structure differ")

    def _values_match(self, reference, candidate, exact=False):
        if isinstance(reference, torch.Tensor):
            if not isinstance(candidate, torch.Tensor):
                return False
            if (
                reference.shape != candidate.shape
                or reference.dtype != candidate.dtype
                or reference.device != candidate.device
            ):
                return False
            if exact or not (reference.is_floating_point() or reference.is_complex()):
                return torch.equal(reference, candidate)
            if reference.dtype in _FP8:
                # allclose has no float8 kernel; one fp8 step is rounding, not a wrong value.
                info = torch.finfo(reference.dtype)
                return torch.allclose(
                    candidate.float(),
                    reference.float(),
                    rtol=max(self.rtol, info.eps),
                    atol=max(self.atol, info.tiny),
                )
            return torch.allclose(candidate, reference, rtol=self.rtol, atol=self.atol)
        if type(reference) is not type(candidate):
            return False
        if isinstance(reference, dict):
            return reference.keys() == candidate.keys() and all(
                self._values_match(value, candidate[key], exact=exact)
                for key, value in reference.items()
            )
        if isinstance(reference, (tuple, list)):
            return len(reference) == len(candidate) and all(
                self._values_match(left, right, exact=exact)
                for left, right in zip(reference, candidate, strict=True)
            )
        return reference == candidate


def _restore_buffers(model, saved):
    buffers = dict(model.named_buffers())
    if buffers.keys() != saved.keys():
        raise ValueError("Buffer registration changed during execution")
    for name, target in buffers.items():
        source = saved[name]
        if target.shape != source.shape or target.dtype != source.dtype:
            raise ValueError(f"Buffer shape or dtype changed during execution: {name}")
        target.copy_(source)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return deepcopy(value)


def _tensor_devices(value):
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        return {key: _tensor_devices(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_devices(item) for item in value]
    return None


def _move_inputs(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_inputs(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_inputs(item, device) for item in value)
    if isinstance(value, list):
        return [_move_inputs(item, device) for item in value]
    return deepcopy(value)


_FP8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}


def _upcast_fp8(value):
    if isinstance(value, torch.Tensor) and value.dtype in _FP8:
        return value.float()
    return value


def _parameters(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def _call_with(fn, workload: dict):
    names = _parameters(fn)
    return fn(**{key: value for key, value in workload.items() if key in names})
