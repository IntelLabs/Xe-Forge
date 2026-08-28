"""
Real input capture: saves the running workload's actual tensors, preserving strides,
contiguity and non-code data dependencies (tuned configs, scales, block tables) that
synthetic inputs get wrong. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from xe_forge.orbit.models import CapturedInvocation


class CaptureError(RuntimeError):
    """Raised when inputs cannot be captured faithfully."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on install
        raise CaptureError("torch is required to capture real inputs") from exc
    return torch


def describe_tensor(tensor: Any) -> dict[str, Any]:
    """Record everything about a tensor that changes kernel behaviour."""
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "contiguous": bool(tensor.is_contiguous()),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def capture_invocation(
    kernel_id: str,
    args: dict[str, Any],
    output_dir: Path,
    *,
    call_index: int = 0,
    reference_output: Any = None,
    data_deps: list[Path] | None = None,
    preserve_strides: bool = True,
    max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> CapturedInvocation:
    """Save one real invocation — tensors, scalars, layout and reference output.

    `args` maps parameter name to value; tensors are saved, everything else is
    recorded as a scalar. Names are preserved because the extracted kernel's `Model`
    must bind them back in the right order.
    """
    torch = _require_torch()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    tensors: list[str] = []
    scalars: dict[str, Any] = {}
    dtype_map: dict[str, str] = {}
    shape_map: dict[str, list[int]] = {}
    stride_map: dict[str, list[int]] = {}
    contiguous_map: dict[str, bool] = {}
    total_bytes = 0

    for name, value in args.items():
        if not isinstance(value, torch.Tensor):
            if isinstance(value, (int, float, bool, str)) or value is None:
                scalars[name] = value
            else:
                scalars[name] = repr(value)
            continue

        meta = describe_tensor(value)
        nbytes = value.element_size() * value.nelement()
        total_bytes += nbytes
        if total_bytes > max_bytes:
            raise CaptureError(
                f"captured tensors exceed {max_bytes} bytes at {name!r}; capture fewer "
                f"invocations or raise max_bytes deliberately"
            )

        path = target / f"{name}.pt"
        cpu_tensor = value.detach().to("cpu")
        if preserve_strides and not meta["contiguous"]:
            # Saving the tensor as-is keeps the view's stride pattern. Reloading with
            # `load_invocation` reconstructs it rather than silently contiguifying,
            # which is what makes a transposed or sliced input survive the round trip.
            torch.save({"data": cpu_tensor, "stride": meta["stride"], "shape": meta["shape"]}, path)
        else:
            torch.save(cpu_tensor, path)

        tensors.append(str(path))
        dtype_map[name] = meta["dtype"]
        shape_map[name] = meta["shape"]
        stride_map[name] = meta["stride"]
        contiguous_map[name] = meta["contiguous"]

    reference_path: str | None = None
    if reference_output is not None:
        reference_path = str(target / "reference_out.pt")
        payload = (
            reference_output.detach().to("cpu")
            if isinstance(reference_output, torch.Tensor)
            else reference_output
        )
        torch.save(payload, reference_path)

    copied_deps: list[str] = []
    for dep in data_deps or []:
        source = Path(dep)
        if not source.is_file():
            raise CaptureError(f"declared data dependency does not exist: {source}")
        destination = target / "data" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_deps.append(str(destination))

    invocation = CapturedInvocation(
        kernel_id=kernel_id,
        call_index=call_index,
        tensors=tensors,
        scalars=scalars,
        dtype_map=dtype_map,
        shape_map=shape_map,
        stride_map=stride_map,
        contiguous_map=contiguous_map,
        output_reference=reference_path,
        data_deps=copied_deps,
    )

    (target / "invocation.json").write_text(
        json.dumps(invocation.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return invocation


def load_invocation(invocation: CapturedInvocation, device: str = "cpu") -> dict[str, Any]:
    """Reload a captured invocation, restoring strides and non-contiguous layouts.

    This is what `Model.get_example_inputs()` calls. A tensor saved from a
    non-contiguous view is reconstructed with its original stride pattern, so the
    benchmark exercises the same memory access the workload did.
    """
    torch = _require_torch()
    restored: dict[str, Any] = dict(invocation.scalars)

    for path_str in invocation.tensors:
        path = Path(path_str)
        name = path.stem
        if not path.is_file():
            raise CaptureError(f"captured tensor missing: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)

        if isinstance(payload, dict) and "data" in payload:
            tensor = payload["data"].to(device)
            stride = payload.get("stride")
            shape = payload.get("shape")
            if stride and shape:
                tensor = torch.as_strided(tensor, tuple(shape), tuple(stride))
        else:
            tensor = payload.to(device)

        restored[name] = tensor

    return restored


def verify_roundtrip(invocation: CapturedInvocation, device: str = "cpu") -> list[str]:
    """Reload captured inputs and confirm shape, dtype and contiguity survived.

    Returns the list of discrepancies; empty means the capture is faithful.
    """
    problems: list[str] = []
    restored = load_invocation(invocation, device=device)

    for name, expected_shape in invocation.shape_map.items():
        tensor = restored.get(name)
        if tensor is None:
            problems.append(f"{name}: missing after reload")
            continue
        if list(tensor.shape) != list(expected_shape):
            problems.append(f"{name}: shape {list(tensor.shape)} != {expected_shape}")
        expected_stride = invocation.stride_map.get(name)
        if expected_stride and list(tensor.stride()) != list(expected_stride):
            problems.append(f"{name}: stride {list(tensor.stride())} != {expected_stride}")
        expected_contig = invocation.contiguous_map.get(name)
        if expected_contig is not None and bool(tensor.is_contiguous()) != expected_contig:
            problems.append(f"{name}: contiguity {tensor.is_contiguous()} != {expected_contig}")
        expected_dtype = invocation.dtype_map.get(name)
        if expected_dtype and str(tensor.dtype).replace("torch.", "") != expected_dtype:
            problems.append(f"{name}: dtype {tensor.dtype} != {expected_dtype}")

    return problems
