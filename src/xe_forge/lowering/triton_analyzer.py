"""Extract a lowering contract from a Triton kernel module.

Two sources, used together:

- the **source** of the ``@triton.jit`` function, read with ``ast`` for the
  operations it uses and the text of its masks (no execution needed);
- one **launch**, captured by wrapping ``JITFunction.run`` while the module's
  ``Model.forward`` runs once on spec inputs. It gives the bound constexprs, the
  argument kinds and dtypes, the evaluated grid and ``num_warps``. Compiled
  artefacts of the launch are not read.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xe_forge.lowering.context import (
    ArgInfo,
    LaunchInfo,
    LoweringContext,
    SemanticSummary,
    TargetInfo,
)

_REDUCTIONS = {"sum", "max", "min", "reduce", "argmax", "argmin", "xor_sum", "cumsum", "associative_scan"}
_TRANSCENDENTAL = {"exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sin", "cos", "sigmoid", "erf", "tanh", "div_rn"}

_TORCH_TO_TL = {
    "torch.float32": "fp32",
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float64": "fp64",
    "torch.int8": "i8",
    "torch.uint8": "u8",
    "torch.int16": "i16",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.bool": "i1",
    "torch.float8_e4m3fn": "fp8e4nv",
    "torch.float8_e5m2": "fp8e5",
}


def load_module(path: str | Path, name: str | None = None):
    """Import a kernel module from a file path (Triton needs the source on disk)."""
    path = Path(path).resolve()
    mod_name = name or f"xe_forge_lowering_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def jit_functions(module) -> dict[str, Any]:
    """The ``@triton.jit`` functions a module defines, by name."""
    from triton.runtime.jit import JITFunction

    return {name: obj for name, obj in vars(module).items() if isinstance(obj, JITFunction)}


def kernel_source(jit_fn) -> str:
    return textwrap.dedent(inspect.getsource(jit_fn.fn))


def summarize_source(source: str) -> SemanticSummary:
    """Ops, masks and casts used in a kernel body, read from its AST."""
    tree = ast.parse(textwrap.dedent(source))
    ops: list[str] = []
    masks: list[str] = []
    casts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name.startswith(("tl.", "triton.language.")):
            short = "tl." + name.split(".")[-1]
            if short not in ops:
                ops.append(short)
        elif name == "to" or name.endswith(".to"):
            casts.append(ast.unparse(node))
        for kw in node.keywords:
            if kw.arg in ("mask", "other"):
                text = f"{kw.arg}={ast.unparse(kw.value)}"
                if text not in masks:
                    masks.append(text)
    leaf = {op.split(".")[-1] for op in ops}
    return SemanticSummary(
        ops=tuple(ops),
        masks=tuple(masks),
        has_reduction=bool(leaf & _REDUCTIONS),
        has_dot="dot" in leaf,
        has_transcendental=bool(leaf & _TRANSCENDENTAL),
        dtype_casts=tuple(casts),
    )


def program_id_dims(source: str) -> tuple[int, ...]:
    tree = ast.parse(textwrap.dedent(source))
    dims: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func).endswith("program_id"):
            arg = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "axis"), None
            )
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                dims.add(arg.value)
    return tuple(sorted(dims))


def _call_name(func: ast.AST) -> str:
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


@dataclass
class LaunchRecord:
    """One captured ``kernel[grid](...)`` call."""

    kernel: str
    args: list[ArgInfo]
    grid: tuple[int, int, int]
    num_warps: int
    threads_per_warp: int
    tensor_shapes: dict[str, list[int]] = field(default_factory=dict)
    tensor_strides: dict[str, list[int]] = field(default_factory=dict)
    scalar_values: dict[str, Any] = field(default_factory=dict)

    @property
    def constexprs(self) -> dict[str, Any]:
        return {a.name: a.value for a in self.args if a.kind == "constexpr"}

    def to_json(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> LaunchRecord:
        d = dict(d)
        d["args"] = [ArgInfo(**a) for a in d["args"]]
        d["grid"] = tuple(d["grid"])
        return cls(**d)


def describe_args(jit_fn, args: tuple, kwargs: dict) -> tuple[list[ArgInfo], dict, dict, dict]:
    """Classify each bound parameter of a launch."""
    import torch

    bound = dict(zip(jit_fn.arg_names, args))
    bound.update({k: v for k, v in kwargs.items() if k in jit_fn.arg_names})
    infos: list[ArgInfo] = []
    shapes: dict[str, list[int]] = {}
    strides: dict[str, list[int]] = {}
    scalars: dict[str, Any] = {}
    for param in jit_fn.params:
        name = param.name
        if name not in bound and param.has_default:
            bound[name] = param.default
        value = bound.get(name)
        if param.is_constexpr:
            infos.append(ArgInfo(name, "constexpr", type(value).__name__, value=_plain(value)))
        elif isinstance(value, torch.Tensor):
            infos.append(ArgInfo(name, "pointer", _TORCH_TO_TL.get(str(value.dtype), str(value.dtype)), address_space="global"))
            shapes[name] = list(value.shape)
            strides[name] = list(value.stride())
        elif isinstance(value, bool):
            infos.append(ArgInfo(name, "scalar", "i1"))
            scalars[name] = value
        elif isinstance(value, int):
            infos.append(ArgInfo(name, "scalar", "i32" if -(2**31) <= value < 2**31 else "i64"))
            scalars[name] = value
        elif isinstance(value, float):
            infos.append(ArgInfo(name, "scalar", "fp32"))
            scalars[name] = value
        elif value is None:
            infos.append(ArgInfo(name, "scalar", "none"))
        else:
            raise TypeError(f"argument {name!r} of type {type(value).__name__} is not supported for lowering")
    return infos, shapes, strides, scalars


def _plain(value):
    if hasattr(value, "value"):  # tl.constexpr wrapper
        value = value.value
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def eval_grid(grid, jit_fn, args: tuple, kwargs: dict) -> tuple[int, int, int]:
    if callable(grid):
        meta = dict(zip(jit_fn.arg_names, args))
        meta.update(kwargs)
        grid = grid(meta)
    grid = tuple(int(g) for g in grid)
    return grid + (1,) * (3 - len(grid))


@contextmanager
def capture_launches(records: list[LaunchRecord]):
    """Record every ``JITFunction.run`` launch made inside the block, then run it."""
    from triton.runtime.jit import JITFunction

    original = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        kernel = original(self, *args, grid=grid, warmup=warmup, **kwargs)
        if not warmup:
            infos, shapes, strides, scalars = describe_args(self, args, kwargs)
            md = kernel.metadata
            records.append(
                LaunchRecord(
                    kernel=self.__name__,
                    args=infos,
                    grid=eval_grid(grid, self, args, kwargs),
                    num_warps=md.num_warps,
                    threads_per_warp=getattr(md, "threads_per_warp", 32),
                    tensor_shapes=shapes,
                    tensor_strides=strides,
                    scalar_values=scalars,
                )
            )
        return kernel

    JITFunction.run = run
    try:
        yield records
    finally:
        JITFunction.run = original


def select_launch(records: list[LaunchRecord], kernel: str | None) -> LaunchRecord:
    """The launch to lower: the named kernel, or the only kernel launched."""
    names = sorted({r.kernel for r in records})
    if not records:
        raise ValueError("Model.forward launched no Triton kernel; nothing to lower")
    if kernel:
        matching = [r for r in records if r.kernel == kernel]
        if not matching:
            raise ValueError(f"kernel {kernel!r} was not launched; launched: {names}")
        return matching[0]
    if len(names) > 1:
        raise ValueError(f"Model.forward launches several kernels {names}; choose one with --lower-kernel")
    return records[0]


def build_context(
    record: LaunchRecord,
    source: str,
    target: TargetInfo,
    *,
    precision: dict[str, list[float]] | None = None,
    family: str | None = None,
) -> LoweringContext:
    return LoweringContext(
        kernel_name=record.kernel,
        triton_source=source,
        args=record.args,
        launch=LaunchInfo(
            grid=record.grid,
            num_warps=record.num_warps,
            threads_per_warp=record.threads_per_warp,
            program_id_dims=program_id_dims(source),
        ),
        semantics=summarize_source(source),
        target=target,
        tensor_shapes=record.tensor_shapes,
        tensor_strides=record.tensor_strides,
        scalar_values=record.scalar_values,
        precision=precision or {},
        family=family,
    )
