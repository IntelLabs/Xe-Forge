"""The lowering contract: everything an LLM needs to lower one Triton kernel.

A Triton kernel leaves much implicit -- which arguments are pointers, what the
constexprs were bound to, how many work-items a program runs on, where each
argument arrives in the thread payload. :class:`LoweringContext` makes those
facts explicit. It is deliberately a *summary*, not a compiler IR: the point of
the experiment is that the model does the lowering reasoning itself.

What may reach the model is fixed here. :meth:`LoweringContext.to_prompt`
serialises an explicit allowlist of fields. Nothing in this module holds
compiler output for the target kernel. The payload layout is not part of the
contract: the model obtains it from ``abi-probe``, which compiles an inert stub
with the same signature (see :mod:`xe_forge.lowering.visa.compiler`).
:class:`ArtifactGuard` checks every prompt against the text of compiler
artefacts that did exist on disk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ArgInfo:
    """One kernel parameter as Triton sees it."""

    name: str
    kind: str  # "pointer" | "scalar" | "constexpr"
    dtype: str  # element dtype for pointers, value dtype for scalars (Triton spelling)
    value: Any = None  # bound value for constexprs; None otherwise
    address_space: str | None = None  # "global" for pointers


@dataclass(frozen=True)
class LaunchInfo:
    grid: tuple[int, int, int]
    num_warps: int
    threads_per_warp: int
    program_id_dims: tuple[int, ...]

    @property
    def local_size(self) -> tuple[int, int, int]:
        return (self.num_warps * self.threads_per_warp, 1, 1)


@dataclass(frozen=True)
class SemanticSummary:
    """A lightweight reading of the kernel body, used to pick knowledge.

    It records which ``tl`` operations appear and the text of masks. It is not
    an IR and it is not shown to the model as an instruction plan.
    """

    ops: tuple[str, ...]
    masks: tuple[str, ...] = ()
    has_reduction: bool = False
    has_dot: bool = False
    has_transcendental: bool = False
    dtype_casts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetInfo:
    name: str
    device: str
    simd_widths: tuple[int, ...]
    grf_bytes: int
    grf_modes: tuple[int, ...]


@dataclass
class LoweringContext:
    """Everything the lowering agent is allowed to know about one kernel."""

    kernel_name: str
    triton_source: str
    args: list[ArgInfo]
    launch: LaunchInfo
    semantics: SemanticSummary
    target: TargetInfo
    tensor_shapes: dict[str, list[int]] = field(default_factory=dict)
    tensor_strides: dict[str, list[int]] = field(default_factory=dict)
    scalar_values: dict[str, Any] = field(default_factory=dict)
    precision: dict[str, list[float]] = field(default_factory=dict)
    family: str | None = None
    baseline_us: float | None = None

    @property
    def constexprs(self) -> dict[str, Any]:
        return {a.name: a.value for a in self.args if a.kind == "constexpr"}

    # Fields that may appear in a prompt, in order. Anything else on the object
    # (paths, digests, the reference module) stays out by construction.
    PROMPT_FIELDS = (
        "kernel_name",
        "target",
        "launch",
        "arguments",
        "constexprs",
        "example_call",
        "semantics",
        "precision",
    )

    def to_prompt(self) -> str:
        """The contract as YAML, built only from :attr:`PROMPT_FIELDS`."""
        doc: dict[str, Any] = {
            "kernel_name": self.kernel_name,
            "target": asdict(self.target),
            "launch": {
                "grid": list(self.launch.grid),
                "num_warps": self.launch.num_warps,
                "threads_per_warp": self.launch.threads_per_warp,
                "work_group_size": list(self.launch.local_size),
                "program_id_dims_used": list(self.launch.program_id_dims),
                "note": (
                    "This is how the Triton kernel launches: one program = one work-group of "
                    f"{self.launch.local_size[0]} work-items = "
                    f"{self.launch.num_warps} hardware threads of SIMD{self.launch.threads_per_warp}. "
                    "It is the default, not a requirement: you may choose another launch "
                    "(see INSTRUCTIONS.md). The grid varies with the input sizes."
                ),
            },
            "arguments": [
                {k: v for k, v in asdict(a).items() if v is not None and k != "value"}
                for a in self.args
                if a.kind != "constexpr"
            ],
            "constexprs": self.constexprs,
            "example_call": {
                "tensor_shapes": self.tensor_shapes,
                "tensor_strides_elements": self.tensor_strides,
                "scalar_values": self.scalar_values,
                "note": "One observed launch. Runtime sizes and scalars change between calls.",
            },
            "semantics": {
                "tl_ops_used": list(self.semantics.ops),
                "masks": list(self.semantics.masks),
                "reduction": self.semantics.has_reduction,
                "matrix_multiply": self.semantics.has_dot,
                "transcendental": self.semantics.has_transcendental,
            },
            "precision": self.precision,
        }
        text = yaml.safe_dump(doc, sort_keys=False, width=100)
        return "kernel_contract:\n" + _indent(text) + "\ntriton_source: |\n" + _indent(
            self.triton_source.rstrip()
        )


def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.splitlines())


_MIN_LEAK_LINE = 24


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


_NOT_EVIDENCE = ("//", "/*", ";", ".decl", ".input", ".kernel", ".version", ".function")


class LeakDetected(RuntimeError):
    """A prompt contained text taken from a compiler artefact."""


class ArtifactGuard:
    """Refuse prompts that contain text from compiler artefacts.

    The guard is fed every compiler artefact that exists while a run is in
    progress: the ABI stub's IGC dumps, and the reference run's Triton cache
    (TTIR, TTGIR, LLVM IR, SPIR-V text). It stores their normalized lines of at
    least ``min_len`` characters. Any such line found in a prompt is treated as
    a leak. Short lines are ignored, because ``ret (M1, 1)`` belongs to every
    vISA kernel and proves nothing.
    """

    TEXT_SUFFIXES = (".visaasm", ".asm", ".ll", ".llir", ".ttir", ".ttgir", ".spvdis", ".isaasm")

    def __init__(self, min_len: int = _MIN_LEAK_LINE):
        self.min_len = min_len
        self._lines: set[str] = set()
        self._allowed: set[str] = set()
        self.sources: list[str] = []

    def allow(self, text: str) -> None:
        """Exempt lines we publish on purpose (the ABI header rendered from ``.ze_info``)."""
        self._allowed.update(_normalize(line) for line in text.splitlines())

    def add_text(self, text: str, source: str) -> None:
        for line in text.splitlines():
            norm = _normalize(line)
            # Comments and declaration directives are shared by every kernel and are not
            # evidence; a leaked lowering shows up in its instructions.
            if len(norm) >= self.min_len and not norm.startswith(_NOT_EVIDENCE):
                self._lines.add(norm)
        self.sources.append(source)

    def add_tree(self, root: str | Path) -> None:
        root = Path(root)
        if not root.exists():
            return
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in self.TEXT_SUFFIXES:
                try:
                    self.add_text(path.read_text(errors="replace"), str(path))
                except OSError:
                    continue

    def check(self, prompt: str) -> None:
        for line in prompt.splitlines():
            norm = _normalize(line)
            if len(norm) >= self.min_len and norm in self._lines and norm not in self._allowed:
                raise LeakDetected(f"prompt contains a line from a compiler artefact: {norm[:80]!r}")

    def __len__(self) -> int:
        return len(self._lines)
