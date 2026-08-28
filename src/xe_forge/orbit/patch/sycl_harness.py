"""
Spec-driven bench harness for a dispatcher-registered SYCL op (plan §9.7, §11.9).

§9.7 has two halves. The override half exists: `sycl_override.py` builds the
out-of-tree extension that shadows an existing op on the XPU key. The half this module
closes is the harness — v10's status line reads *"the spec-driven bench harness for a
dispatcher-registered SYCL op, correctness and weighted benchmarking through the same
YAML contract, still does not exist."* Without it an optimized SYCL op can be
installed but not judged: it never passes through the correctness gate or the weighted
objective, because those gates consume a `kernel.py` with a `Model`, and a dispatcher
op has neither.

So this module renders exactly that: a `kernel.py` whose module-level `Model` drives
`torch.ops.<namespace>.<name>` through the dispatcher — the same call path the
framework takes — and writes it into the candidate-directory layout
`optimize_kernel_dir` resolves. That is what lets a dispatcher-registered SYCL op be
driven through `optimize_kernel_dir` like any Python kernel, which was the point of
treating language as a dimension rather than a special case (§11.3).

Two constraints shape the generated file:

* **It is self-contained** — torch and stdlib only. A candidate directory travels to
  machines that have torch but not this repository, so the harness cannot import
  xe_forge.
* **An unreachable op fails at construction**, naming the op and every load attempt.
  The lazy alternative — resolving the op at first `forward` — runs cleanly through
  `Model()` and then surfaces as a bare `AttributeError` inside a measurement loop,
  which answers the wrong question in the wrong place.
"""

from __future__ import annotations

import keyword
from collections.abc import Sequence
from pathlib import Path

# The candidate-directory layout `orbit/optimize/kernel_dir.py` resolves. Restated
# here rather than imported so patch/ does not grow a dependency on the optimizer
# package; the names are contract on both sides and pinned by both test suites.
KERNEL_FILE = "kernel.py"
REFERENCE_FILE = "kernel_pytorch.py"
SPEC_FILE = "spec.yaml"

# Only values whose repr round-trips as a literal may be baked into the generated
# harness; anything richer would make the file depend on the generating process.
_LITERAL_TYPES = (bool, int, float, str, bytes, type(None))


def _split_op(op: str) -> tuple[str, str]:
    """Split "namespace::name", defaulting a bare name to aten like sycl_override."""
    namespace, _, name = op.partition("::")
    if not name:
        namespace, name = "aten", op
    if not namespace or not name:
        raise ValueError(f"not a dispatcher op name: {op!r} (expected 'namespace::name')")
    return namespace, name


def _validate_arg_names(arg_names: Sequence[str] | None) -> None:
    """Reject names that would only fail later, as a SyntaxError far from the caller."""
    if not arg_names:
        return
    seen: set[str] = set()
    for candidate in arg_names:
        if not candidate.isidentifier() or keyword.iskeyword(candidate) or candidate == "self":
            raise ValueError(
                f"arg name {candidate!r} is not usable as a forward() parameter; the "
                f"rendered harness would fail at exec time, far from this call"
            )
        if candidate in seen:
            raise ValueError(f"duplicate arg name {candidate!r}")
        seen.add(candidate)


def _literal_fixed_args(fixed_args: Sequence[object] | None) -> tuple[object, ...]:
    fixed = tuple(fixed_args or ())
    for value in fixed:
        if not isinstance(value, _LITERAL_TYPES):
            raise ValueError(
                f"fixed arg {value!r} ({type(value).__name__}) cannot be baked into a "
                f"self-contained harness; only literals survive the repr round-trip"
            )
    return fixed


def render_dispatcher_model(
    op: str,
    *,
    loader_module: str | None = None,
    library_path: str | None = None,
    arg_names: list[str] | None = None,
    fixed_args: Sequence[object] | None = None,
) -> str:
    """Render the `kernel.py` whose `Model` drives a dispatcher-registered op.

    `op` is "namespace::name" as the dispatcher knows it (a bare name defaults to
    `aten`, matching `render_override_source`). Registration is a side effect of
    loading, so the harness is told how the op becomes reachable: `loader_module` is
    imported, or `library_path` is handed to `torch.ops.load_library` — the two forms
    `sycl_override.generate` produces. With neither, the surrounding process must
    already have registered the op.

    `arg_names`, when given, become `forward`'s named parameters so the signature
    mirrors the spec's `params`; otherwise `forward` takes `*tensors`. `fixed_args`
    are trailing non-tensor arguments (an epsilon, a flag) baked in at generation
    time, because the spec's `inputs` section describes tensors and scalars cannot
    arrive through it.

    Construction resolves the op eagerly and raises a specific error naming the op
    and every load attempt if it is unreachable — never an `AttributeError` at first
    `forward`, which would put a configuration failure inside a measurement loop.
    """
    namespace, name = _split_op(op)
    qualified = f"{namespace}::{name}"
    _validate_arg_names(arg_names)
    fixed = _literal_fixed_args(fixed_args)

    if arg_names:
        params = ", ".join(arg_names)
        forward = (
            f"    def forward(self, {params}):\n        return self._op({params}, *FIXED_ARGS)\n"
        )
    else:
        forward = (
            "    def forward(self, *tensors):\n        return self._op(*tensors, *FIXED_ARGS)\n"
        )

    return f'''"""
Bench harness for the dispatcher-registered op {qualified} (plan §9.7, §11.9).

Generated by xe-orbit. This is the harness half of the SYCL kernel contract: `Model`
calls `torch.ops.{namespace}.{name}` through the dispatcher — the same call path the
framework takes — so the spec's correctness and weighted benchmarking gates apply to
a dispatcher-registered op exactly as they do to a Python kernel. The override half,
building the shadowing extension, is `orbit/patch/sycl_override.py`.

An unreachable op fails at construction, naming the op and every load attempt. It
does not fail at first `forward`: that would surface as a bare AttributeError inside
a measurement loop, blaming the measurement for a configuration problem.
"""

import importlib

import torch

NAMESPACE = {namespace!r}
OP_NAME = {name!r}

# How the op becomes reachable. A dispatcher op exists only after the module or
# shared object that registers it has been loaded into this process, so loading is
# not setup — it is the registration itself.
LOADER_MODULE = {loader_module!r}
LIBRARY_PATH = {library_path!r}

# Trailing non-tensor arguments, baked in at generation time: the spec's `inputs`
# section describes tensors, so scalars like an epsilon cannot arrive through it.
FIXED_ARGS = {fixed!r}


def _resolve_op():
    """Make the op reachable, then return its overload packet — or say exactly why not.

    Every attempt is recorded, success and failure alike, because "op not registered"
    is only actionable when the error also says what was already tried.
    """
    attempts = []
    if LOADER_MODULE is not None:
        try:
            importlib.import_module(LOADER_MODULE)
            attempts.append(f"imported loader module {{LOADER_MODULE!r}}")
        except Exception as error:
            attempts.append(f"importing loader module {{LOADER_MODULE!r}} failed: {{error}}")
    if LIBRARY_PATH is not None:
        try:
            torch.ops.load_library(LIBRARY_PATH)
            attempts.append(f"loaded {{LIBRARY_PATH!r}} via torch.ops.load_library")
        except Exception as error:
            attempts.append(f"torch.ops.load_library({{LIBRARY_PATH!r}}) failed: {{error}}")
    if not attempts:
        attempts.append(
            "nothing was loaded (this harness was generated without a loader module "
            "or library path), so the op must be registered by the surrounding process"
        )
    try:
        return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            f"operator {{NAMESPACE}}::{{OP_NAME}} is not registered with the "
            "dispatcher. Tried: " + "; ".join(attempts) + ". Pass loader_module or "
            "library_path when rendering this harness, or import the registering "
            "module before constructing Model."
        ) from error


class Model(torch.nn.Module):
    """Launches {qualified} through the dispatcher.

    Xe-Forge resolves `Model` by duck typing — a module-level attribute, constructed
    no-arg — so nothing here imports beyond torch and the standard library, and the
    file stays valid on machines that have torch but not xe-orbit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = _resolve_op()

{forward}'''


def _reference_is_a_stub(source: str) -> bool:
    """A stub raises rather than computing; treating it as real is a trap.

    Mirrors `kernel_dir._reference_is_a_stub` on source text instead of a path —
    deliberately restated rather than imported, for the same reason the layout
    constants are: patch/ must not depend on the optimizer package.
    """
    return "NotImplementedError" in source


def emit_dispatcher_candidate(
    op: str,
    target_dir: Path | str,
    *,
    spec_source: str | None = None,
    reference_source: str | None = None,
    loader_module: str | None = None,
    library_path: str | None = None,
    arg_names: list[str] | None = None,
    fixed_args: Sequence[object] | None = None,
) -> dict[str, object]:
    """Write the candidate directory `optimize_kernel_dir` resolves (plan §8, §9.7).

    candidates/<op>/
        kernel.py            the dispatcher harness — always written
        kernel_pytorch.py    eager reference — only when the caller supplies one
        spec.yaml            inputs and weighted bench variants — only when supplied

    Nothing is fabricated for a missing piece. A guessed reference would pass a
    meaningless correctness gate and be wrong in the model — the exact failure the
    correctness ladder exists to prevent (§19) — so a missing reference or spec is
    written into the summary's notes instead of papered over, and `resolve_candidate`
    downstream keeps its honest refusals.

    Returns a summary with the written paths (`None` for pieces not written) and the
    notes explaining what a caller still owes the candidate.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    namespace, name = _split_op(op)
    qualified = f"{namespace}::{name}"
    notes: list[str] = []

    kernel_path = target / KERNEL_FILE
    kernel_path.write_text(
        render_dispatcher_model(
            op,
            loader_module=loader_module,
            library_path=library_path,
            arg_names=arg_names,
            fixed_args=fixed_args,
        ),
        encoding="utf-8",
    )

    reference_path: Path | None = None
    if reference_source is not None:
        reference_path = target / REFERENCE_FILE
        reference_path.write_text(reference_source, encoding="utf-8")
        if _reference_is_a_stub(reference_source):
            notes.append(
                f"{REFERENCE_FILE} is a stub — it raises rather than computing — so "
                f"the correctness gate will raise rather than compare; supply the "
                f"eager-mode equivalent before trusting any result"
            )
    else:
        notes.append(
            f"no {REFERENCE_FILE}: Orbit does not fabricate a reference — a plausible "
            f"but wrong one passes a meaningless gate — so correctness cannot be "
            f"checked until the eager-mode equivalent of {qualified} is supplied"
        )

    spec_path: Path | None = None
    if spec_source is not None:
        spec_path = target / SPEC_FILE
        spec_path.write_text(spec_source, encoding="utf-8")
    else:
        notes.append(
            f"no {SPEC_FILE}: optimize_kernel_dir refuses a candidate without one, so "
            f"the spec must be written into the directory before handing it over"
        )

    return {
        "op": qualified,
        "kernel_path": str(kernel_path),
        "reference_path": str(reference_path) if reference_path else None,
        "spec_path": str(spec_path) if spec_path else None,
        "notes": notes,
    }
