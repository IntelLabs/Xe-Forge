"""
Patch-back: getting the optimized kernel into the running workload (plan §13).

This is, with extraction, the most likely place for the project to stall. v1 gave it
one line. It needs a mechanism ladder, and the rung matters:

    P1  Operator override through the PyTorch dispatcher   touches nothing
    P2  Custom op + Inductor post-grad pattern replacement  compile pipeline only
    P3  Framework registry substitution                     framework config
    P4  Import-time module shim (recorded monkey-patch)     process state
    P5  Source patch and rebuild                            source tree + build

**Always take the highest rung that works**, because higher rungs touch less and revert
cleanly. P1 is the default: the optimized kernel becomes a small importable module that
registers itself, the framework is left entirely untouched, and reverting is just not
importing it. It also ports across frameworks for free — the same override works under
plain PyTorch, vLLM and SGLang if all three dispatch through the same op — and it
covers SYCL as well as Triton, because torch-xpu-ops, IPEX, vLLM-XPU and sgl-kernel-xpu
all register their kernels as dispatcher ops (§11.8).

**Do not write into the Inductor cache.** Those paths are content-hashed and regenerate
on every recompile, config change and version bump; a patch against generated code is
not a durable artifact.

The non-negotiable part is verification. An override that silently fails to take effect
— wrong dispatch key, wrong overload, registration after the first call, a
`torch.compile` graph captured before registration — produces a clean "no change"
result that looks exactly like an honest negative. So verification is a **dispatch
assertion**, not an inspection: re-profile, and confirm the new kernel appears in the
trace *and the old one does not*.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field

from xe_forge.orbit.models import Artifact, KernelRecord, PatchPoint


class PatchError(RuntimeError):
    """Raised when a patch cannot be applied, or cannot be proven to have applied."""


# Ordered best-first. `apply` walks this list and takes the first rung that is both
# available for the kernel and supported by the environment.
RUNG_ORDER = ("P1", "P2", "P3", "P4", "P5")

_RUNG_DESCRIPTION = {
    "P1": "operator override through the PyTorch dispatcher",
    "P2": "custom op + Inductor post-grad pattern replacement",
    "P3": "framework registry substitution",
    "P4": "import-time module shim",
    "P5": "source patch and rebuild",
}

_RUNG_TOUCHES = {
    "P1": "nothing in the framework",
    "P2": "the compile pipeline only",
    "P3": "framework configuration",
    "P4": "process state",
    "P5": "the source tree and its build",
}


class AppliedPatch(Artifact):
    """The record every applied patch leaves behind (§13).

    Records rung, target symbol, the registration call, the revert procedure and the
    verification result — so a patch can always be undone and always be explained.
    """

    kernel_id: str
    rung: str
    target: str
    mechanism: str = ""
    module_path: str | None = None
    registration_call: str = ""
    revert_procedure: str = ""
    applied: bool = False
    verified: bool = False
    verification_detail: str = ""
    worktree: str | None = None
    env_overrides: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


@dataclass
class DispatchAssertion:
    """Outcome of re-profiling after a patch (§13).

    Both halves matter. `new_kernel_present` alone is not enough: a workload that runs
    both the old and new kernel has not been patched, it has been made slower.
    """

    new_kernel_present: bool = False
    old_kernel_absent: bool = False
    observed_kernels: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def took_effect(self) -> bool:
        return self.new_kernel_present and self.old_kernel_absent


def available_rungs(kernel: KernelRecord, patch_points: list[PatchPoint]) -> list[PatchPoint]:
    """Patch points for this kernel, ordered highest rung first."""
    ordered = sorted(
        patch_points,
        key=lambda p: RUNG_ORDER.index(p.rung) if p.rung in RUNG_ORDER else len(RUNG_ORDER),
    )
    return ordered


def choose_rung(kernel: KernelRecord, patch_points: list[PatchPoint]) -> PatchPoint:
    """Pick the highest rung that applies, or explain why none does."""
    ordered = available_rungs(kernel, patch_points)
    if not ordered:
        raise PatchError(
            f"no patch point available for {kernel.id} ({kernel.runtime_name}). "
            f"Provider {kernel.provider.value} with no registered op cannot be reached "
            f"by operator override; for an opaque library primitive the action is a "
            f"backend or config change, not source replacement (§13)."
        )
    return ordered[0]


def render_operator_override(
    kernel: KernelRecord,
    op_name: str,
    implementation_module: str,
    device_key: str = "XPU",
) -> str:
    """Generate the P1 override module.

    The output is a small importable module that registers an implementation for an
    existing op on the device key, shadowing the default. Nothing in the framework is
    modified; reverting is not importing this module.
    """
    namespace, _, op = op_name.partition("::")
    if not op:
        namespace, op = "aten", op_name

    return f'''"""
Operator override for {op_name} (plan §13, rung P1).

Generated by xe-orbit. This module registers an alternative implementation for an
existing dispatcher op on the {device_key} key, shadowing the default. It touches
nothing in the framework: no fork, no vendored patch, no rebuild.

To apply:   import this module before the first call to the op.
To revert:  do not import it.

The registration must happen before the op is first dispatched, and before any
`torch.compile` graph that contains it is captured — a graph captured earlier will
have already specialized on the original implementation, and the override will
silently not take effect. That failure looks exactly like an honest "no change"
result, which is why the dispatch assertion in §13 is mandatory rather than optional.
"""

import torch

from {implementation_module} import optimized_kernel

_LIBRARY = torch.library.Library({namespace!r}, "IMPL")


def _dispatch(*args, **kwargs):
    return optimized_kernel(*args, **kwargs)


_LIBRARY.impl({op!r}, _dispatch, {device_key!r})

REGISTRATION = "torch.library.Library({namespace!r}, 'IMPL').impl({op!r}, ..., {device_key!r})"
TARGET_OP = {op_name!r}
'''


def _is_native(kernel: KernelRecord) -> bool:
    """True when the override must be compiled rather than imported."""
    language = getattr(kernel.language, "value", None)
    return language in ("sycl", "sycl_tla", "cpp")


def apply_patch(
    kernel: KernelRecord,
    patch_points: list[PatchPoint],
    candidate_module: str,
    output_dir: Path,
    worktree: Path | None = None,
    device_name: str | None = None,
    build_native: bool = True,
) -> AppliedPatch:
    """Apply the highest available rung and write the patch record.

    This writes the override module; it does not import it. Applying a patch into the
    measuring process would contaminate the baseline, so the module is written here and
    imported by the workload under test.
    """
    point = choose_rung(kernel, patch_points)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    record = AppliedPatch(
        kernel_id=kernel.id,
        rung=point.rung,
        target=point.target,
        mechanism=point.mechanism or _RUNG_DESCRIPTION.get(point.rung, ""),
        worktree=str(worktree) if worktree else None,
    )

    if point.rung == "P1" and _is_native(kernel):
        # SYCL and C++ ops reach the dispatcher the same way, but the override has to
        # be compiled first. Everything after that — the assertion, the revert, the
        # version binding — is identical, which is why language is a dimension here
        # rather than a special case (§11.3, §11.8).
        from xe_forge.orbit.patch.sycl_override import generate

        artifacts = generate(
            kernel,
            point.target,
            target,
            device_name=device_name,
            build=build_native,
        )
        record.module_path = str(artifacts.loader_path)
        record.registration_call = (
            f"TORCH_LIBRARY_IMPL({point.target.split('::')[0]}, XPU, m) "
            f'{{ m.impl("{point.target.split("::")[-1]}", ...); }}'
        )
        record.revert_procedure = (
            f"do not import {artifacts.loader_path.name}; the extension is inert unless "
            f"torch.ops.load_library is called. No framework source was modified."
        )
        record.env_overrides = {"PYTHONPATH": str(target)}
        record.applied = artifacts.built
        record.notes.append(
            f"P1 for a native op: out-of-tree extension compiled with "
            f"{artifacts.build.compiler}, no fork of PyTorch, vLLM or SGLang (§11.8)."
        )
        if not artifacts.built:
            record.notes.append(f"not compiled: {artifacts.reason}")
        if artifacts.build.aot_target:
            record.notes.append(f"AOT target pinned to {artifacts.build.aot_target}")
        return record

    if point.rung == "P1":
        module_source = render_operator_override(kernel, point.target, candidate_module)
        module_path = target / f"orbit_override_{kernel.id}.py"
        module_path.write_text(module_source, encoding="utf-8")
        record.module_path = str(module_path)
        record.registration_call = f"torch.library.Library(...).impl({point.target!r}, ...)"
        record.revert_procedure = (
            f"do not import {module_path.name}; no framework state was modified"
        )
        record.env_overrides = {"PYTHONPATH": str(target)}
        record.applied = True
        record.notes.append(
            f"P1 touches {_RUNG_TOUCHES['P1']}. The same override ports to any framework "
            f"that dispatches through {point.target}."
        )
        return record

    # Rungs below P1 are described but not auto-applied: each one modifies something
    # (the compile pipeline, framework config, process state, or the source tree), and
    # doing that implicitly is how an experiment becomes unreproducible.
    record.applied = False
    record.revert_procedure = "n/a — not applied"
    record.notes.append(
        f"rung {point.rung} ({_RUNG_DESCRIPTION.get(point.rung, 'unknown')}) touches "
        f"{_RUNG_TOUCHES.get(point.rung, 'unknown state')} and is not applied "
        f"automatically. Apply it deliberately and record the change."
    )
    return record


def assert_dispatch(
    observed_kernels: list[str],
    original_kernel: str,
    replacement_marker: str,
) -> DispatchAssertion:
    """Confirm the patch actually took effect (§13).

    A dispatch assertion, not an inspection: the new kernel must appear in the
    re-profiled trace *and* the old one must be gone. Checking only for the new kernel
    would pass a workload that now runs both, which is a regression wearing a success's
    clothes.
    """
    normalized = [k.lower() for k in observed_kernels]
    new_present = any(replacement_marker.lower() in k for k in normalized)
    old_absent = not any(original_kernel.lower() in k for k in normalized)

    assertion = DispatchAssertion(
        new_kernel_present=new_present,
        old_kernel_absent=old_absent,
        observed_kernels=list(observed_kernels),
    )

    if assertion.took_effect:
        assertion.detail = (
            f"{replacement_marker!r} present and {original_kernel!r} absent: the override "
            f"is what executes"
        )
    elif new_present and not old_absent:
        assertion.detail = (
            f"both {replacement_marker!r} and {original_kernel!r} appear in the trace: the "
            f"override did not replace the original, it was added alongside it"
        )
    elif not new_present:
        assertion.detail = (
            f"{replacement_marker!r} never appears: the override did not take effect. "
            f"Common causes are the wrong dispatch key, the wrong overload, registration "
            f"after the first call, or a torch.compile graph captured before registration. "
            f"This would otherwise look like an honest 'no change' result."
        )
    else:
        assertion.detail = "inconclusive"

    return assertion


def revert_patch(record: AppliedPatch) -> bool:
    """Undo an applied patch. Returns whether anything needed undoing."""
    if not record.applied:
        return False
    if record.rung == "P1" and record.module_path:
        # P1 revert is genuinely nothing: the module is inert unless imported. Removing
        # it is housekeeping, not a rollback.
        path = Path(record.module_path)
        if path.is_file():
            path.unlink()
        record.applied = False
        return True
    raise PatchError(
        f"rung {record.rung} has no automatic revert; follow the recorded procedure: "
        f"{record.revert_procedure}"
    )


# ---------------------------------------------------------------------------
# Candidate isolation
# ---------------------------------------------------------------------------


def create_worktree(repo: Path, branch: str, destination: Path) -> Path:
    """Put a candidate in its own git worktree (§13).

    Every candidate is applied in its own worktree, never the working tree. That
    isolates experiments, makes rollback free, and keeps parallel candidate evaluation
    possible later without a redesign.
    """
    destination = Path(destination)
    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(destination)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise PatchError(f"could not create worktree at {destination}: {result.stderr.strip()}")
    return destination


def remove_worktree(repo: Path, destination: Path) -> bool:
    """Remove a candidate worktree, leaving the main tree untouched."""
    result = subprocess.run(
        ["git", "worktree", "remove", "--force", str(destination)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.returncode == 0
