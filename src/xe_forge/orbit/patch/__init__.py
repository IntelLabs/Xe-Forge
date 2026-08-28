"""Patch-back: the P1-P5 mechanism ladder and its dispatch assertion (plan §13),
plus the spec-driven harness that lets a dispatcher-registered SYCL op be driven
through the same Model + YAML contract as any Python kernel (plan §9.7)."""

from xe_forge.orbit.patch.ladder import (
    RUNG_ORDER,
    AppliedPatch,
    DispatchAssertion,
    PatchError,
    apply_patch,
    assert_dispatch,
    available_rungs,
    choose_rung,
    create_worktree,
    remove_worktree,
    render_operator_override,
    revert_patch,
)
from xe_forge.orbit.patch.sycl_harness import (
    emit_dispatcher_candidate,
    render_dispatcher_model,
)

__all__ = [
    "RUNG_ORDER",
    "AppliedPatch",
    "DispatchAssertion",
    "PatchError",
    "apply_patch",
    "assert_dispatch",
    "available_rungs",
    "choose_rung",
    "create_worktree",
    "emit_dispatcher_candidate",
    "remove_worktree",
    "render_dispatcher_model",
    "render_operator_override",
    "revert_patch",
]
