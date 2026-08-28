"""Patch-back: the P1-P5 mechanism ladder, its dispatch assertion, and the
spec-driven harness for dispatcher-registered SYCL ops."""

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
