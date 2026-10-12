"""Captured op -> what the kernel-locator is told to look for.

Which repository carries an op, which kernel or library call implements it, in which
language, and which of its arguments is the activation or the weight is the naming agent's
answer (``frontend-name.md.j2`` -> ``names.json``): it reads the framework and the
repositories instead of matching names against tables kept here. Each op carries its entry
as ``framework_meta["named"]``; this module only turns it into a ``Naming``.

The one rule kept in code is PyTorch's own: an ``aten::`` op is implemented inside PyTorch.

Routes: ``kernel`` (a named kernel in a repository Xe-Forge can edit), ``library`` (a
library primitive -- change the call, not the kernel), ``aten`` (inside PyTorch, no kernel
repository) and ``unnamed`` (not established).
"""

from __future__ import annotations

from xe_forge.frontend.ir import CapturedOp, Naming

_IMPL = {"kernel": "provider", "library": "onednn", "aten": "aten"}


def key(op: CapturedOp) -> str:
    """What names.json is keyed by: the implementing op, which a patched build keeps."""
    return op.framework_op


def _base(op: str) -> str:
    return op.split("::", 1)[-1].split(".", 1)[0]


def name_op(op: CapturedOp) -> Naming:
    entry = op.framework_meta.get("named")
    if entry:
        op.impl_class = _IMPL.get(entry.get("route"), "unknown")
        return Naming(
            entry.get("route", "unnamed"),
            entry.get("name") or _base(op.framework_op),
            entry.get("repo"),
            entry.get("dsl"),
            via=entry.get("via", "named by the naming agent"),
        )
    if op.framework_op.startswith("aten::"):
        op.impl_class = "aten"
        return Naming("aten", _base(op.framework_op), via="ATen op inside PyTorch")
    op.impl_class = "unknown"
    return Naming("unnamed", _base(op.framework_op), via="not named (no names.json entry)")
