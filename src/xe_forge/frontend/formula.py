"""Per-call cost formulas: arithmetic over an op's dims, one parser for checking and evaluating.

Numbers, names and ``+ - * / // **`` with parentheses, ``min``/``max`` and a conditional
(``a if X < 0 else b``) -- a window that caps a read, a flag that changes one; no other calls,
attributes or indexing. Inputs may be negative (``-1`` for "none"); a cost may not.
"""

from __future__ import annotations

import ast
import math
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}  # fmt: skip
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}  # fmt: skip
_CALLS = {"min": min, "max": max}


def _parse(expr) -> ast.Expression | None:
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        ok = (
            isinstance(n, (ast.Expression, ast.Name, ast.Load))
            or (isinstance(n, ast.Constant) and type(n.value) in (int, float))
            or (isinstance(n, ast.BinOp) and type(n.op) in _OPS)
            or (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub))
            or type(n) in _OPS
            or isinstance(n, ast.USub)
            or isinstance(n, ast.IfExp)
            or (isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP)
            or type(n) in _CMP
            or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _CALLS and n.args and not n.keywords)
        )  # fmt: skip
        if not ok:
            return None
    return tree


def names(expr) -> set[str] | None:
    """The dims a formula uses, or None when it is not one."""
    tree = _parse(expr)
    if tree is None:
        return None
    calls = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in calls}


def evaluate(expr, dims: dict) -> float | None:
    """The formula's value at ``dims``; None when it cannot be evaluated."""
    tree = _parse(expr)
    if tree is None:
        return None

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        # Floats throughout: an int ** int (``10**10**10``, from agent-written costs) would
        # run unbounded; a float one overflows at once.
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            v = dims.get(n.id)
            if type(v) not in (int, float):
                raise ValueError(n.id)
            return float(v)
        if isinstance(n, ast.UnaryOp):
            return -walk(n.operand)
        if isinstance(n, ast.IfExp):
            return walk(n.body) if walk(n.test) else walk(n.orelse)
        if isinstance(n, ast.Compare):
            return _CMP[type(n.ops[0])](walk(n.left), walk(n.comparators[0]))
        if isinstance(n, ast.Call):
            return _CALLS[n.func.id](walk(a) for a in n.args)
        return _OPS[type(n.op)](walk(n.left), walk(n.right))

    try:
        value = walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    # A negative base to a fractional power is complex; inf/nan, or below zero, is no cost.
    ok = isinstance(value, float) and math.isfinite(value) and value >= 0
    return value if ok else None
