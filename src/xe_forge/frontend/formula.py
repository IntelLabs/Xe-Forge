"""Per-call cost formulas: arithmetic over an op's dims, one parser for checking and evaluating.

Numbers, names and ``+ - * / // **`` with parentheses; no calls, attributes or indexing.
"""

from __future__ import annotations

import ast
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}  # fmt: skip


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
        )
        if not ok:
            return None
    return tree


def names(expr) -> set[str] | None:
    """The dims a formula uses, or None when it is not one."""
    tree = _parse(expr)
    return None if tree is None else {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def evaluate(expr, dims: dict) -> float | None:
    """The formula's value at ``dims``; None when it cannot be evaluated."""
    tree = _parse(expr)
    if tree is None:
        return None

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            v = dims.get(n.id)
            if type(v) not in (int, float) or v < 0:
                raise ValueError(n.id)
            return v
        if isinstance(n, ast.UnaryOp):
            return -walk(n.operand)
        return _OPS[type(n.op)](walk(n.left), walk(n.right))

    try:
        return float(walk(tree))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
