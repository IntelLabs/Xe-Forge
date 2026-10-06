"""Launch configuration chosen by the lowered kernel.

The kernel may pick its own launch geometry: anything is allowed as long as the same
inputs produce the same outputs as the Triton kernel. It says so in an optional
directive at the top of the ``.visaasm`` file::

    // @launch num_warps=8 threads_per_warp=16 slm_bytes=4096 grid=(cdiv(M, 4), 1, 1)

All fields are optional; omitted ones keep the Triton kernel's own launch. ``grid``
is an arithmetic expression over the kernel's scalar arguments and constexprs (and
``grid0``/``grid1``/``grid2``, the Triton launch's own grid), with ``cdiv``, ``min``
and ``max``. It is evaluated by a small whitelist evaluator, never by ``eval``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass

_DIRECTIVE = re.compile(r"^\s*//\s*@launch\b(.*)$", re.MULTILINE)
_FIELD = re.compile(r"(\w+)\s*=\s*([^\s(]+)")


@dataclass(frozen=True)
class LaunchConfig:
    num_warps: int
    threads_per_warp: int
    slm_bytes: int = 0
    grid: tuple[str, ...] | None = None  # expressions; None keeps the Triton grid

    @property
    def work_group_size(self) -> int:
        return self.num_warps * self.threads_per_warp

    def key(self) -> str:
        """Identifies the ABI: the grid does not change the payload, the rest does."""
        return hashlib.sha1(f"{self.num_warps}/{self.threads_per_warp}/{bool(self.slm_bytes)}".encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> LaunchConfig:
        d = dict(d)
        if d.get("grid") is not None:
            d["grid"] = tuple(d["grid"])
        return cls(**d)


class LaunchError(ValueError):
    """The directive or the grid expression is not acceptable."""


def parse_launch(visa_text: str, default: LaunchConfig) -> LaunchConfig:
    m = _DIRECTIVE.search(visa_text)
    if not m:
        return default
    text = m.group(1)
    grid_text = None
    g = re.search(r"\bgrid\s*=\s*\(", text)
    if g:
        depth, end = 0, None
        for pos in range(g.end() - 1, len(text)):
            depth += text[pos] == "("
            depth -= text[pos] == ")"
            if depth == 0:
                end = pos + 1
                break
        if end is None:
            raise LaunchError("unbalanced parentheses in grid")
        grid_text = text[g.end() - 1:end]
        text = text[:g.start()] + text[end:]
    fields = dict(_FIELD.findall(text))
    if grid_text is not None:
        fields["grid"] = grid_text
    unknown = set(fields) - {"num_warps", "threads_per_warp", "slm_bytes", "grid"}
    if unknown:
        raise LaunchError(f"unknown @launch field(s): {sorted(unknown)}")
    try:
        nw = int(fields.get("num_warps", default.num_warps))
        tpw = int(fields.get("threads_per_warp", default.threads_per_warp))
        slm = int(fields.get("slm_bytes", default.slm_bytes))
    except ValueError as e:
        raise LaunchError(f"@launch values must be integers: {e}") from None
    if nw < 1 or nw & (nw - 1) or nw > 32:
        raise LaunchError("num_warps must be a power of two between 1 and 32")
    if tpw not in (16, 32):
        raise LaunchError("threads_per_warp must be 16 or 32")
    if nw * tpw > 1024:
        raise LaunchError("a work-group has at most 1024 work-items")
    if not 0 <= slm <= 128 * 1024:
        raise LaunchError("slm_bytes must be between 0 and 131072")
    grid = default.grid
    if "grid" in fields:
        text = fields["grid"].strip()
        if not (text.startswith("(") and text.endswith(")")):
            raise LaunchError("grid must be a parenthesized tuple, e.g. grid=(cdiv(n, 256), 1, 1)")
        parts = _split_top(text[1:-1])
        if not 1 <= len(parts) <= 3:
            raise LaunchError("grid has one to three dimensions")
        for p in parts:
            try:
                _check(ast.parse(p, mode="eval").body)
            except SyntaxError as e:
                raise LaunchError(f"grid expression is not valid: {p!r} ({e.msg})") from None
        grid = tuple(parts)
    return LaunchConfig(nw, tpw, slm, grid)


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


_FUNCS = {"cdiv": lambda a, b: -(-a // b), "min": min, "max": max}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        _check(node.left)
        _check(node.right)
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        for a in node.args:
            _check(a)
        return
    raise LaunchError(f"grid expressions may use integers, names, + - * // %, cdiv/min/max only: {ast.unparse(node)}")


def eval_grid(exprs: tuple[str, ...], names: dict[str, int]) -> tuple[int, int, int]:
    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise LaunchError(f"unknown name in grid: {node.id}; available: {sorted(names)}")
            return int(names[node.id])
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        return _FUNCS[node.func.id](*[ev(a) for a in node.args])

    dims = [int(ev(ast.parse(e, mode="eval").body)) for e in exprs]
    if any(d < 1 for d in dims):
        raise LaunchError(f"grid evaluates to {dims}; every dimension must be at least 1")
    return tuple(dims + [1] * (3 - len(dims)))
