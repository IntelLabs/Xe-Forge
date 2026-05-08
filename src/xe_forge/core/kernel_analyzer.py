"""AST-based PyTorch kernel analysis.

Provides fast, deterministic analysis of PyTorch reference kernels to
identify operations, kernel type, fusion opportunities, and suggest
starting templates. Supplements (not replaces) the LLM-based AnalyzerAgent.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalysisResult:
    kernel_type: str = "unknown"
    operations: list[str] = field(default_factory=list)
    activations: list[str] = field(default_factory=list)
    reductions: list[str] = field(default_factory=list)
    elementwise: list[str] = field(default_factory=list)
    shapes: dict[str, int] = field(default_factory=dict)
    fusion_opportunities: list[str] = field(default_factory=list)
    memory_pattern: str = "block_pointers"
    has_gemm: bool = False
    suggested_template: str | None = None


class _ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operations: list[str] = []
        self.has_matmul = False
        self.has_linear = False
        self.activations: list[str] = []
        self.reductions: list[str] = []
        self.elementwise: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, "id") and node.func.value.id == "torch":
                op = node.func.attr
                self.operations.append(op)
                if op == "matmul":
                    self.has_matmul = True
                elif op in ("sum", "mean", "max", "min"):
                    self.reductions.append(op)
                elif op in ("sigmoid", "tanh", "relu", "gelu", "silu"):
                    self.activations.append(op)
                elif op == "clamp":
                    self.elementwise.append("clamp")
            elif hasattr(node.func.value, "attr") and node.func.value.attr == "functional":
                op = node.func.attr
                self.operations.append(f"F.{op}")
                if op in ("gelu", "relu", "silu", "softmax", "sigmoid"):
                    self.activations.append(op)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        op_map = {
            ast.Mult: "multiply",
            ast.Div: "divide",
            ast.Add: "add",
            ast.Sub: "subtract",
        }
        name = op_map.get(type(node.op))
        if name:
            self.elementwise.append(name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            if hasattr(node.value.func, "attr") and node.value.func.attr == "Linear":
                self.has_linear = True
        self.generic_visit(node)


class KernelAnalyzer:
    """Analyze a PyTorch kernel file via AST inspection."""

    def analyze(self, filepath: str | Path) -> AnalysisResult:
        filepath = Path(filepath)
        source = filepath.read_text()
        return self.analyze_source(source)

    def analyze_source(self, source: str) -> AnalysisResult:
        tree = ast.parse(source)
        visitor = _ASTVisitor()
        visitor.visit(tree)

        shapes: dict[str, int] = {}
        dim_names = (
            "batch_size",
            "in_features",
            "out_features",
            "hidden_size",
            "input_size",
        )
        for line in source.split("\n"):
            if "=" in line and any(d in line for d in dim_names):
                m = re.match(r"(\w+)\s*=\s*(\d+)", line.strip())
                if m:
                    shapes[m.group(1)] = int(m.group(2))

        has_gemm = visitor.has_matmul or visitor.has_linear

        if has_gemm:
            if visitor.activations or visitor.elementwise:
                kernel_type = "gemm_epilogue"
            elif visitor.reductions:
                kernel_type = "gemm_reduction"
            else:
                kernel_type = "gemm"
        elif visitor.reductions:
            kernel_type = "reduction"
        elif visitor.elementwise:
            kernel_type = "elementwise"
        else:
            kernel_type = "unknown"

        fusion: list[str] = []
        if has_gemm:
            if len(visitor.activations) <= 2 and len(visitor.elementwise) <= 3:
                fusion.append("Light epilogue fusion (GEMM + activation + elementwise)")
            else:
                fusion.append("Heavy epilogue — consider partial fusion or split")
            if visitor.reductions:
                fusion.append("WARNING: GEMM + reduction — use 2D GEMM then separate reduction")

        memory_pattern = "block_pointers"
        if len(visitor.reductions) > 1:
            memory_pattern = "tensor_descriptors"

        template_map = {
            "gemm": "templates/gemm_template.py",
            "gemm_epilogue": "templates/gemm_epilogue_template.py",
            "reduction": "templates/reduction_template.py",
            "gemm_reduction": "templates/reduction_template.py",
        }

        return AnalysisResult(
            kernel_type=kernel_type,
            operations=visitor.operations,
            activations=visitor.activations,
            reductions=visitor.reductions,
            elementwise=visitor.elementwise,
            shapes=shapes,
            fusion_opportunities=fusion,
            memory_pattern=memory_pattern,
            has_gemm=has_gemm,
            suggested_template=template_map.get(kernel_type),
        )


def format_analysis(result: AnalysisResult, name: str = "") -> str:
    """Format analysis results for human-readable output."""
    parts: list[str] = []
    header = f"Analysis: {name}" if name else "Analysis"
    parts.append(f"{'=' * 70}")
    parts.append(header)
    parts.append(f"{'=' * 70}")
    parts.append(f"Kernel Type: {result.kernel_type.upper()}")
    parts.append(f"Memory Pattern: {result.memory_pattern}")
    if result.shapes:
        parts.append("Shapes:")
        for k, v in result.shapes.items():
            parts.append(f"  {k}: {v}")
    parts.append(f"Operations: {len(result.operations)} total")
    if result.has_gemm:
        parts.append("  GEMM/Linear: yes")
    if result.activations:
        parts.append(f"  Activations: {', '.join(set(result.activations))}")
    if result.reductions:
        parts.append(f"  Reductions: {', '.join(set(result.reductions))}")
    if result.elementwise:
        parts.append(f"  Elementwise: {', '.join(set(result.elementwise))}")
    if result.fusion_opportunities:
        parts.append("Fusion Opportunities:")
        for opp in result.fusion_opportunities:
            parts.append(f"  -> {opp}")
    if result.suggested_template:
        parts.append(f"Suggested Template: {result.suggested_template}")
    return "\n".join(parts)
