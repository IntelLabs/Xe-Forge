"""Static kernel validation for multiple DSLs.

Consolidates Triton8's 12 static checks with Xe-Forge's inline validation
from optimizer_agent.py. Runs without any LLM — pure source analysis.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    check_name: str
    severity: str  # "error", "warning", "info"
    message: str
    line: int | None = None
    suggestion: str | None = None


class KernelValidator:
    """Multi-DSL static kernel validator."""

    def validate(
        self,
        code: str,
        dsl: str = "triton",
        stage: str | None = None,
    ) -> list[ValidationIssue]:
        """Validate kernel source code, returning a list of issues."""
        dsl = dsl.lower()
        if dsl == "triton":
            return self._validate_triton(code, stage)
        if dsl == "sycl":
            return self._validate_sycl(code)
        return self._validate_generic(code)

    # ------------------------------------------------------------------
    # Triton validation (12 Triton8 checks + Xe-Forge inline checks)
    # ------------------------------------------------------------------

    def _validate_triton(self, code: str, stage: str | None = None) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        lines = code.split("\n")

        # -- Xe-Forge structural checks (from optimizer_agent inline) --

        # Syntax parse
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    "syntax_error",
                    "error",
                    f"Python syntax error: {e}",
                    line=e.lineno,
                )
            )
            return issues

        if "import triton" not in code:
            issues.append(
                ValidationIssue(
                    "missing_import",
                    "error",
                    "Missing 'import triton'. Triton kernel must import triton.",
                )
            )

        if "@triton.jit" not in code:
            issues.append(
                ValidationIssue(
                    "missing_jit",
                    "error",
                    "No @triton.jit decorated function found.",
                )
            )

        if "class Model" not in code:
            issues.append(
                ValidationIssue(
                    "missing_model_class",
                    "error",
                    "No 'class Model' found. Kernel must define a Model class.",
                )
            )

        # num_warps power-of-2
        for i, line in enumerate(lines):
            m = re.search(r"num_warps\s*[=:]\s*(\d+)", line)
            if m:
                val = int(m.group(1))
                if val & (val - 1) != 0:
                    issues.append(
                        ValidationIssue(
                            "num_warps_pow2",
                            "error",
                            f"num_warps={val} is not a power of 2.",
                            line=i + 1,
                        )
                    )

        # Block size power-of-2
        for i, line in enumerate(lines):
            for m in re.finditer(r"BLOCK_(?:SIZE_)?[A-Z]\s*[=:]\s*(\d+)", line):
                val = int(m.group(1))
                if val & (val - 1) != 0:
                    issues.append(
                        ValidationIssue(
                            "block_size_pow2",
                            "error",
                            f"Block size {val} is not a power of 2.",
                            line=i + 1,
                        )
                    )

        # grf_mode in triton.Config
        for i, line in enumerate(lines):
            if "triton.Config" in line and "grf_mode" in line:
                if '"large"' in line or "'large'" in line:
                    issues.append(
                        ValidationIssue(
                            "grf_mode_value",
                            "warning",
                            "grf_mode should be '256' (string), not 'large'.",
                            line=i + 1,
                            suggestion="Use grf_mode='256'",
                        )
                    )

        # .cpu() return check
        for i, line in enumerate(lines):
            if ".cpu()" in line and "return" in line:
                issues.append(
                    ValidationIssue(
                        "cpu_return",
                        "warning",
                        "Returning .cpu() tensor — this forces device-to-host sync.",
                        line=i + 1,
                    )
                )

        # Fusion stage: vendor library replacement check
        if stage == "fusion":
            vendor_patterns = [
                "torch.nn.functional.linear",
                "F.linear",
                "torch.matmul",
                "torch.bmm",
            ]
            for i, line in enumerate(lines):
                for pat in vendor_patterns:
                    if pat in line and "@triton.jit" in code:
                        issues.append(
                            ValidationIssue(
                                "vendor_in_fusion",
                                "warning",
                                f"'{pat}' still present after fusion stage. "
                                "Expected to be replaced by fused Triton kernel.",
                                line=i + 1,
                            )
                        )

        # -- Triton8 checks --

        # 1. Autotune parameter defaults
        autotune_params: set[str] = set()
        in_autotune = False
        for _i, line in enumerate(lines):
            if "@triton.autotune" in line:
                in_autotune = True
            if in_autotune and "Config" in line:
                autotune_params.update(re.findall(r"'(\w+)':", line))
            if in_autotune and "@triton.jit" in line:
                in_autotune = False

        in_kernel_sig = False
        for i, line in enumerate(lines):
            if "@triton.jit" in line:
                in_kernel_sig = True
            if in_kernel_sig:
                for param in autotune_params:
                    if f"{param}:" in line and "=" in line:
                        issues.append(
                            ValidationIssue(
                                "autotune_default",
                                "error",
                                f"Autotune parameter '{param}' has a default value. "
                                "This causes 'Conflicting meta-parameters'. Remove the default.",
                                line=i + 1,
                            )
                        )
                if ")" in line:
                    break

        # 2. Grid dimensionality with swizzling
        has_swizzling = "GROUP_SIZE_M" in code or "swizzle" in code.lower()
        for i, line in enumerate(lines):
            if "grid" in line and "=" in line:
                if line.count("triton.cdiv") >= 2 and "(" in line and ")" in line:
                    paren_depth = 0
                    comma_count = 0
                    start = line.index("(")
                    for ch in line[start:]:
                        if ch == "(":
                            paren_depth += 1
                        elif ch == ")":
                            paren_depth -= 1
                        elif ch == "," and paren_depth == 1:
                            comma_count += 1
                    if comma_count >= 1 and has_swizzling:
                        issues.append(
                            ValidationIssue(
                                "grid_swizzle_conflict",
                                "error",
                                "Grid is 2D but tile swizzling (GROUP_SIZE_M) is used. "
                                "Grid must be 1D with swizzling.",
                                line=i + 1,
                            )
                        )

        # 3. boundary_check format
        for i, line in enumerate(lines):
            if "boundary_check" in line:
                if "True" in line or "False" in line:
                    issues.append(
                        ValidationIssue(
                            "boundary_check_bool",
                            "error",
                            "boundary_check uses booleans. Use dimension indices (0, 1).",
                            line=i + 1,
                        )
                    )
                if ".load(" in line:
                    for j in range(max(0, i - 20), i):
                        if "make_tensor_descriptor" in lines[j]:
                            issues.append(
                                ValidationIssue(
                                    "descriptor_boundary_check",
                                    "error",
                                    "Tensor descriptor .load() does NOT accept boundary_check. "
                                    "Descriptors handle boundaries internally.",
                                    line=i + 1,
                                )
                            )
                            break

        # 4. float64 usage
        for i, line in enumerate(lines):
            if "float64" in line.lower() or "tl.float64" in line:
                issues.append(
                    ValidationIssue(
                        "float64_usage",
                        "warning",
                        "float64 detected — 5-10x slower on XPU. Use float32 unless required.",
                        line=i + 1,
                    )
                )

        # 5. int32 overflow in batch offsets
        batch_pattern = re.compile(r"(program_id|pid|bid)\s*\*\s*stride")
        for i, line in enumerate(lines):
            if batch_pattern.search(line):
                if ".to(tl.int64)" not in line:
                    issues.append(
                        ValidationIssue(
                            "int32_overflow",
                            "warning",
                            "Batch offset may overflow int32. Cast to int64.",
                            line=i + 1,
                            suggestion="offset = bid.to(tl.int64) * stride",
                        )
                    )

        # 6. num_warps=32 without autotune
        for i, line in enumerate(lines):
            if "num_warps=32" in line or "num_warps = 32" in line:
                if "@triton.autotune" not in code or code.count("num_warps=32") == 1:
                    issues.append(
                        ValidationIssue(
                            "num_warps_no_autotune",
                            "warning",
                            "num_warps=32 without autotuning. Sweep {4,8,16,32}.",
                            line=i + 1,
                        )
                    )

        # 7. Mixed block pointer and tensor descriptor APIs
        has_block_ptr = "make_block_ptr" in code
        has_tensor_desc = "make_tensor_descriptor" in code
        if has_block_ptr and has_tensor_desc:
            issues.append(
                ValidationIssue(
                    "mixed_memory_apis",
                    "info",
                    "Both block pointers and tensor descriptors found. "
                    "Don't mix APIs for the same load/store operation.",
                )
            )

        # 8. Device-to-host sync in hot path
        for i, line in enumerate(lines):
            if ".item()" in line or "float(tensor" in line or "int(tensor" in line:
                if "def forward" in "\n".join(lines[max(0, i - 20) : i]):
                    issues.append(
                        ValidationIssue(
                            "device_host_sync",
                            "error",
                            "Device-to-host sync (.item()) in forward pass kills performance.",
                            line=i + 1,
                        )
                    )

        # 9. Weight transpose in forward() hot path
        for i, line in enumerate(lines):
            if ".t()" in line and ".contiguous()" in line:
                if "def forward" in "\n".join(lines[max(0, i - 20) : i]):
                    issues.append(
                        ValidationIssue(
                            "weight_transpose_forward",
                            "warning",
                            "Weight .t().contiguous() in forward(). Pre-pack once and cache.",
                            line=i + 1,
                        )
                    )

        # 10. GEMM N-loop serialization
        for i, line in enumerate(lines):
            if re.search(r"for.*range\(.*,\s*N\s*[,)]", line):
                context = code[max(0, code.rfind("\n", 0, i) - 500) : code.find("\n", i) + 500]
                if "tl.dot" in context:
                    issues.append(
                        ValidationIssue(
                            "gemm_n_serialization",
                            "error",
                            "GEMM loops over N tiles inside one program. Use 2D grid.",
                            line=i + 1,
                        )
                    )

        # 11. tl.exp usage
        for i, line in enumerate(lines):
            if "tl.exp(" in line and "tl.math.exp2" not in code:
                issues.append(
                    ValidationIssue(
                        "tl_exp",
                        "info",
                        "tl.exp() found. Consider exp2-based: exp(x) = exp2(x * 1.44269504).",
                        line=i + 1,
                    )
                )

        # 12. get_inputs / get_init_inputs
        if "class Model" in code:
            if "def get_inputs" not in code:
                issues.append(
                    ValidationIssue(
                        "missing_get_inputs",
                        "warning",
                        "Model class found but no get_inputs(). Required by ai-bench.",
                    )
                )
            if "def get_init_inputs" not in code:
                issues.append(
                    ValidationIssue(
                        "missing_get_init_inputs",
                        "warning",
                        "Model class found but no get_init_inputs(). Required by ai-bench.",
                    )
                )

        return issues

    # ------------------------------------------------------------------
    # SYCL validation
    # ------------------------------------------------------------------

    def _validate_sycl(self, code: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if "#include" not in code:
            issues.append(
                ValidationIssue(
                    "missing_include",
                    "warning",
                    "No #include directives found in SYCL source.",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Generic validation
    # ------------------------------------------------------------------

    def _validate_generic(self, code: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    "syntax_error",
                    "error",
                    f"Syntax error: {e}",
                    line=e.lineno,
                )
            )

        return issues


def format_issues(issues: list[ValidationIssue]) -> str:
    """Format validation issues for human-readable output."""
    if not issues:
        return "No issues found."

    severity_icon = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}
    parts: list[str] = []
    for issue in issues:
        loc = f" (line {issue.line})" if issue.line else ""
        label = severity_icon.get(issue.severity, issue.severity.upper())
        parts.append(f"  [{label}] {issue.check_name}: {issue.message}{loc}")
        if issue.suggestion:
            parts.append(f"          Suggestion: {issue.suggestion}")

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    header = (
        f"Validation: {len(errors)} error(s), {len(warnings)} warning(s), "
        f"{len(issues) - len(errors) - len(warnings)} info"
    )
    return header + "\n" + "\n".join(parts)
