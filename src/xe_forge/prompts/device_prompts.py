"""
Device and DSL-aware prompt components for LLM-driven optimization.

The PromptLibrary maps (dsl, device_type) tuples to prompt text components
used by the analyzer, optimizer, and planner agents.
"""

from __future__ import annotations

_DEVICE_DESCRIPTIONS: dict[str, str] = {
    "xpu": "Intel XPU (Data Center GPU Max / Ponte Vecchio)",
    "cuda": "NVIDIA CUDA GPU",
    "cpu": "CPU",
}

_DSL_NAMES: dict[str, str] = {
    "triton": "Triton",
    "gluon": "Gluon",
    "sycl": "SYCL/XeTLA",
    "cuda": "CUDA C++",
}

_DEVICE_TUNING_DEFAULTS: dict[str, dict[str, int | str]] = {
    "xpu": {
        "BLOCK_M": 256,
        "BLOCK_N": 256,
        "BLOCK_K": 32,
        "num_warps": 32,
        "grf_mode": "large (256 registers)",
    },
    "cuda": {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "num_warps": 4,
    },
    "cpu": {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 1,
    },
}


class PromptLibrary:
    """Provides device/DSL-aware prompt components."""

    def __init__(self, dsl: str = "triton", device_type: str = "xpu"):
        self.dsl = dsl
        self.device_type = device_type

    def device_description(self) -> str:
        return _DEVICE_DESCRIPTIONS.get(self.device_type, self.device_type.upper())

    def dsl_name(self) -> str:
        return _DSL_NAMES.get(self.dsl, self.dsl.upper())

    def system_preamble(self) -> str:
        return (
            f"You are an expert {self.dsl_name()} kernel optimizer for {self.device_description()}."
        )

    def target_device_line(self) -> str:
        return f"TARGET DEVICE: {self.device_description()}"

    def device_specific_stage_label(self) -> str:
        labels = {
            "xpu": "DEVICE SPECIFIC (Intel XPU)",
            "cuda": "DEVICE SPECIFIC (NVIDIA CUDA)",
            "cpu": "DEVICE SPECIFIC (CPU)",
        }
        return labels.get(self.device_type, "DEVICE SPECIFIC")

    def tuning_defaults(self) -> dict[str, int | str]:
        return _DEVICE_TUNING_DEFAULTS.get(
            self.device_type,
            _DEVICE_TUNING_DEFAULTS["xpu"],
        )

    def stage_guidance(self, stage: str) -> str:
        """Return device-specific guidance for a given optimization stage."""
        defaults = self.tuning_defaults()

        if stage == "device_specific":
            if self.device_type == "xpu":
                return (
                    f"Intel XPU tuning: BLOCK_M={defaults['BLOCK_M']}, "
                    f"BLOCK_N={defaults['BLOCK_N']}, BLOCK_K={defaults['BLOCK_K']}, "
                    f"num_warps={defaults['num_warps']}, "
                    f"grf_mode={defaults.get('grf_mode', 'large')}. "
                    "Use tl.extra.intel.libdevice for exp2/sigmoid. "
                    "Avoid .cpu() returns."
                )
            if self.device_type == "cuda":
                return (
                    f"NVIDIA CUDA tuning: BLOCK_M={defaults['BLOCK_M']}, "
                    f"BLOCK_N={defaults['BLOCK_N']}, BLOCK_K={defaults['BLOCK_K']}, "
                    f"num_warps={defaults['num_warps']}. "
                    "Use shared memory for reductions. "
                    "Consider warp-level primitives."
                )
            return "Apply device-appropriate tuning parameters."

        if stage == "block_pointers":
            if self.dsl == "triton":
                return (
                    "Convert manual pointer arithmetic to tl.make_block_ptr(). "
                    "Ensure boundary_check is correct."
                )
            return ""

        if stage == "autotuning":
            if self.dsl == "triton":
                return (
                    "Add @triton.autotune with configs covering tile sizes "
                    "and warp counts. Include key= for dynamic shapes."
                )
            return "Add autotuning configurations."

        return ""

    def code_requirements(self) -> list[str]:
        """Return DSL-specific code validation rules."""
        if self.dsl == "triton":
            return [
                "Must contain @triton.jit decorated function",
                "num_warps must be a power of 2",
                "grf_mode must NOT appear inside triton.Config (only as kernel_arg)"
                if self.device_type == "xpu"
                else "",
            ]
        if self.dsl == "gluon":
            return [
                "Must use Gluon API for kernel definition",
            ]
        if self.dsl == "sycl":
            return [
                "Must be valid SYCL C++ code",
            ]
        if self.dsl == "cuda":
            return [
                "Must be valid CUDA C++ code",
            ]
        return []

    def optimizer_signature_doc(self) -> str:
        """Return the docstring for the OptimizationSignature."""
        defaults = self.tuning_defaults()
        dsl_name = self.dsl_name()
        device_desc = self.device_description()

        lines = [
            f"You are an expert {dsl_name} kernel optimizer for {device_desc}.",
            "",
            "Given the current kernel code, a specific optimization stage, and detected issues,",
            "produce an OPTIMIZED version of the kernel that fixes the issues.",
        ]

        if self.device_type == "xpu" and self.dsl == "triton":
            lines.extend(
                [
                    "",
                    f"DEVICE_SPECIFIC: BLOCK_M={defaults['BLOCK_M']}, BLOCK_N={defaults['BLOCK_N']}, "
                    f"BLOCK_K={defaults['BLOCK_K']}, num_warps={defaults['num_warps']},",
                    f"grf_mode={defaults.get('grf_mode', 'large')}, "
                    "tl.extra.intel.libdevice for exp2/sigmoid.",
                ]
            )
        elif self.device_type == "cuda" and self.dsl == "triton":
            lines.extend(
                [
                    "",
                    f"DEVICE_SPECIFIC: BLOCK_M={defaults['BLOCK_M']}, BLOCK_N={defaults['BLOCK_N']}, "
                    f"BLOCK_K={defaults['BLOCK_K']}, num_warps={defaults['num_warps']}.",
                ]
            )

        return "\n".join(lines)

    def analyzer_signature_doc(self) -> str:
        """Return the docstring for the AnalysisSignature."""
        dsl_name = self.dsl_name()
        device_desc = self.device_description()
        return (
            f"You are a {dsl_name} kernel analysis expert for {device_desc}.\n"
            f"Analyze the given {dsl_name} kernel code and identify optimization opportunities.\n"
            f"Categorize each issue by type and severity (1=minor to 5=critical)."
        )

    def planner_signature_doc(self) -> str:
        """Return the docstring for the PlanningSignature."""
        return (
            f"You are an expert in GPU kernel optimization for {self.device_description()}.\n"
            "Determine the optimal order to apply optimization stages."
        )
