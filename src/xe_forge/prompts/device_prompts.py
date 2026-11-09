"""
Device and DSL-aware prompt components for LLM-driven optimization.

The PromptLibrary maps (dsl, device_type) tuples to prompt text components
used by the analyzer, optimizer, and planner agents.
"""

from __future__ import annotations

_DEVICE_DESCRIPTIONS: dict[str, str] = {
    "xpu": "Intel XPU",
    "cuda": "NVIDIA CUDA GPU",
    "cpu": "CPU",
}

_DSL_NAMES: dict[str, str] = {
    "triton": "Triton",
    "gluon": "Gluon",
    "sycl": "SYCL/Sycl-tla",
    "cuda": "CUDA C++",
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

    def stage_guidance(self, stage: str) -> str:
        """Return device-specific guidance for a given optimization stage."""
        if stage == "device_specific":
            if self.device_type == "xpu":
                return (
                    "Intel XPU tuning: choose tile sizes, num_warps and grf_mode by "
                    "autotuning on this device. Use tl.extra.intel.libdevice for "
                    "exp2/sigmoid. Avoid .cpu() returns."
                )
            if self.device_type == "cuda":
                return (
                    "NVIDIA CUDA tuning: choose tile sizes and num_warps by autotuning on "
                    "this device. Use shared memory for reductions. "
                    "Consider warp-level primitives."
                )
            return "Apply device-appropriate tuning parameters."

        if stage == "tensor_descriptors":
            if self.dsl == "triton":
                return (
                    "Convert manual pointer arithmetic and any tl.make_block_ptr()/tl.advance() "
                    "(deprecated) to tl.make_tensor_descriptor() with desc.load([...]) / "
                    "desc.store([...]). The last stride must be 1."
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
                    "DEVICE_SPECIFIC: tile sizes, num_warps and grf_mode autotuned on this "
                    "device; tl.extra.intel.libdevice for exp2/sigmoid.",
                ]
            )
        elif self.device_type == "cuda" and self.dsl == "triton":
            lines.extend(
                [
                    "",
                    "DEVICE_SPECIFIC: tile sizes and num_warps autotuned on this device.",
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

    def render_for_signature(self, sig_name: str, **extra_context) -> str:
        """Render the Jinja2 template for a given Signature class name.

        Returns the rendered instruction text to be injected via append_instructions().
        Falls back to device_context_addendum() if the template is not found.
        """
        from xe_forge.prompts import render_signature_instructions

        context = {
            "dsl": self.dsl,
            "dsl_name": self.dsl_name(),
            "device_type": self.device_type,
            "device_description": self.device_description(),
            **extra_context,
        }
        try:
            return render_signature_instructions(sig_name, **context)
        except Exception:
            return self.device_context_addendum()

    def device_context_addendum(self) -> str:
        """Return a short addendum to append to any Signature for the current device/DSL.

        Returns an empty string for the default XPU/Triton target (already hardcoded in
        Signature docstrings). For other device targets, returns a correction block so the
        LLM uses the right hardware constraints.
        """
        if self.device_type == "xpu":
            return ""

        lines = [
            f"=== TARGET DEVICE OVERRIDE: {self.device_description()} ===",
            f"DSL: {self.dsl_name()}",
            "Ignore any XPU-specific guidance above; tune for this device.",
        ]

        if self.device_type == "cuda" and self.dsl == "triton":
            lines.append("Use shared memory for reductions. Consider warp-level primitives.")
            lines.append(
                "Do NOT use tl.extra.intel.libdevice — use standard math intrinsics instead."
            )
        elif self.dsl == "sycl":
            lines.append("Use SYCL/CUTLASS C++ with icpx -fsycl compilation.")

        return "\n".join(lines)
