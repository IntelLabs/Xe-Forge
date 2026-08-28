"""
Authoring a fused kernel for a region (plan §7.6, §13.8).

Region detection finds where fusion would pay — on a Qwen decode trace, `gemm+activation`
at 39.6% of GPU time and `gemm+rmsnorm` at 27.3%, together two thirds of the run. What it
did next was route them to Xe-Fuse and, when that was not installed, stop. That made an
external project a hard dependency for the only path that reaches the opaque GEMM, which
is both fragile and unnecessary.

AMD's Hyperloom does not route fusion anywhere. Its handler says what it does outright:
*"Run autonomous kernel fusion via forge-fusion (serving-validated). **Authors**
serving-safe fused kernels and returns a **source patch + env flags** for the integrate
gate."* The fused kernel is generated for the shapes at hand and then goes through the
same apply/verify/measure gate as anything else.

That is the right default here too, and every piece already exists:

* the region says which kernels to fuse and which intermediate disappears;
* §11.8's operator override compiles a single SYCL translation unit in 7-8 seconds and
  registers on the XPU dispatch key, so nothing in vLLM is forked;
* the differential harness checks the fused result against the unfused one;
* §17 measures it and the loop keeps or reverts.

Xe-Fuse remains **one executor among several** — its templates are autotuned and worth
using where they match — but it is no longer the only one, and its absence is a missing
option rather than a dead end.

What a fusion must preserve, and why it is stated rather than assumed: a fused kernel
computes the same values as the sequence it replaces, so the differential check is the
gate that matters. Fusion changes *when* values reach memory, not what they are — and a
"fusion" that changes results is a different model, not a faster one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FusionExecutor(StrEnum):
    """Who writes the fused kernel."""

    # An agent authors SYCL for these exact shapes, compiled as an operator override.
    # The default, because it needs nothing installed beyond a compiler.
    AUTHOR = "author"
    # Xe-Fuse's autotuned templates, when the region matches a preset and it is present.
    XE_FUSE = "xe_fuse"


@dataclass
class FusionTask:
    """One region, framed as a job an author can actually do."""

    region_id: str
    pattern: str
    gpu_share: float
    ops: list[str] = field(default_factory=list)
    intermediate_shape: list[int] = field(default_factory=list)
    intermediate_dtype: str = ""
    intermediate_bytes: int = 0
    executor: FusionExecutor = FusionExecutor.AUTHOR

    @property
    def eliminable_mb(self) -> float:
        return self.intermediate_bytes / (1024 * 1024)

    def describe(self) -> str:
        """Prompt context for whoever writes the kernel.

        Leads with the intermediate, because that is the entire point of the change. A
        fusion brief that opens with "these two kernels are adjacent" invites a rewrite
        of the epilogue; one that opens with "this tensor never needs to reach memory"
        states the actual objective.
        """
        lines = [
            f"REGION {self.region_id}: {self.pattern} — {self.gpu_share * 100:.1f}% of GPU time",
            "",
            "The sequence to replace:",
        ]
        lines.extend(f"  {i + 1}. {op}" for i, op in enumerate(self.ops))

        lines.append("")
        if self.intermediate_shape:
            size = f" ({self.eliminable_mb:.2f} MB per pass)" if self.intermediate_bytes else ""
            dtype = self.intermediate_dtype or "dtype unrecorded"
            lines.append(
                f"The intermediate that disappears: {self.intermediate_shape} {dtype}{size}. "
                f"It is written to memory by the producer and read straight back by the "
                f"consumer; a fused kernel keeps it in registers or SLM and never "
                f"materializes it."
            )
        else:
            lines.append(
                "The intermediate tensor was not recorded, so the traffic saved cannot be "
                "estimated — the fusion may still be correct, but its value is unmeasured "
                "until it is benchmarked."
            )

        lines += [
            "",
            "Requirements:",
            "  - Identical results. Fusion changes when values reach memory, not what",
            "    they are; a fused kernel that changes outputs is a different model.",
            "  - Register the fused implementation as an operator override on the XPU",
            "    dispatch key so nothing in the framework is forked (§11.8).",
            "  - Accumulate reductions in fp32 regardless of the tensor dtype.",
            "  - Use sycl::reduce_over_group, not a hand-rolled barrier tree.",
        ]
        return "\n".join(lines)


def task_from_region(region, kernels, executor: FusionExecutor | None = None) -> FusionTask:
    """Build a fusion task from a detected region and the catalog it refers to."""
    by_id = {k.id: k for k in kernels}
    ops = []
    for kernel_id in region.kernel_ids:
        record = by_id.get(kernel_id)
        name = (record.runtime_name if record else kernel_id).split("<")[0]
        ops.append(f"{kernel_id}: {name}")

    tensor = region.intermediate_tensors[0] if region.intermediate_tensors else None
    return FusionTask(
        region_id=region.id,
        pattern=region.fusion_pattern or "unnamed",
        gpu_share=region.gpu_time_share,
        ops=ops,
        intermediate_shape=list(tensor.shape) if tensor else [],
        intermediate_dtype=(tensor.dtype if tensor else "") or "",
        intermediate_bytes=(tensor.bytes if tensor else 0) or 0,
        executor=executor or default_executor(region),
    )


def default_executor(region) -> FusionExecutor:
    """Prefer Xe-Fuse where it is installed and matches; otherwise author.

    Authoring is the fallback that always exists, which is what makes the region path
    usable at all. Xe-Fuse's templates are autotuned and preferable when they apply, so
    they win when available — but their absence costs an option, not the path.
    """
    try:
        from xe_forge.orbit.analysis.xe_fuse import match_model_preset, xe_fuse_available

        if xe_fuse_available() and match_model_preset(region):
            return FusionExecutor.XE_FUSE
    except Exception:
        pass
    return FusionExecutor.AUTHOR


def describe_executor(executor: FusionExecutor) -> str:
    if executor is FusionExecutor.XE_FUSE:
        return "Xe-Fuse (installed, region matches a shipped preset)"
    return "author a fused SYCL kernel for these shapes, as an operator override"
