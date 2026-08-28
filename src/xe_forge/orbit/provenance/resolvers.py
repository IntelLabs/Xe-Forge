"""
Kernel provenance: name -> provider -> source -> available actions. An ambiguous name
resolves to low confidence, never a guess; an unknown kernel gets PROFILE_MORE, never
an optimization action. Design rationale: docs/DESIGN.md.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from xe_forge.orbit.models import (
    ActionType,
    ExtractionLevel,
    KernelLanguage,
    Provider,
    ResolutionMethod,
    SourceLocation,
)


class ProvenanceResult(BaseModel):
    """What a resolver could establish about one kernel."""

    provider: Provider = Provider.UNKNOWN
    language: KernelLanguage | None = None
    framework_op: str | None = None
    source: SourceLocation = Field(default_factory=SourceLocation)
    actions: list[ActionType] = Field(default_factory=list)
    default_extraction: ExtractionLevel | None = None
    build_system: str | None = None
    aot: bool | None = None
    confidence: float = 0.0
    dispatch_chain: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# A resolver inspects a runtime kernel name and returns a result, or None to pass.
Resolver = Callable[[str], ProvenanceResult | None]


# --- patterns -------------------------------------------------------------

# Inductor names its generated kernels `triton_poi_fused_...`, `triton_red_...`,
# `triton_per_...` by convention.
_INDUCTOR_RE = re.compile(r"^triton_(poi|red|per|mm|tem)_", re.I)
# Hand-written Triton kernels typically surface under their Python function name.
_TRITON_RE = re.compile(r"(^|_)(triton|jit)_|_kernel$", re.I)
# oneDNN primitive names embed the implementation and propagation kind.
_ONEDNN_RE = re.compile(r"(onednn|dnnl|jit_(gemm|conv|uni)|brgemm|gemm_kernel)", re.I)
_ONEMKL_RE = re.compile(r"(onemkl|mkl_|cblas_|gemm_batch)", re.I)
# SYCL kernel names encode the functor/lambda type, usually via typeinfo mangling.
_SYCL_RE = re.compile(r"(_ZTS|sycl|typeinfo name for|__sycl_kernel|SYCL_kernel)", re.I)
_SYCL_TLA_RE = re.compile(r"(cutlass|sycl_tla|xe_mma|CollectiveMma|gemm_universal)", re.I)
_IPEX_RE = re.compile(r"(torch_ipex|ipex::|at::AtenIpex)", re.I)
_XPU_OPS_RE = re.compile(r"(at::native::xpu|torch_xpu_ops|xpu::)", re.I)
_VLLM_RE = re.compile(r"(vllm|_C::|paged_attention|unified_attn|fused_moe)", re.I)

# A templated SYCL name that resolves to several instantiations must not be pinned to
# one; this catches the "multiple template arguments present" shape.
_TEMPLATE_ARGS_RE = re.compile(r"<[^<>]*>")
# Runtime memory operations. Level Zero and the torch profiler report these in the same
# stream as kernels — `Memcpy D2H (DEVICE -> HOST)`, `Memset`, `zeMemoryCopy` — and they
# are not kernels. They must be attributed as what they are, not left to the unknown
# fallback, which would advise profiling further for a source file that cannot exist.
_TRANSFER_RE = re.compile(
    r"(^|\b)(memcpy|memset|mem_?fill|zeCommandListAppendMemory|"
    r"urEnqueueUSM|copy_(h2d|d2h|d2d))",
    re.I,
)


def _actions_for_editable_python() -> list[ActionType]:
    return [
        ActionType.KERNEL_REWRITE,
        ActionType.KERNEL_AUTOTUNE,
        ActionType.REGION_FUSION,
        ActionType.LAYOUT_CHANGE,
    ]


def _actions_for_editable_native() -> list[ActionType]:
    return [
        ActionType.KERNEL_REWRITE,
        ActionType.COMPILER_OPTION,
        ActionType.KERNEL_TILE_SEARCH,
        ActionType.REGION_FUSION,
        ActionType.LAYOUT_CHANGE,
    ]


def _actions_for_opaque() -> list[ActionType]:
    """No editable source, but far from unactionable."""
    return [
        ActionType.REGION_FUSION,
        ActionType.BACKEND_CHANGE,
        ActionType.LAYOUT_CHANGE,
        ActionType.LIBRARY_CONFIG,
    ]


# --- individual resolvers -------------------------------------------------


def resolve_inductor(name: str) -> ProvenanceResult | None:
    if not _INDUCTOR_RE.search(name):
        return None
    return ProvenanceResult(
        provider=Provider.INDUCTOR,
        language=KernelLanguage.TRITON,
        actions=_actions_for_editable_python(),
        default_extraction=ExtractionLevel.E2,
        build_system="jit",
        confidence=0.94,
        dispatch_chain=["torch.compile", "inductor", name],
        notes=[
            "Inductor-generated: the kernel body lives in the Inductor cache module "
            "and the launch wrapper in output_code.py; both are needed, and the "
            "bundle must pin the torch version."
        ],
    )


def resolve_vllm(name: str) -> ProvenanceResult | None:
    if not _VLLM_RE.search(name):
        return None
    is_native = bool(_SYCL_RE.search(name) or "_C::" in name)
    return ProvenanceResult(
        provider=Provider.CUSTOM,
        language=KernelLanguage.SYCL if is_native else KernelLanguage.TRITON,
        framework_op=_first_group(_VLLM_RE, name),
        actions=_actions_for_editable_native() if is_native else _actions_for_editable_python(),
        # vLLM attention and MoE reach through deep dispatch, so the honest default is
        # an in-situ harness; E2 is earned by proving the closure, not assumed.
        default_extraction=ExtractionLevel.E3,
        build_system="setuptools" if is_native else "jit",
        confidence=0.88,
        dispatch_chain=["vllm", "platform_dispatch", name],
        notes=["vLLM kernel: launch wrapper may build block tables or KV-cache metadata"],
    )


def resolve_sycl(name: str) -> ProvenanceResult | None:
    if not (_SYCL_RE.search(name) or _XPU_OPS_RE.search(name) or _IPEX_RE.search(name)):
        return None

    is_tla = bool(_SYCL_TLA_RE.search(name))
    provider = Provider.IPEX if _IPEX_RE.search(name) else Provider.SYCL

    # Confidence is graded, not binary. A heavily templated name that could
    # match several instantiations is reported as ambiguous rather than pinned.
    template_args = _TEMPLATE_ARGS_RE.findall(name)
    ambiguous = len(template_args) > 1
    confidence = 0.55 if ambiguous else 0.85

    notes = []
    candidates: list[str] = []
    if ambiguous:
        candidates = template_args
        notes.append(
            "templated name resolves to multiple instantiations; the concrete template "
            "arguments the workload used must be recovered before extraction"
        )

    return ProvenanceResult(
        provider=provider,
        language=KernelLanguage.SYCL_TLA if is_tla else KernelLanguage.SYCL,
        actions=_actions_for_editable_native(),
        default_extraction=ExtractionLevel.E3,
        build_system="cmake",
        aot=None,
        confidence=confidence,
        source=SourceLocation(confidence=confidence, candidates=candidates),
        dispatch_chain=["aten", "xpu_key", name],
        notes=notes
        or ["SYCL op: closure comes from compile_commands.json, not an AST walk"],
    )


def resolve_onednn(name: str) -> ProvenanceResult | None:
    if not (_ONEDNN_RE.search(name) or _ONEMKL_RE.search(name)):
        return None
    provider = Provider.ONEMKL if _ONEMKL_RE.search(name) else Provider.ONEDNN
    return ProvenanceResult(
        provider=provider,
        language=KernelLanguage.OPAQUE,
        actions=_actions_for_opaque(),
        default_extraction=ExtractionLevel.E4,
        build_system="prebuilt",
        confidence=0.91,
        dispatch_chain=["aten", "library_dispatch", name],
        notes=[
            "Opaque library primitive: no source extraction. Capture the verbose "
            "problem string as the isolated reproducer. Still actionable via "
            "fusion, backend, layout and library config."
        ],
    )


def resolve_triton(name: str) -> ProvenanceResult | None:
    if not _TRITON_RE.search(name):
        return None
    return ProvenanceResult(
        provider=Provider.TRITON,
        language=KernelLanguage.TRITON,
        actions=_actions_for_editable_python(),
        default_extraction=ExtractionLevel.E2,
        build_system="jit",
        confidence=0.80,
        dispatch_chain=["triton", name],
        notes=["hand-written Triton: closure is an AST walk over @triton.jit helpers"],
    )


def resolve_runtime_transfer(name: str) -> ProvenanceResult | None:
    """Attribute host/device copies and fills, which are not kernels.

    Left to the unknown fallback these read as "no provenance; needs more profiling",
    which is wrong in a way that wastes the reader's time: no amount of further
    profiling will produce a source file for a `Memcpy D2H`, because it does not have
    one. They are still worth reporting — transfer time on the critical path is a real
    finding — but the action space is host-side (pin memory, batch or elide the copy,
    overlap it with compute), never a kernel rewrite.
    """
    if not _TRANSFER_RE.search(name):
        return None
    return ProvenanceResult(
        provider=Provider.RUNTIME,
        language=None,
        actions=[ActionType.HOST_OPTIMIZATION, ActionType.LAYOUT_CHANGE],
        default_extraction=ExtractionLevel.E4,
        build_system="runtime",
        confidence=0.95,
        dispatch_chain=["runtime", name],
        notes=[
            "runtime memory operation, not a kernel: there is no source to extract. "
            "Actionable on the host side — pinned memory, fewer or larger transfers, "
            "overlap with compute — and reported rather than optimized in place."
        ],
    )


def resolve_unknown(name: str) -> ProvenanceResult:
    """Terminal fallback. Never proposes an optimization action."""
    return ProvenanceResult(
        provider=Provider.UNKNOWN,
        language=None,
        actions=[ActionType.PROFILE_MORE],
        default_extraction=None,
        confidence=0.2,
        dispatch_chain=[name],
        notes=[
            "no provenance: this kernel is a finding to report, not a target to guess "
            "at. If it holds significant GPU time, that is the headline, not a footnote."
        ],
    )


# Order matters: the most specific pattern wins. Runtime transfers first, because they
# are not kernels and should never fall through to a kernel resolver. Then Inductor
# before generic Triton, vLLM before generic SYCL, and library primitives before
# anything that merely mentions "gemm".
DEFAULT_RESOLVERS: tuple[Resolver, ...] = (
    resolve_runtime_transfer,
    resolve_inductor,
    resolve_onednn,
    resolve_vllm,
    resolve_sycl,
    resolve_triton,
)


def resolve(name: str, resolvers: tuple[Resolver, ...] = DEFAULT_RESOLVERS) -> ProvenanceResult:
    """Attribute one runtime kernel name, falling back to an explicit unknown."""
    if not name or not name.strip():
        return resolve_unknown(name or "")
    for resolver in resolvers:
        result = resolver(name)
        if result is not None:
            if not result.source.symbol:
                result.source.symbol = name
            # Every resolver in this chain matches a name pattern — the NAME_MATCH
            # tier, an estimate, not an exact hit. The build-graph and symbol-index
            # tiers live in the language backends and stamp their own methods.
            if result.source.method is ResolutionMethod.UNRESOLVED:
                result.source.method = ResolutionMethod.NAME_MATCH
            if result.source.confidence is None:
                result.source.confidence = result.confidence
            return result
    return resolve_unknown(name)


def extraction_tractability(level: ExtractionLevel | None) -> float:
    """How cheaply a kernel at this extraction level can be iterated on.

    An E2 bundle iterates in seconds, an E3 harness in minutes, an E4 kernel not at
    all. Used as a tie-breaker in ranking — and capped there, so it cannot overturn
    an order-of-magnitude difference in end-to-end headroom.
    """
    return {
        ExtractionLevel.E1: 1.0,
        ExtractionLevel.E2: 0.95,
        ExtractionLevel.E3: 0.55,
        ExtractionLevel.E0: 0.3,
        ExtractionLevel.E4: 0.1,
    }.get(level, 0.2)  # type: ignore[arg-type]


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None
