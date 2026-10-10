"""Captured op -> what the kernel-locator is told to look for.

This is a lookup, not a resolution: the namespace an op is registered under says which
repository carries it, and the op's name is what the locator greps for. Reading the
source, the registration and the build is the locator's job, inside Xe-Forge.

An op registered in Python (``vllm::unified_attention_with_output``) reaches its kernel
through a provider op (``_vllm_fa2_C::varlen_fwd``); the trace parser already keys such
work on the provider op, with its own arguments, so that is the one named. A kernel
launched outside any op is a Triton kernel launched from the framework's own Python, so its
repository is the framework's.

Routes: ``kernel`` (a named kernel in a repository Xe-Forge can edit), ``library`` (a
library primitive such as oneDNN -- change the call, not the kernel), ``aten`` (an op
inside PyTorch, no kernel repository) and ``unnamed`` (no rule matched).
"""

from __future__ import annotations

import re

from xe_forge.frontend.ir import CapturedOp, Naming

# torch.ops namespace -> (repository, DSL). One row per provider; a new provider is a row.
PROVIDERS: dict[str, tuple[str, str]] = {
    "_C": ("vllm-xpu-kernels", "sycl"),
    "_C_cache_ops": ("vllm-xpu-kernels", "sycl"),
    "_C_custom_ar": ("vllm-xpu-kernels", "sycl"),
    "_moe_C": ("vllm-xpu-kernels", "sycl"),
    "_xpu_C": ("vllm-xpu-kernels", "sycl"),
    "_vllm_fa2_C": ("vllm-xpu-kernels", "sycl"),
    "_vllm_fa3_C": ("vllm-xpu-kernels", "sycl"),
    "sgl_kernel": ("sgl-kernel-xpu", "sycl"),
}

# Kernel symbols a library primitive launches; an op whose time is all in these is a call.
LIBRARY_KERNELS = re.compile(r"^(gemm_kernel|xe_|gen_|ref_|jit:|dnnl|onednn|ocl:)")

TRITON_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Where a framework's own Python (and so its Triton kernels) lives.
FRAMEWORK_REPOS = {"vllm": "vllm", "sglang": "sglang"}


def _namespace(op: str) -> str:
    return op.split("::", 1)[0] if "::" in op else ""


def _base(op: str) -> str:
    return op.split("::", 1)[-1].split(".", 1)[0]


def name_op(op: CapturedOp, framework: str = "vllm") -> Naming:
    impl = op.framework_op
    ns = _namespace(impl)
    if ns in PROVIDERS:
        repo, dsl = PROVIDERS[ns]
        op.impl_class = "provider"
        return Naming("kernel", _base(impl), repo, dsl, via=f"torch.ops.{ns}")
    if ns == "kernel":
        # A Triton kernel is named after its @triton.jit function: a plain identifier.
        # ATen functors, templates and copies launched outside any op are not.
        if not TRITON_NAME.match(_base(impl)):
            op.impl_class = "unknown"
            return Naming("unnamed", _base(impl), via="non-Triton kernel launched outside any op")
        repo = FRAMEWORK_REPOS.get(framework)
        op.impl_class = "triton"
        if repo is None:
            return Naming("unnamed", _base(impl), via="kernel launched outside any op")
        return Naming("kernel", _base(impl), repo, "triton", via="launched outside any op")
    symbols = list(op.kernel_symbols)
    if symbols and all(LIBRARY_KERNELS.match(k) for k in symbols):
        op.impl_class = "onednn"
        return Naming("library", _base(impl), via="library primitive: " + symbols[0])
    if ns == "aten":
        op.impl_class = "aten"
        return Naming("aten", _base(impl), via="ATen op inside PyTorch")
    op.impl_class = "python" if ns else "unknown"
    return Naming("unnamed", _base(impl), via=f"no rule for namespace {ns!r}")
