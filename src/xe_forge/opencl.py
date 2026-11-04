"""Run an OpenCL C kernel on torch's XPU queue: ``load(src, name)(gws, lws, *args)``.

Compiled ahead of time to SPIR-V and cached. A tensor passes its device pointer, ``int`` is
int32, ``float`` is float, ``long_(v)`` is int64.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CACHE = Path(
    os.environ.get("XE_FORGE_OPENCL_CACHE")
    or Path(os.environ.get("XE_FORGE_CACHE", Path.home() / ".cache/xe_forge")) / "opencl"
)
_PIPELINE = ["-target", "spir64", "-Xclang", "-finclude-default-header", "-emit-llvm", "-c"]
OPTIONS = os.environ.get("XE_FORGE_OPENCL_OPTIONS", "-cl-std=CL3.0 -O2")
# The llvm-spirv default emits an extension the GPU loader rejects.
SPIRV_OPTIONS = os.environ.get(
    "XE_FORGE_SPIRV_OPTIONS", "--spirv-ext=+all,-SPV_KHR_untyped_pointers"
)

_CPP = "#include <torch/extension.h>\nint64_t build(py::bytes spirv);\nvoid launch(int64_t h, std::string name, std::vector<int64_t> gws, std::vector<int64_t> lws, py::list args);"

_SYCL = r"""
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
namespace syclex = sycl::ext::oneapi::experimental;
using exe_bundle = sycl::kernel_bundle<sycl::bundle_state::executable>;
static std::vector<exe_bundle> bundles;

int64_t build(py::bytes spirv) {
  std::string s = spirv;
  std::vector<std::byte> bytes(s.size());
  std::memcpy(bytes.data(), s.data(), s.size());
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::spirv, bytes);
  bundles.push_back(syclex::build(src));
  return (int64_t)bundles.size() - 1;
}

template <int D> sycl::nd_range<D> range(const std::vector<int64_t>& g, const std::vector<int64_t>& l) {
  sycl::range<D> G, L;
  for (int i = 0; i < D; ++i) { G[i] = g[i]; L[i] = l[i]; }
  return sycl::nd_range<D>(G, L);
}

void launch(int64_t h, std::string name, std::vector<int64_t> gws, std::vector<int64_t> lws, py::list args) {
  TORCH_CHECK(gws.size() == lws.size() && gws.size() >= 1 && gws.size() <= 3, "1-3 dims, gws/lws same rank");
  sycl::kernel k = bundles.at(h).ext_oneapi_get_kernel(name);
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.submit([&](sycl::handler& cgh) {
    for (size_t i = 0; i < args.size(); ++i) {
      py::handle a = args[i];
      if (THPVariable_Check(a.ptr())) cgh.set_arg(i, THPVariable_Unpack(a.ptr()).data_ptr());
      else if (py::hasattr(a, "_xe_forge_long")) cgh.set_arg(i, a.attr("value").cast<int64_t>());
      else if (py::isinstance<py::bool_>(a) || py::isinstance<py::int_>(a)) cgh.set_arg(i, a.cast<int32_t>());
      else if (py::isinstance<py::float_>(a)) cgh.set_arg(i, a.cast<float>());
      else TORCH_CHECK(false, "argument ", i, ": tensor, int, float or long_ expected");
    }
    if (gws.size() == 1) cgh.parallel_for(range<1>(gws, lws), k);
    else if (gws.size() == 2) cgh.parallel_for(range<2>(gws, lws), k);
    else cgh.parallel_for(range<3>(gws, lws), k);
  });
}
"""

_runtime = None


class long_:
    """A 64-bit integer kernel argument (``long``, ``size_t``)."""

    _xe_forge_long = True

    def __init__(self, value: int):
        self.value = int(value)


def _tool(name: str) -> str:
    for root in (os.environ.get("XE_FORGE_CLANG_DIR"), os.environ.get("CMPLR_ROOT")):
        if root:
            for p in (Path(root) / name, Path(root) / "bin" / "compiler" / name):
                if p.is_file():
                    return str(p)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"{name} not found: set XE_FORGE_CLANG_DIR or source oneAPI (CMPLR_ROOT)"
        )
    return found


_TOOLS: str | None = None


def _versions() -> str:
    global _TOOLS
    if _TOOLS is None:
        out = []
        for name in ("clang", "llvm-spirv"):
            try:
                tool = _tool(name)
                r = subprocess.run([tool, "--version"], capture_output=True, text=True)
                out.append(f"{tool}:{r.stdout.strip().splitlines()[:1]}")
            except (RuntimeError, OSError):
                out.append(f"{name}:?")
        _TOOLS = ";".join(out)
    return _TOOLS


def compile_spirv(
    source: str,
    defines: dict | None = None,
    options: str = "",
    spirv_options: str | None = None,
) -> bytes:
    """OpenCL C -> SPIR-V bytes, cached by everything that shapes them."""
    flags = [f"-D{k}={v}" if v is not None else f"-D{k}" for k, v in (defines or {}).items()]
    clang = [*_PIPELINE, *OPTIONS.split(), *options.split(), *flags]
    spirv = (SPIRV_OPTIONS if spirv_options is None else spirv_options).split()
    tools = _versions()  # another compiler makes another module
    key = hashlib.sha256("\0".join([source, *clang, "|", *spirv, tools]).encode()).hexdigest()[:24]
    cached = CACHE / f"{key}.spv"
    if cached.is_file():
        return cached.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        cl, bc, spv = Path(tmp, "k.cl"), Path(tmp, "k.bc"), Path(tmp, "k.spv")
        cl.write_text(source)
        for step in (
            [_tool("clang"), *clang, str(cl), "-o", str(bc)],
            [_tool("llvm-spirv"), *spirv, str(bc), "-o", str(spv)],
        ):
            r = subprocess.run(step, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"OpenCL compile failed: {' '.join(step)}\n{r.stderr}")
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copy(spv, cached)
    return cached.read_bytes()


class Kernel:
    def __init__(self, handle: int, name: str):
        self.handle, self.name = handle, name

    def __call__(self, global_size, local_size, *args):
        _runtime.launch(self.handle, self.name, list(global_size), list(local_size), list(args))


def runtime():
    """The SYCL extension that builds SPIR-V bundles and launches them, built once."""
    global _runtime
    if _runtime is None:
        from torch.utils.cpp_extension import load_inline

        _runtime = load_inline(
            name="xe_forge_opencl_runtime",
            cpp_sources=_CPP,
            sycl_sources=_SYCL,
            functions=["build", "launch"],
            verbose=False,
        )
    return _runtime


def load(
    source: str,
    name: str,
    defines: dict | None = None,
    options: str = "",
    spirv_options: str | None = None,
) -> Kernel:
    """Compile ``source`` and return kernel ``name`` ready to launch on torch's queue."""
    spirv = compile_spirv(source, defines, options, spirv_options)
    return Kernel(runtime().build(spirv), name)
