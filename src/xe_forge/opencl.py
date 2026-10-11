"""Run an OpenCL C kernel on torch's XPU queue, for ``--dsl opencl`` trial modules.

    from xe_forge.opencl import load
    k = load(cl_source, "my_kernel", defines={"INPUT0_TYPE": "half"})
    k((global_size,), (local_size,), out, x, n)       # tensors -> device pointers, int -> int32

The source is compiled ahead of time -- oneAPI ``clang`` (OpenCL C -> LLVM bitcode) then
``llvm-spirv`` -- and the SPIR-V is loaded through SYCL on torch's current queue, so the
kernel runs in the same stream as every other op: torch events time it and the output is
ordinary tensors. (Runtime OpenCL compilation would need ``ocloc``; SPIR-V goes to the
driver directly.) Compiled SPIR-V is cached by content under ``~/.cache/xe_forge/opencl``.

Arguments: a tensor passes its device pointer (``__global`` buffers); a Python ``int`` is
an ``int``; a ``float`` is a ``float``. Wrap a 64-bit integer as ``long_(v)``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CACHE = Path(os.environ.get("XE_FORGE_OPENCL_CACHE", Path.home() / ".cache/xe_forge/opencl"))
# Every argument that shapes the module is part of the cache key, so changing one rebuilds.
_CLANG_ARGS = ["-target", "spir64", "-O2", "-Xclang", "-finclude-default-header", "-emit-llvm"]
_SPIRV_ARGS = [
    "--spirv-ext=+all,-SPV_KHR_untyped_pointers"
]  # the GPU loader rejects untyped pointers

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


def compile_spirv(source: str, defines: dict | None = None, options: str = "") -> bytes:
    """OpenCL C -> SPIR-V bytes, cached by (source, defines, options)."""
    flags = [f"-D{k}={v}" if v is not None else f"-D{k}" for k, v in (defines or {}).items()]
    flags += options.split()
    key = hashlib.sha256(
        "\0".join([source, *flags, *_CLANG_ARGS, *_SPIRV_ARGS]).encode()
    ).hexdigest()[:24]
    cached = CACHE / f"{key}.spv"
    if cached.is_file():
        return cached.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        cl, bc, spv = Path(tmp, "k.cl"), Path(tmp, "k.bc"), Path(tmp, "k.spv")
        cl.write_text("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" + source)
        std = [] if any(f.startswith("-cl-std") for f in flags) else ["-cl-std=CL3.0"]
        cmd = [_tool("clang"), *std, *_CLANG_ARGS, "-c", str(cl), "-o", str(bc), *flags]
        for step in (cmd, [_tool("llvm-spirv"), *_SPIRV_ARGS, str(bc), "-o", str(spv)]):
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
    """The SYCL side (built once per process): SPIR-V -> kernel bundle, and launch."""
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


def load(source: str, name: str, defines: dict | None = None, options: str = "") -> Kernel:
    """Compile ``source`` and return kernel ``name`` ready to launch on torch's queue."""
    return Kernel(runtime().build(compile_spirv(source, defines, options)), name)
