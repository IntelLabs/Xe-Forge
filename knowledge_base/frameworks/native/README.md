# The native framework

A model's own inference engine, written for it: a C++ core, C++ kernels on the host, SYCL
kernels on an Intel GPU, behind one small kernel interface. It is a framework like any other
to X-Forge's run. The capture agent finds no engine for the model and reports it unsupported;
the patch agent writes the engine in its own repository (`contract.md`); `patch check` gates
its tokens; the run then captures it, names its ops, puts it against the roofline, and
optimizes its kernels in that repository like any other kernel repository.

The goal is the roofline of the whole model. The run prints it (`Model roofline: ...` in the
run's output and in `results/SUMMARY.md`): the time every step would take at the device's
measured bandwidth and peak, against the time the model took (on a host, its wall time with
the timeline off, so what runs between steps counts). A framework that dispatches each small
op on its own pays a fixed cost per op that no kernel change removes; an engine written for one
model can pay it once per step instead (`decode_single_region`, `spinning_pool`).

- `framework.yaml` -- facts the run's code reads: no upstream to clone, and the quantized gate.
- `contract.md` -- what the tools and the run expect of an engine. Not negotiable.
- `runtime/` -- runtime C to start from: safetensors index (`xf_st.c`), weight binding with
  bf16 / raw-byte storage (`xf_bind.c`), dump and profile hooks (`xf_hooks.c`), a command line
  (`xf_run.c`), and the shared header (`xf.h`). Bring it into the C++ core and adapt it to
  `contract.md`; it predates the kernel interface and the timeline.
- `patterns.yaml` -- what earlier engines established: a rule, the pattern, how to check it.
  Add to it what this engine establishes, in its form, with no measured numbers, model names
  or device names.
- `examples/` -- hot-path host kernels earlier engines kept (AMX prefill linear with an
  fp32-accurate hi/lo split, AVX-512 decode GEMV over bf16 weights, full-team decode
  attention). Read, adapt and test again; their signatures are those engines', not a contract.

Implementations of low-bit formats worth reading before writing one: llama.cpp's CPU kernels
(`ggml/src/ggml-cpu/`: the quants and the x86 paths), its SYCL backend (`ggml/src/ggml-sycl/`)
and the format definitions in `ggml/src/ggml-quants.c`.
