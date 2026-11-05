# Native engine knowledge base

Read by the agent that writes a model's native C engine (`native-engine.md.j2`).

- `contract.md` -- what the measuring tools expect of an engine. Not negotiable.
- `runtime/` -- runtime C to copy and adapt: safetensors index (`xf_st.c`), weight binding
  with bf16 / raw-byte storage (`xf_bind.c`), dump and profile hooks (`xf_hooks.c`), the CLI
  (`xf_run.c`), and the shared header (`xf.h`).
- `patterns.yaml` -- what earlier engines established: a rule, the pattern, how to check it.
- `examples/` -- hot-path kernels earlier engines kept (AMX prefill linear with fp32-accurate
  hi/lo split, AVX-512 decode GEMV over bf16 weights, full-team decode attention). Read, adapt,
  re-verify; their signatures are those engines', not a contract.

Implementations of low-bit formats worth reading before writing one: llama.cpp's CPU kernels
(`ggml/src/ggml-cpu/` -- quants and the x86 paths) and the format definitions in
`ggml/src/ggml-quants.c`; kimi-k3-in-c's `src/core/k3_gq.c` on its 2-bit branches (IQ2_XS with
VNNI, Q8_0).
