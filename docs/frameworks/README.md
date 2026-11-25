# Frameworks

Nothing in the code or in the agent templates reads these pages or names a framework:
`--framework` is free text handed to the agents. A framework may also have a knowledge base,
`knowledge_base/frameworks/<framework>/`: the agents are told to read it first, and the facts
in its `framework.yaml` (no upstream to clone, a quantized gate) change what they are asked
and what the patch check gates, never a test of the name. The pages record what a person passes to
`xe-forge run` for a framework and what the agents do with it. How a run works is in
[FRAMEWORK.md](../FRAMEWORK.md); every flag is in [REFERENCE.md](../REFERENCE.md).

| `--framework` | Workloads | Status | Page |
|---|---|---|---|
| `vllm` | text | run end to end | [vllm.md](vllm.md) |
| `sglang` | text | run end to end | [sglang.md](sglang.md) |
| `llama.cpp` | text | run through prove; compiled-runtime support still being generalized | [llama.cpp.md](llama.cpp.md) |
| `openvino` | text | run through optimize | [openvino.md](openvino.md) |
| `vllm-omni` | vision | run | [vllm-omni.md](vllm-omni.md) |
| `RLDX-1` | action | being brought up | none yet |
| `native` | text | run end to end: the engine is written for the model | [native.md](native.md) |

A page is named by the string passed to `--framework`.

## Adding a page

1. Copy [TEMPLATE.md](TEMPLATE.md) to `<framework>.md`.
2. Run the framework with `xe-forge run --framework <framework> ...`.
3. Fill sections 3 to 5 from the run: `<run>/capture-request.json` (what was asked),
   `<run>/setup/framework.json` and `<venv>/xe-forge-setup.json` (what the setup agent built,
   dropped and noted), and the `repo` fields of `<run>/names.json` (where the kernels live).
4. Add the row above.

Write no run ids, job numbers, timings or speed figures on a page, and no machine paths: use
placeholders such as `<venv>`, `<repos>`, `<calibration.json>`.
