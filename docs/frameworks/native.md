# native

## 1. What it is

A model's own inference engine, written by Xe-Forge's patch agent for that model: a C++ core,
C++ kernels on the host and SYCL kernels on an Intel GPU, behind one small kernel interface.
It has no upstream; its knowledge base is
[knowledge_base/frameworks/native/](../../knowledge_base/frameworks/native/README.md), and
its contract is that directory's `contract.md`.

## 2. Workloads

`text`: decoder language models, batch 1 first.

## 3. Command

```
xe-forge run --framework native --model <hub id or checkpoint dir> --device cpu \
    --framework-python <python with torch and transformers> \
    --batch 1 --input-len 512 --output-len 32 --prompt-file <text file> \
    [--quantization <scheme>] \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- `--framework-python` is required: there is no framework to set up. The interpreter holds
  torch and transformers; the engine's binding is layered over it in its own venv, and the
  reference of the patch check runs under it.
- `--device cpu` runs the host kernels; `--device xpu` (or `xpu:N`) the SYCL ones. On a GPU,
  export `XE_FORGE_TRACE_STEPS=0` so the device trace covers every decode step.
- `--quantization` (optional; without it the checkpoint runs as stored): the engine quantizes the weights itself. The quantized tokens are gated
  against the original model holding the weights the quantized form decodes to, and how far
  each tensor is from the checkpoint is reported (`patch check --quantized-gate`).

## 4. What the setup agent builds

Nothing: with `--framework-python` there is no setup. The capture agent finds no engine for
the model and reports it unsupported; the patch agent writes the engine in a new repository
under `--kernel-repos` (a skeleton commit as the base, the model's engine on top), builds it,
and the patch check gates it. Capture then runs on it, with the engine's per-step timeline
(`XF_TIMELINE`) as the host's node timeline, or the device trace on a GPU.

## 5. Kernel repositories

The engine's own repository, `native-<model>` under `--kernel-repos`: the host kernels in
`src/kernels/cpu/` (cpp), the GPU kernels in `src/kernels/sycl/` (sycl). The run optimizes
them there. On a host, each trial is a directory of the repository's files: the engine is
built with them (`build.sh`, with its kernel tests) and timed inside the model by the run's
own launcher (`xe-forge-skill inmodel`), so a kernel that runs inside the engine's parallel
region is timed there, and its tokens are scored again on that build. `--prove` scores them
once more on the build with every winning kernel.

## 6. The goal

The whole model at its roofline: `Model roofline:` in the run's output and in
`results/SUMMARY.md` puts the captured time against the bound of every op at the device's
measured bandwidth and peak.
