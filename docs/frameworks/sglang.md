# sglang

## 1. What it is

SGLang is a serving framework for language models ([sgl-project/sglang](https://github.com/sgl-project/sglang)).

## 2. Workloads

`text`: decoder language models, at their own precision or a declared `--precision` and
`--kv-cache`.

## 3. Command

```
xe-forge run --framework sglang --model <hub id> \
    --framework-source https://github.com/sgl-project/sglang@<tag> \
    --batch 4 --input-len 512 --output-len 32 --prompt-file <text file> \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- With an XPU build of SGLang already installed, pass `--framework-python <venv>/bin/python`
  instead of `--framework-source`.
- `--framework-note` is not usually needed.
- `--model` is a Hugging Face id or a local snapshot.

## 4. What the setup agent builds

Generic: SGLang from its XPU install instructions, the XPU build of torch first, and the
dependencies that serve only another vendor's device left out (listed in `dropped`).

The capture agent starts an in-process `sgl.Engine` eagerly (no CUDA graphs, no
`torch.compile`), with the radix cache off and a prefill chunk large enough for the batch to be
one prefill, and records the trace with SGLang's own `start_profile` / `stop_profile`.

## 5. Kernel repositories

| Repository | Holds | Routes |
|---|---|---|
| [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) | SGLang's SYCL kernels for XPU (attention, norms, activations, rotary embedding, KV cache) | `kernel` (`sycl`) |
| [sglang](https://github.com/sgl-project/sglang) | Triton kernels (attention), and the code that makes oneDNN GEMM calls | `kernel` (`triton`), `library` |

A patch for a missing architecture is made in a clone of `sglang` under `--kernel-repos`.

## 6. Known limits

- The model runs in a scheduler subprocess. The trace is recorded there, and the launcher has
  to let that process exit normally, or unitrace writes no data for it.
- The capture runs eagerly, so kernels only a compiled path would generate are not seen.

## 7. Status

Run end to end.

Last checked: 2026-11-17, Xe-Forge 05c492a
