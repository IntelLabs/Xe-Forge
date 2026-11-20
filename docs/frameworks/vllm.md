# vllm

## 1. What it is

vLLM is an LLM inference and serving engine ([vllm-project/vllm](https://github.com/vllm-project/vllm)).

## 2. Workloads

`text`: decoder language models, at their own precision or a declared `--quantization`,
`--precision` and `--kv-cache`.

## 3. Command

```
xe-forge run --framework vllm --model <hub id> \
    --framework-source https://github.com/vllm-project/vllm@<tag> \
    --batch 4 --input-len 512 --output-len 32 --prompt-file <text file> \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- With an XPU build of vLLM already installed, pass `--framework-python <venv>/bin/python`
  instead of `--framework-source`.
- `--framework-note` is not usually needed.
- `--model` is a Hugging Face id or a local snapshot.

## 4. What the setup agent builds

Generic: vLLM from its XPU install instructions, the XPU build of torch first, and the
dependencies that serve only another vendor's device left out (listed in `dropped`).

The capture agent drives vLLM's offline engine with the engine core in the same process, so
torch.profiler sees its kernels, eagerly (no compilation, no CUDA graphs), with prefix caching
off and the prompts passed as token ids.

## 5. Kernel repositories

| Repository | Holds | Routes |
|---|---|---|
| [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) | the SYCL kernels vLLM's XPU ops launch (attention, norms, activations, rotary embedding, KV cache, quantization), and oneDNN GEMM calls | `kernel` (`sycl`), `library` |
| [vllm](https://github.com/vllm-project/vllm) | Triton kernels (KV cache, sampling and others), and the code that makes oneDNN GEMM calls | `kernel` (`triton`), `library` |

A patch the model needs to run (a missing architecture or op) is made in a clone of `vllm`
under `--kernel-repos`; later kernels of the run may then be named in that clone.

## 6. Known limits

- The capture runs eagerly, so kernels only a compiled path would generate are not seen.
- A number format this device has no kernel for, only an emulation, stops the run
  (`"unsupported": "precision"`); it is not patched.

## 7. Status

Run end to end.

Last checked: 2026-11-17, Xe-Forge 05c492a
