# llama.cpp

## 1. What it is

llama.cpp is LLM inference in C/C++ on the ggml tensor library, with a SYCL backend for Intel
GPUs ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)).

## 2. Workloads

`text`: decoder language models in GGUF, converted from a Hugging Face checkpoint or published
as GGUF.

## 3. Command

```
x-forge run --framework llama.cpp --model <hub id> \
    --framework-python <venv>/bin/python --reference-python <torch venv>/bin/python \
    --batch 4 --input-len 512 --output-len 32 --prompt-file <text file> \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- `--framework-python`: a venv with llama-cpp-python built on llama.cpp's SYCL backend. The
  setup agent can build one instead (`--framework-source`).
- `--reference-python`: an interpreter with GPU torch and transformers. The llama.cpp venv
  has neither, and `patch check` scores tokens against transformers.
- The SYCL backend needs the oneAPI runtime libraries (oneMKL, oneDNN, TBB) on the library
  path before the run starts; X-Forge does not set them.
- `--model` is a Hugging Face id. The capture agent converts it with llama.cpp's
  `convert_hf_to_gguf.py` into `$X_FORGE_CACHE/models/llama.cpp/`, reused when it is there.

## 4. What the setup agent builds

Generic: see [TEMPLATE.md](TEMPLATE.md).

llama.cpp is not a PyTorch framework, so the capture agent writes `ops.json` instead of a
torch trace: a ggml scheduler callback records every graph node (op, argument shapes and
dtypes), and each node's kernel-name patterns are checked against what unitrace saw. It runs
eagerly (llama.cpp's own `GGML_SYCL_ENABLE_GRAPH=0`) and clears the KV cache before each repeat.

## 5. Kernel repositories

| Repository | Holds | Routes |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | the ggml SYCL kernels (norms, elementwise, attention, rotary embedding, copies, KV cache, quantized GEMM), and the oneDNN and oneMKL GEMM calls | `kernel` (`sycl`), `library` |

## 6. Known limits

- The token-embedding lookup (`GET_ROWS`) runs on the CPU by llama.cpp's design, so it is not
  in the device time.

## 7. Status

Run through prove; compiled-runtime support still being generalized.

Last checked: 2026-11-17, X-Forge 05c492a
