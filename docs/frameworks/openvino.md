# openvino

## 1. What it is

OpenVINO is Intel's toolkit for optimizing and running inference, whose GPU plugin runs
models as OpenCL kernels and oneDNN primitives ([openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)).

## 2. Workloads

`text`: decoder language models as an OpenVINO IR, at fp16 or with compressed (int8, int4)
weights.

## 3. Command

```
x-forge run --framework openvino --model <hub id of an OpenVINO IR> \
    --framework-source https://github.com/openvinotoolkit/openvino@<release tag> \
    --batch 4 --input-len 512 --output-len 32 --prompt-file <text file> \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 \
    -o runs/<name>
```

- With OpenVINO already installed, pass `--framework-python <venv>/bin/python` instead of
  `--framework-source`.
- `--model` is a pre-converted IR (a hub repository of `.xml` and `.bin`), or a checkpoint the
  capture agent converts with OpenVINO's own tools into
  `$X_FORGE_CACHE/models/openvino/`.
- The precision is the IR's: leave `--precision` and `--quantization` at `auto` unless the run
  is meant to declare them.

## 4. What the setup agent builds

Generic: the OpenVINO Python package from the project's own install docs.

OpenVINO is not a PyTorch framework, so the capture agent writes `ops.json`: the executed nodes
from the runtime model and its per-infer profiling information, shapes from OpenVINO's shape
inference, arguments named by role, and each node's kernels and launch ranges aligned to
unitrace's timeline. Generation is a hand-written greedy loop over the compiled model.

## 5. Kernel repositories

| Repository | Holds | Routes |
|---|---|---|
| [openvino](https://github.com/openvinotoolkit/openvino) | the GPU plugin's OpenCL kernels (attention, copies and reorders, KV cache, norms, rotary embedding), and its oneDNN GEMM and convolution calls | `kernel` (`opencl`), `library` |

## 6. Known limits

- The GPU plugin may reorder weights on a compiled model's first execution, so a warm-up run
  can launch more than the measured run; the launcher must make the runs it calls identical
  (`repeats` in `meta.json`).

## 7. Status

Run through optimize.

Last checked: 2026-11-17, X-Forge 05c492a
