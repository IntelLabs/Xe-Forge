# vllm-omni

## 1. What it is

vLLM-Omni extends vLLM to models that generate images, video and audio, diffusion
transformers among them ([vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)).

## 2. Workloads

`vision`: text-to-image, image editing and text- or image-to-video diffusion models, measured
over their denoising steps at a resolution.

## 3. Command

```
x-forge run --framework vllm-omni --model <hub id> --workload vision \
    --framework-source https://github.com/vllm-project/vllm-omni@<tag> \
    --steps 28 --height 1024 --width 1024 --prompt "<prompt>" \
    --calibration <calibration.json> --kernel-repos <repos> --top 3 --prove \
    -o runs/<name>
```

- With vLLM-Omni already installed over an XPU build of vLLM, pass
  `--framework-python <venv>/bin/python` instead of `--framework-source`.
- A video model takes `--frames`; an editing or image-to-video model takes `--input-image`
  (repeatable). The input image's size sets the shapes the model runs at.
- `--negative-prompt` only for a pipeline that takes one.
- `--framework-note` is not usually needed.

## 4. What the setup agent builds

Generic: vLLM-Omni over an XPU build of vLLM, from the projects' install docs.

The capture agent may turn on weight offloading when a model does not fit the device; it never
lowers the precision to make one fit. A model vLLM-Omni has no pipeline for, or an op it runs
only on the CPU or in plain torch on this device, goes to the patch step first: the patch adds
the pipeline or the op, and `patch check` compares its output with the framework's own torch
path for that op (`X_FORGE_NATIVE=1`), stage by stage ([FRAMEWORK.md](../FRAMEWORK.md#output-quality-checks)).

## 5. Kernel repositories

vLLM-Omni runs on vLLM, so most kernels are in the repositories on the [vllm](vllm.md) page
(`vllm-xpu-kernels`, `vllm`). Pipelines, and patches that add one, are in
[vllm-omni](https://github.com/vllm-project/vllm-omni).

## 6. Known limits

- An optimized kernel is kept on its per-kernel tolerance, and `--prove` compares device time.
  A difference within that tolerance can still grow over the denoising steps: look at the
  images in `outputs/` before using a patch.
- An encoded video is not compared (the encoder is lossy); the frames are, as `.npy`.

## 7. Status

Run (the models in [RESULTS.md](../RESULTS.md#vision-models)).

Last checked: 2026-11-17, X-Forge 05c492a
