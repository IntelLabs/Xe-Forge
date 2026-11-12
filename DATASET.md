# Workload inputs

## Phase 1: real text for LLMs

Done:
- `xe-forge run --prompt-file` slices the text into batch × input_len token ids
  (`profile.token_ids(..., prompt_file=...)`). The capture launcher and the patch check's
  `generate.sh` both use it; without it, the run uses the synthetic ids.
- The run copies the file into `inputs/` and records it in `meta.json` as
  `runtime.inputs` (`{name: {"file", "sha256"}}`). `capture parse` fails when a file is
  missing or its hash has changed. A resumed run with a different prompt file captures again.
- The patch check scores on the run's prompt file. A sequence that fails in bf16 is judged
  again in float32, and that verdict is final. The tolerance stays at 0.3 nat.
- Tests cover the slicing, the hash check, and passing the file to the patch check.

Open:
- Pin a default text set: a WikiText-2 slice, or a few ShareGPT conversations if the
  licence allows. Download it at a fixed revision, enough for batch × input_len, and allow
  `--dataset name@revision` alongside `--prompt-file`.
- Decide whether the patch check needs mean KL or perplexity as well as the per-token gap.

## Phase 2: realistic workload shapes

- Several workload points instead of one: a list of (batch, input_len, output_len), or a
  length distribution sampled from ShareGPT or BurstGPT. Capture each point and merge them;
  `inputs.json` already weights several configurations.
- MoE routing: require real text for MoE models, and record how many tokens each expert
  received. Without that, the grouped-GEMM shapes and weights are wrong.

## Phase 3: real activations in kernel tests

- An optional launcher hook that saves a few input tensors per captured shape into
  `raw/activations/`, listed in `runtime.inputs` so `capture parse` hashes them.
- `skills/benchmark.py` checks correctness on those tensors as well as on `get_inputs`.
  This catches fp8, int8 and fused-norm kernels that pass only on Gaussian inputs.

## Phase 4: VLMs

- Media inputs in the launcher contract: a fixed image list plus prompts, as jsonl of
  prompt + image files, each file in `runtime.inputs`.
- Replace the "text decoder that runs on token ids" rule in `frontend-capture.md.j2`. Image
  sizes come from the pinned set, since resolution sets the vision encoder's shapes.
- Image sets: VisionArena-Chat for a serving mix; a DocVQA or ChartQA slice for high
  resolution.
- Correctness: the same teacher-forced check with images in the context. For `--prove`, a
  small benchmark slice (MMBench, OCRBench; about 100 samples each).

## Phase 5: VLAs

- A VLA workload: recorded observations (camera frames, instruction, robot state) from
  LIBERO, at batch 1. `meta.json` records the denoising steps and the action chunk size.
- Correctness per trial: L2 error between predicted and reference actions on those
  observations. This needs a new check beside the patch check.
- Prove: task success rate, only for winning patches, since it is expensive.
- Score per control step at batch 1 (latency), not throughput.

## Housekeeping

- Check licences before using any dataset; ShareGPT especially. Download at a pinned
  revision rather than committing data to the repo.
