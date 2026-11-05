# Native engine generation (`xe-forge generate`)

An agent writes a native C engine for a model and makes its hot paths fast; tools measure it.
The framework is that agent, its instructions
(`src/xe_forge/claude/templates/native-engine.md.j2`), a knowledge base
(`knowledge_base/native/`) and the measuring tools. Nothing model-specific is code.

```bash
MODEL=Qwen/Qwen3-1.7B OUT=../native-runs/qwen3 sbatch scripts/submit_native_generate.sh
```

The agent works through four steps:
1. **Extract.** It writes one PyTorch function per operation of the forward pass, copied from
   the model's Python.
2. **Design.** It lays out the engine from the knowledge base's runtime.
3. **Write C.** It writes the engine, then verifies it, first on a surrogate and then on the
   model.
4. **Hot paths.** It rewrites the steps the profile shows to be hot, using intrinsics or
   assembly, and keeps a change only when verify passes and the time improved.

| Tool | Measures |
|---|---|
| `python -m xe_forge.frontend.surrogate inspect / build --overrides` | the model's config and checkpoint; a random model of the same architecture at a small config, saved in the checkpoint's layout |
| `python -m xe_forge.frontend.verify` | the engine against the reference, through `knowledge_base/native/contract.md`: every dumped step against its module, the logits, incremental decode against recompute, and greedy tokens |
| `XF_PROFILE` (the engine's runtime) | time per step at prefill and at decode |

`generate` runs the agent, then gates its engine with `verify` on the surrogate and on the
model, and writes `GENERATE.md`. When transformers cannot run the model, `--reference-source`
points at its Python, and the agent writes a reference adapter (`reference.py`) from it.
