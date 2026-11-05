"""Verify a native engine against its reference model (transformers, or a reference adapter's
``load``), fp32 on CPU. The engine is measured only through knowledge_base/native/contract.md:
its dumps name the reference module each step reproduces.

Levels (each a ``GATE <level> PASS|REJECT <evidence>`` line, then ``VERDICT`` and ``DONE``):
  layer   every step with a ``module`` vs that module's output, prefill of the prompt
  logits  the prefill's last-position logits
  decode  incremental decode == full recompute, token for token
  tokens  greedy tokens == transformers' greedy tokens

    python -m xe_forge.frontend.verify --model-dir DIR --engine xf-run -o v.json [--reference ref.py]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

STEP_RTOL = 1e-3  # max |native - reference| / max |reference| per step output
LOGITS_RTOL = 1e-3


def _engine(engine: Path, weights: Path, ids: list[int], new: int, env: dict, extra=()) -> dict:
    cmd = [str(engine), str(weights), "--ids", ",".join(map(str, ids)), "--new", str(new), *extra]
    if env.get("XF_NUMA") is not None:
        node = env["XF_NUMA"]
        cmd = ["numactl", f"--cpunodebind={node}", f"--membind={node}", *cmd]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or "DONE" not in r.stdout:
        raise RuntimeError(f"engine failed ({r.returncode}): {r.stderr.strip()[-2000:]}")
    out: dict = {}
    for line in r.stdout.splitlines():
        key, _, value = line.partition(":")
        if key == "TOKENS":
            out["tokens"] = [int(x) for x in value.split()]
        elif key in ("LOAD_MS", "PREFILL_MS", "DECODE_MS_PER_TOKEN"):
            out[key.lower()] = float(value)
    return out


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-12))


def _logits(out):
    """A forward's logits: ``.logits``, the first of a tuple, or the tensor itself."""
    if hasattr(out, "logits"):
        return out.logits
    return out[0] if isinstance(out, (tuple, list)) else out


def _reference(model_dir: Path, modules: list[str], config: dict, ids: list[int], new: int,
               reference: Path | None = None):  # fmt: skip
    """(module outputs on the prompt keyed by module path, last logits, greedy tokens). The
    reference is the transformers model, or what a reference adapter's ``load`` builds."""
    import torch

    if reference is not None:
        from xe_forge.frontend.surrogate import adapter

        m = adapter(reference)["load"](model_dir, config)
    else:
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32,
                                                 attn_implementation="eager")  # fmt: skip
    m = m.float().eval()  # a quantized checkpoint dequantizes to bf16 on CPU; exact in fp32
    outs: dict[str, np.ndarray] = {}
    hooks = []
    for path in modules:

        def hook(_mod, _inp, out, path=path):
            t = out[0] if isinstance(out, tuple) else out
            outs.setdefault(path, t.detach().float().reshape(len(ids), -1).numpy().copy())

        try:
            hooks.append(m.get_submodule(path).register_forward_hook(hook))
        except AttributeError:
            pass  # reported as unmatched by the caller
    with torch.no_grad():
        logits = _logits(m(torch.tensor([ids])))[0, -1].float().numpy()
    for h in hooks:
        h.remove()
    with torch.no_grad():
        if reference is not None:  # greedy by full recompute: no cache to trust
            seq = list(ids)
            for _ in range(new):
                seq.append(int(_logits(m(torch.tensor([seq])))[0, -1].argmax()))
            return outs, logits, seq[len(ids) :]
        gen = m.generate(torch.tensor([ids]), max_new_tokens=new, min_new_tokens=new,
                         do_sample=False, num_beams=1)  # fmt: skip
    return outs, logits, gen[0, len(ids) :].tolist()


def verify(
    model_dir: Path, engine: Path, ids: list[int], new: int, env: dict,
    config: dict | None = None, reference: Path | None = None,
) -> dict:  # fmt: skip
    from xe_forge.frontend.surrogate import merged_config

    gates: list[tuple[str, bool, str]] = []
    config = config or merged_config(model_dir)

    with tempfile.TemporaryDirectory() as tmp:
        dump, logits_file = Path(tmp) / "dump", Path(tmp) / "logits.bin"
        run = _engine(engine, model_dir, ids, new, {**env, "XF_DUMP": str(dump)},
                      ["--logits", str(logits_file)])  # fmt: skip
        rows = [json.loads(line) for line in (dump / "index.jsonl").read_text().splitlines()]
        for row in rows:
            row["path"] = row["module"].replace("{i}", str(row["layer"])) if row["module"] else ""
        modules = sorted({r["path"] for r in rows if r["path"]})
        ref_outs, ref_logits, ref_tokens = _reference(model_dir, modules, config, ids, new,
                                                      reference)  # fmt: skip
        nout = len(ref_logits)
        native_logits = np.fromfile(logits_file, dtype=np.float32).reshape(-1, nout)
        worst, first_bad, compared = 0.0, None, 0
        missing = [m for m in modules if m not in ref_outs]
        if missing:
            gates.append(("layer", False, f"no reference module (or it never ran): {missing[:5]}"))
        for row in rows:
            path = row["path"]
            ref = ref_outs.get(path)
            if not path or ref is None:
                continue
            got = np.fromfile(dump / row["file"], dtype=np.float32).reshape(row["rows"], -1)
            if got.shape != ref.shape and got.shape[0] == 1:
                ref = ref[-1:]  # a last-row step compares with the reference's last row
            if got.shape != ref.shape:
                gates.append(
                    ("layer", False, f"{path}: native {got.shape} vs reference {ref.shape}")
                )
                first_bad = first_bad or path
                break
            rel = _rel(got, ref)
            compared += 1
            if rel > worst:
                worst = rel
            if rel > STEP_RTOL and first_bad is None:
                first_bad = f"{row['block']}.{row['layer']}.{row['step']} ({path}) rel {rel:.2e}"
        if not any(g[0] == "layer" for g in gates):
            ok = compared > 0 and first_bad is None
            why = (f"{compared} step outputs, worst rel {worst:.2e} (limit {STEP_RTOL:g})"
                   + (f"; first over: {first_bad}" if first_bad else ""))  # fmt: skip
            gates.append(("layer", ok, why if compared else "no step output matched a module"))

    rel = _rel(native_logits[0], ref_logits)
    top_n = set(np.argsort(-native_logits[0])[:10].tolist())
    top_r = set(np.argsort(-ref_logits)[:10].tolist())
    same = int(np.argmax(native_logits[0])) == int(np.argmax(ref_logits))
    gates.append(("logits", same and rel <= LOGITS_RTOL,
                  f"rel {rel:.2e}, argmax {'equal' if same else 'differs'}, top-10 overlap "
                  f"{len(top_n & top_r)}/10"))  # fmt: skip

    full = _engine(engine, model_dir, ids, new, env, ["--recompute"])
    gates.append(("decode", full["tokens"] == run["tokens"],
                  f"incremental {run['tokens']} vs recompute {full['tokens']}"))  # fmt: skip
    first = next(
        (i for i, (a, b) in enumerate(zip(run["tokens"], ref_tokens, strict=False)) if a != b), None
    )
    gates.append(("tokens", run["tokens"] == ref_tokens,
                  f"{len(ref_tokens)} greedy tokens" + ("" if first is None else
                  f"; first differs at {first}: native {run['tokens'][first]} vs "
                  f"reference {ref_tokens[first]}")))  # fmt: skip
    return {
        "model_dir": str(model_dir), "engine": str(engine),
        "ids": ids, "new": new, "tokens": run["tokens"], "reference_tokens": ref_tokens,
        "timing": {k: run.get(k) for k in ("load_ms", "prefill_ms", "decode_ms_per_token")},
        "gates": [{"level": g, "pass": ok, "evidence": e} for g, ok, e in gates],
        "verdict": "PASS" if all(ok for _, ok, _ in gates) else "FAIL",
    }  # fmt: skip


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")


def prompt_ids(model_dir: Path, prompt: str | None, vocab: int, n: int = 12) -> list[int]:
    """The prompt in the model's own tokenizer, when the directory has one and it gives a real
    prompt (a surrogate has none: a tokenizer conjured for it maps the text to one unknown id);
    else ``n`` distinct ids spread over the vocab."""
    if prompt and any((model_dir / f).is_file() for f in _TOKENIZER_FILES):
        try:
            from transformers import AutoTokenizer

            ids = AutoTokenizer.from_pretrained(str(model_dir)).encode(prompt)
            if len(ids) >= 4 and len(set(ids)) > 1 and all(0 <= i < vocab for i in ids):
                return ids
        except (OSError, ValueError, TypeError):
            pass
    return [(7919 * j + 13) % vocab for j in range(n)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m xe_forge.frontend.verify")
    p.add_argument(
        "--model-dir", type=Path, required=True, help="config.json + safetensors, or a GGUF file"
    )
    p.add_argument("--engine", type=Path, required=True, help="the assembled xf-run")
    p.add_argument("--prompt", help="text; tokenized with the model's tokenizer when it has one")
    p.add_argument("--new", type=int, default=8, help="greedy tokens to generate")
    p.add_argument("--threads", type=int, default=0, help="OMP threads (default: OMP_NUM_THREADS)")
    p.add_argument("--numa", type=int, help="bind the engine to this NUMA node (numactl)")
    p.add_argument(
        "--reference", type=Path, help="reference adapter (load) instead of transformers"
    )
    p.add_argument("-o", "--output", type=Path, required=True)
    a = p.parse_args(argv)
    from xe_forge.frontend.surrogate import merged_config

    config = merged_config(a.model_dir)
    ids = prompt_ids(a.model_dir, a.prompt, int(config["vocab_size"]))
    env = dict(os.environ)
    if a.threads:
        env["OMP_NUM_THREADS"] = str(a.threads)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    if a.numa is not None:
        env["XF_NUMA"] = str(a.numa)
    result = verify(a.model_dir, a.engine, ids, a.new, env, reference=a.reference)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    for g in result["gates"]:
        print(f"GATE {g['level']} {'PASS' if g['pass'] else 'REJECT'} {g['evidence']}")
    print(f"TIMING {json.dumps(result['timing'])}")
    print(f"VERDICT: {result['verdict']}")
    print("DONE")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
