"""``patch check``: the gate on a patch an agent wrote to make a framework run a model (an
architecture or op it lacks). The agent writes, builds and debugs the patch; this decides.

<dir>/patch.json: {"python": <the patched stack's interpreter>, "repo": <clone under --repos,
patch committed on top of "base">, "base": <commit>, "package": <the module it patches>}.
<dir>/generate.sh: runs the model under $FRAMEWORK_PYTHON, greedy, on
``xe_forge.frontend.profile.token_ids(vocab, $BATCH, $INPUT_LEN, $PROMPT_FILE)``, generating
$OUTPUT_LEN tokens, and writes $OUT/tokens.json ([[generated ids], ...]). The check sets those:
the prompt is natural text (the run's --prompt-file, else PROMPT below), never the run's
synthetic ids -- on those every next-token distribution is flat, so near-ties decide tokens and
a correct quantized patch fails at random; OUTPUT_LEN is the tokens it scores.

Checks, in order: the patch is git's own diff of base..HEAD and applies to a fresh clone at
base; the patched package imports from outside the framework's venv; then each of generate.sh's
first --min-match tokens of every sequence is within --tolerance nat of the best token of a
Hugging Face transformers reference fed the same prefix (teacher forcing: one forward per
sequence, on the run's device -- the GPU, or the host when the run is on it -- in --precision,
bf16 unless the run declares another) -- under patch.json's "reference_python" (the
agent's, when the framework's transformers cannot load the model), else --reference-python,
else the framework's interpreter. A sequence that fails there is judged again, all its positions,
in float32 (on the GPU, or the CPU when it does not fit there), over DRAWS forwards: one
clean, the others with the embeddings perturbed by a seeded 1e-6 relative noise. A model
that quantizes activations on the fly is ill-conditioned (rounding steps turn that noise into
nats at some positions), so any two correct implementations disagree there: a token fails
when its smallest gap over the draws exceeds the tolerance plus their spread, and a sequence
where more than one token in eight passes only within that spread is too unstable to judge,
a FAIL. A stable model has no spread: the tolerance alone. If the float32 pass fails, the first
verdict stands. The agent owns that environment; the scoring and precision are fixed here.
generate.sh runs the checkpoint at its own precision: the gate judges the port. With
--quantization (the run's, when it declares one), generate.sh runs once more with
XE_FORGE_QUANTIZATION set and the same reference scores those tokens too; that is recorded
in check.json's "quantized" and PR.md, a fact for whoever chose the quantization, never a
failure -- unless --quantized-gate dequantized: a framework that quantizes the weights itself
names what they decode to, and its quantized tokens are gated against the original model
holding those (scored in float32), the same tokens against the checkpoint reported beside.
Exit 0 passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

PROMPT = Path(__file__).with_name("patch_prompt.txt")  # plain English prose, any tokenizer
DRAWS = 8  # float32 forwards of a sequence the first pass fails: one clean, the rest perturbed

_REFERENCE = """
import json, os, sys, torch
from xe_forge.frontend import profile
model_id, batch, input_len, n, trust = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5] == "1"
got = json.load(open(sys.argv[6]))
prompt_file, dtype, draws = sys.argv[7] or None, sys.argv[8], int(sys.argv[10])
rows = [int(i) for i in sys.argv[9].split(",")] if sys.argv[9] else range(min(len(got), batch))
dev = profile.device_type()
if dev == "cpu" and os.environ.get("XE_FORGE_DEVICE", "").split(":")[0] != "cpu":
    sys.exit("the reference interpreter sees no GPU")
ref_path = sys.argv[11] if len(sys.argv) > 11 else ""
if ref_path:  # the agent's runner: load(model, dtype, device) -> (model, tokenizer or Vocab)
    import importlib.util
    spec = importlib.util.spec_from_file_location("xf_reference", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m, tok = mod.load(model_id, getattr(torch, dtype), dev)
    m, source = m.eval(), [ref_path, ""]
else:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), trust_remote_code=trust)
    source = [transformers.__version__, transformers.__file__]
    try:
        m = m.to(dev).eval()
    except RuntimeError as e:  # float32 that does not fit the GPU runs on the CPU
        if dtype != "float32" or "out of memory" not in str(e).lower():
            raise
        m, dev = m.to("cpu").eval(), "cpu"
export = sys.argv[12] if len(sys.argv) > 12 else ""
facts = None
if export:  # the framework's dequantized weights in place of the checkpoint's: each must be one of its tensors
    import glob
    from safetensors.torch import load_file
    sd, errs, bad = m.state_dict(), {}, []
    for f in sorted(glob.glob(os.path.join(export, "*.safetensors"))):
        for k, v in load_file(f).items():
            if k not in sd:
                bad.append(f"{k}: not a tensor of the model")
            elif tuple(v.shape) != tuple(sd[k].shape):
                bad.append(f"{k}: shape {list(v.shape)}, the model's {list(sd[k].shape)}")
            elif not torch.isfinite(v).all():
                bad.append(f"{k}: not finite")
            else:
                o = sd[k].float().cpu()
                errs[k] = float((v.float() - o).norm() / max(float(o.norm()), 1e-30))
                sd[k] = v.to(sd[k].device, sd[k].dtype)
    bad += [] if errs else [f"{export} holds no tensor"]
    bad += [f"{k}: relative error {e:.3g} against the checkpoint, further from it than zeros"
            for k, e in errs.items() if e > 1]
    if bad:
        print(json.dumps({"export_problems": bad}))
        sys.exit(3)
    m.load_state_dict(sd, strict=True)
    worst = sorted(errs.items(), key=lambda kv: -kv[1])[:5]
    facts = {"tensors": len(errs), "mean_rel_error": sum(errs.values()) / len(errs),
             "worst": [[k, round(e, 6)] for k, e in worst]}
vocab = tok if hasattr(tok, "ordinary") else profile.hf_vocab(tok)
prompts = profile.token_ids(vocab, batch, input_len, prompt_file)
def logits(x):  # transformers' output, a tuple's first item, or the tensor itself
    out = m(input_ids=x, use_cache=False) if not ref_path else m(x)
    out = getattr(out, "logits", out)
    return out[0] if isinstance(out, (tuple, list)) else out
seed = [None]  # draw k > 0: the embeddings times (1 + 1e-6 * N(0, 1)), seeded k
def perturb(mod, inp, out):
    if seed[0] is None:
        return out
    z = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed[0]))
    return out * (1 + 1e-6 * z.to(out.device, out.dtype))
embed = getattr(m, "get_input_embeddings", None)
if embed is not None:
    embed().register_forward_hook(perturb)
else:  # nothing to perturb: one clean forward, no spread
    draws = 1
gaps = {}
for i in rows:
    p, g = prompts[i], got[i][:n]  # one row, the whole sequence's logits: no batch, no slice
    gaps[i] = []
    for k in range(draws):
        seed[0] = k or None
        with torch.no_grad():
            x = torch.tensor([p + g], device=dev)
            lp = logits(x)[0, len(p) - 1:-1].float().log_softmax(-1)
        best = lp.max(-1).values
        gaps[i].append((best - lp[torch.arange(len(g), device=dev),
                                  torch.tensor(g, device=dev)]).tolist())
print(json.dumps({"gaps": gaps, "device": dev, "transformers": source, "export": facts}))
"""


def _native(work: Path, python: str, out: Path) -> tuple[subprocess.CompletedProcess, float]:
    """generate.sh again, with the framework's own plain-torch path instead of the new op.

    ``XE_FORGE_NATIVE=1`` asks generate.sh for the framework's own plain path for the op (the
    switch it offers, which the patch agent finds): the same build, weights, seed and
    precision, with the patched kernels unused. So the comparison differs in one thing only -- who runs the op -- which is what a
    reference has to do. The two wall times are the patch's worth: the op gain.
    """
    t0 = time.perf_counter()
    run = subprocess.run(["bash", str(work / "generate.sh")], cwd=work, capture_output=True,
                         text=True, env=_env(FRAMEWORK_PYTHON=python, OUT=str(out),
                                             XE_FORGE_NATIVE="1"))  # fmt: skip
    return run, time.perf_counter() - t0


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _env(**extra) -> dict:
    src = str(Path(__file__).resolve().parents[2])  # this package, for profile.token_ids
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    return {**env, **extra}


def _by_native(work: Path, python: str, patched: Path, record: dict,
               margin: float | None = None, into: Path | None = None) -> list[str]:  # fmt: skip
    """Judge an image or video patch against the framework's own torch path for the same op.

    Writes ``quality.json`` into ``into``, else beside the patch: the artifacts generate.sh saved (the prompt
    embedding, the step-1 and final latents, the picture or the frames), stage by stage. The
    gate is the earliest stage present, since the sampling loop amplifies a difference the
    further down you look. ``op_gain`` is the whole pipeline, native over patched.
    """
    from xe_forge.frontend import quality

    with tempfile.TemporaryDirectory() as nat:
        native, native_s = _native(work, python, Path(nat))
        record["native_s"] = round(native_s, 1)
        if native.returncode != 0:
            return [
                "the native run (XE_FORGE_NATIVE=1: the framework's plain-torch path for this "
                f"op) failed, so the patch has no reference: {native.stderr.strip()[-400:]}. "
                "generate.sh must honour XE_FORGE_NATIVE"
            ]
        if record.get("patched_s"):
            record["op_gain"] = round(native_s / record["patched_s"], 3)
        judged = quality.quality(into or work, {"A": [Path(nat)], "B": [patched]},
                                 {"A": [Path(nat)], "B": [patched]},
                                 margin=quality.MARGIN if margin is None else margin)  # fmt: skip
        record["reference"] = f"the framework's plain-torch path ({judged['verdict']})"
        record["quality"] = judged
        if judged["verdict"] == "absent":
            return [
                "generate.sh saved nothing the check can compare. Write the prompt embedding, "
                "the step-1 and final latents and the picture (or the frames) into $OUT, with "
                "the names the patch prompt lists"
            ]
        if judged["verdict"] in ("fail", "inconclusive"):
            return [
                "the patched output does not match the framework's own torch path: "
                + "; ".join(
                    f"{s['stage']} {s['verdict']} ({s['why']})"
                    for s in judged["stages"]
                    if s["verdict"] in ("fail", "inconclusive")
                )
            ]
    return []


def _quantized(work: Path, python: str, quant: str, prompt: str, n: int, args, score) -> dict:
    """generate.sh again with XE_FORGE_QUANTIZATION, scored by the same reference: per
    sequence the gaps, the worst, and the share of tokens that are the reference's best. With
    ``--quantized-gate dequantized`` the reference is the original model holding the
    dequantized export generate.sh names in $OUT/dequantized.json, scored in float32 (the
    export's precision), and ``export`` says how far that export is from the checkpoint."""
    gate = getattr(args, "quantized_gate", None) == "dequantized"
    with tempfile.TemporaryDirectory() as out:
        env = _env(FRAMEWORK_PYTHON=python, OUT=out, BATCH=str(args.batch), OUTPUT_LEN=str(n),
                   PROMPT_FILE=prompt, XE_FORGE_QUANTIZATION=quant,
                   **({} if args.input_len is None else {"INPUT_LEN": str(args.input_len)}))  # fmt: skip
        t0 = time.perf_counter()
        gen = subprocess.run(["bash", str(work / "generate.sh")], cwd=work, env=env,
                             capture_output=True, text=True)  # fmt: skip
        rec = {"quantization": quant, "run_s": round(time.perf_counter() - t0, 1), "gated": gate}
        tokens = Path(out) / "tokens.json"
        if gen.returncode != 0 or not tokens.is_file():
            return {**rec, "error": f"generate.sh with XE_FORGE_QUANTIZATION={quant} failed "
                    f"(exit {gen.returncode}): {gen.stderr.strip()[-400:]}"}  # fmt: skip
        got = json.loads(tokens.read_text())
        export = ""
        if gate:
            named = Path(out) / "dequantized.json"
            if not named.is_file():
                return {**rec, "tokens": got, "error": "generate.sh wrote no $OUT/dequantized.json "
                        "naming the weights its quantized run decodes to"}  # fmt: skip
            told = json.loads(named.read_text())
            if not isinstance(told, dict) or not told.get("weights"):
                return {**rec, "tokens": got, "error": '$OUT/dequantized.json is not '
                        '{"weights": "<directory>"}'}  # fmt: skip
            export = str((work / told["weights"]).resolve())
            rec["export_dir"] = export
        ref = score("float32" if gate else args.precision, tokens=tokens, export=export)
        if ref.returncode != 0:
            last = (ref.stdout.strip().splitlines() or [""])[-1]
            told = json.loads(last).get("export_problems") if last.startswith("{") else None
            return {
                **rec,
                "tokens": got,
                "error": ("the dequantized export does not fit the model: " + "; ".join(told[:5]))
                if told
                else f"the reference failed: {ref.stderr.strip()[-400:]}",
            }
        rec.update(tokens=got, export=json.loads(ref.stdout.strip().splitlines()[-1]).get("export"),
                   **_gap_facts(ref))  # fmt: skip
        if gate:  # what the person who chose the quantization judges: the same tokens against
            # the checkpoint itself, reported
            orig = score(args.precision, tokens=tokens)
            rec["against_checkpoint"] = (_gap_facts(orig) if orig.returncode == 0 else
                                         {"error": f"the reference failed: {orig.stderr.strip()[-300:]}"})  # fmt: skip
        return rec


def _gap_facts(ref: subprocess.CompletedProcess) -> dict:
    """Per sequence the gaps, the worst, the mean and the share of the reference's best, from a
    reference's output."""
    res = json.loads(ref.stdout.strip().splitlines()[-1])
    gaps = [g[0] for _, g in sorted(res["gaps"].items(), key=lambda kv: int(kv[0]))]
    flat = [x for g in gaps for x in g]
    return {"gaps": gaps, "worst": [round(max(g, default=float("inf")), 4) for g in gaps],
            "mean_gap": round(sum(flat) / len(flat), 4) if flat else None,
            "best_share": round(sum(x == 0 for x in flat) / len(flat), 4) if flat else None}  # fmt: skip


def _quantized_gate(q: dict, args, tol: float) -> list[str]:
    """The problems of a gated quantized run (``--quantized-gate dequantized``); none when the
    quantized run is only reported."""
    if not q.get("gated"):
        return []
    if q.get("error"):
        return [f"{q['quantization']}: {q['error']}"]
    n = args.min_match
    if (
        len(q["gaps"]) != len(q["tokens"])
        or any(len(g) < n for g in q["tokens"])
        or any(w > tol for w in q["worst"])
    ):
        return [f"{q['quantization']}: tokens fall more than {tol} nat below the original model "
                f"holding the dequantized export, within the first {n}: worst per sequence "
                f"{q['worst']}"]  # fmt: skip
    return []


def check(args) -> tuple[list[str], dict]:
    """(problems, record) of the patch in ``args.dir``."""
    work = Path(args.dir).resolve()
    meta = json.loads((work / "patch.json").read_text())
    repo = Path(args.repos) / str(meta.get("repo", ""))
    base, python, package = meta.get("base", ""), meta.get("python", ""), meta.get("package", "")
    record = {"repo": str(repo), "base": base}
    if not (repo / ".git").exists():
        return [f"repo {repo} is not a git clone under {args.repos}"], record
    if _git(repo, "merge-base", "--is-ancestor", base, "HEAD").returncode != 0:
        return [f"base {base!r} is not an ancestor of {repo}'s HEAD"], record
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    diff = _git(repo, "diff", "--binary", f"{base}..{head}").stdout
    if not diff.strip():
        return [f"no change between base {base} and HEAD {head}"], record
    patch = work / f"{repo.name}.patch"
    patch.write_text(diff)  # git's diff, not the agent's
    record.update(head=head, patch=str(patch), url=_git(repo, "remote", "get-url", "origin").stdout.strip(),
                  files=_git(repo, "diff", "--name-only", f"{base}..{head}").stdout.split())  # fmt: skip
    from xe_forge.skills.upstream import commit_problems, write_mbox

    problems = commit_problems(repo, base)
    write_mbox(repo, base, patch.with_suffix(".mbox"))
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "fresh"
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), str(fresh)],
                       check=False)  # fmt: skip
        _git(fresh, "checkout", "-q", base)
        if _git(fresh, "apply", "--check", str(patch)).returncode != 0:
            problems.append(f"{patch.name} does not apply to a fresh clone at {base}")
    prefix = subprocess.run([args.framework_python, "-c", "import sys; print(sys.prefix)"],
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    where = subprocess.run([python, "-c", f"import {package}; print({package}.__file__)"],
                           capture_output=True, text=True, env=_env())  # fmt: skip
    record["package_file"] = where.stdout.strip()
    if where.returncode != 0 or not where.stdout.strip():
        problems.append(f"{python} cannot import {package!r}: {where.stderr.strip()[-300:]}")
    elif prefix and where.stdout.strip().startswith(prefix):
        problems.append(f"{package} imports from the framework's venv ({prefix}), not the patch")
    if getattr(args, "require_build", False):
        script = work / str(meta.get("build") or "")
        if not meta.get("build") or not script.is_file():
            problems.append('patch.json names no "build" script beside it: the run builds trials '
                            "of this repository with it")  # fmt: skip
        elif not all(v in script.read_text() for v in ("$SRC", "$VENV")):
            problems.append(f"{script.name} does not read $SRC and $VENV: it must build any tree "
                            "of the repository into any venv")  # fmt: skip
    if problems:
        return problems, record
    return score(work, meta, python, args, record), record


def score(work: Path, meta: dict, python: str, args, record: dict,
          into: Path | None = None) -> list[str]:  # fmt: skip
    """generate.sh under ``python``, its tokens scored by the reference: the problems, with
    ``record`` filled. ``check`` runs it on the patch's own interpreter once the patch holds;
    ``patch score`` runs it on another build of the same stack, so a change made after the
    check is judged on the tokens the check judged. ``into``: where an image or video
    comparison writes its record, else beside the patch. ``record["unscored"]``: the reference
    could not score, which says nothing of the tokens."""
    problems: list[str] = []
    # The candidate first, then the reference: generate.sh never sees what it is checked against.
    n, tol, prompt = args.min_match, args.tolerance, str(args.prompt_file or PROMPT)
    with tempfile.TemporaryDirectory() as out:
        t0 = time.perf_counter()
        run = subprocess.run(["bash", str(work / "generate.sh")], cwd=work,
                             env=_env(FRAMEWORK_PYTHON=python, OUT=out, BATCH=str(args.batch),
                                      OUTPUT_LEN=str(n), PROMPT_FILE=prompt,
                                      **({} if args.input_len is None
                                         else {"INPUT_LEN": str(args.input_len)})),
                             capture_output=True, text=True)  # fmt: skip
        record["patched_s"] = round(time.perf_counter() - t0, 1)
        got_path = Path(out) / "tokens.json"
        if run.returncode != 0:
            return [f"generate.sh failed (exit {run.returncode}): {run.stderr.strip()[-400:]}"]
        if not got_path.is_file():
            # No tokens: an image or a video. The reference is then the framework's own
            # plain-torch path for the same op, not transformers -- there is nothing to
            # teacher-force, but there is something to compare, and the op gain to measure.
            if getattr(args, "no_reference", False):
                record.update(reference="skipped (--no-reference)")
                return problems
            return _by_native(work, python, Path(out), record, getattr(args, "margin", None), into)
        if args.input_len is None:  # tokens are scored on the prompt they continue
            return ["generate.sh wrote tokens.json: pass --input-len to score them"]
        got = json.loads(got_path.read_text())
        if getattr(args, "no_reference", False):
            # Everything above still had to hold: the patch is git's own diff, it applies
            # to a fresh clone at its base commit, the patched package imports from the
            # patch and not the framework's venv, and generate.sh ran the model to
            # completion under it. What is skipped is only the comparison against a
            # transformers reference -- there is no token to score in an image or a video.
            # So this says the patch builds and the model runs with it, and says NOTHING
            # about whether the output is right. Judge that yourself.
            record.update(output=got, reference="skipped (--no-reference)")
            return problems

        # The model's own reference runner, when transformers cannot load it (patch.json).
        runner = str((work / meta["reference"]).resolve()) if meta.get("reference") else ""

        def teacher(dtype: str, rows: list[int] | None = None, draws: int = 1, tokens=None,
                    export: str = ""):  # fmt: skip
            """Teacher forcing on the candidate's own tokens, on the run's device; ``export``:
            a directory of dequantized weights the reference model holds in place of its own."""
            return subprocess.run([
                meta.get("reference_python") or args.reference_python or args.framework_python,
                "-c", _REFERENCE, args.model, str(args.batch), str(args.input_len), str(n),
                "1" if args.trust_remote_code or meta.get("trust_remote_code") else "0",
                str(tokens or got_path), prompt, dtype,
                ",".join(map(str, rows)) if rows else "", str(draws), runner, export],
                capture_output=True, text=True, env=_env())  # fmt: skip

        ref = teacher(args.precision)
        if ref.returncode != 0:
            record["unscored"] = True
            return [
                f"the reference failed ({ref.stderr.strip()[-400:]}); patch.json's "
                '"reference_python" (GPU torch and what loads the model), "trust_remote_code", '
                'or "reference" (a runner with load(model, dtype, device)) set it up'
            ]
        res = json.loads(ref.stdout.strip().splitlines()[-1])
        first = [res["gaps"][k][0] for k in sorted(res["gaps"], key=int)]
        gaps, judge = list(first), [args.precision] * len(first)
        # A sequence the first pass fails: float32 judges all its positions, over DRAWS forwards.
        again = [i for i, g in enumerate(first)
                 if i < len(got) and len(got[i]) >= n and max(g, default=0) > tol]  # fmt: skip
        fp32 = None
        if again:
            ref32 = teacher("float32", again, DRAWS)
            fp32 = {
                "rows": again,
                "error": ref32.stderr.strip()[-400:] if ref32.returncode else None,
            }
            if not ref32.returncode:
                res32 = json.loads(ref32.stdout.strip().splitlines()[-1])
                fp32.update(device=res32["device"], gaps=res32["gaps"], excused={})
                for k, runs in res32["gaps"].items():
                    # per token: its smallest gap over the draws, and their spread
                    ls = [(min(c), max(c) - min(c)) for c in zip(*runs, strict=True)]
                    gaps[int(k)] = [max(0.0, lo - s) for lo, s in ls]
                    judge[int(k)] = "float32"
                    fp32["excused"][k] = sum(lo > tol >= lo - s for lo, s in ls)
                    if fp32["excused"][k] * 8 > len(ls):
                        problems.append(
                            f"sequence {k}: the reference is too unstable to judge -- "
                            f"{fp32['excused'][k]} of {len(ls)} tokens pass only within its own "
                            f"spread over {DRAWS} perturbed float32 forwards")  # fmt: skip
        quant = getattr(args, "quantization", None)
        if quant and quant != "auto" and not problems:
            record["quantized"] = _quantized(work, python, quant, prompt, n, args, teacher)
            problems += _quantized_gate(record["quantized"], args, tol)
    worst = [round(max(r, default=float("inf")), 4) for r in gaps]
    record.update(tokens=got, gaps=gaps, worst=worst, judge=judge, judge_first=args.precision, first_gaps=first, fp32=fp32,
                  tolerance=tol, required=n, transformers=res.get("transformers"),
                  prompt_file=prompt)  # fmt: skip
    if len(gaps) != len(got) or any(len(g) < n for g in got) or any(w > tol for w in worst):
        problems.append(f"tokens fall more than {tol} nat below transformers' best within the "
                        f"first {n}: worst per sequence {worst} ({len(got)} of {args.batch})")  # fmt: skip
    return problems


def _quantized_facts(q: dict | None) -> dict:
    if not q:
        return {}
    name = q["quantization"]
    if q.get("error"):
        return {f"{name}: scored": f"no ({q['error'][:200]})"}
    facts = {f"{name}: worst gap per sequence (nat)": q["worst"],
             f"{name}: mean gap (nat)": q["mean_gap"],
             f"{name}: share of tokens the reference's best": q["best_share"]}  # fmt: skip
    if q.get("export"):  # gated against its own dequantized weights; the format's loss, reported
        e = q["export"]
        facts[f"{name}: judged against"] = "the original model holding its dequantized weights"
        facts[f"{name}: mean relative error of those weights"] = round(e["mean_rel_error"], 6)
        facts[f"{name}: worst tensors"] = ", ".join(f"{k} {v}" for k, v in e["worst"])
        c = q.get("against_checkpoint") or {}
        facts[f"{name}: against the checkpoint, worst gap per sequence (nat)"] = (
            c.get("worst") if not c.get("error") else f"not scored ({c['error'][:200]})"
        )
        facts[f"{name}: against the checkpoint, share of tokens its best"] = c.get("best_share")
    return facts


UNSCORED = 2  # patch score: nothing was judged (the reference, or generate.sh's contract)


def rescore(args) -> int:
    """``patch score``: the scoring of ``check`` alone, generate.sh under ``--python`` (another
    build of the patched stack), the record written to ``--out``. The patch's own files,
    check.json, PR.md and quality.json, are left as its check wrote them. Exit 1: the output
    fails; UNSCORED: it could not be judged."""
    work, out = Path(args.dir).resolve(), Path(args.out).resolve()
    meta = json.loads((work / "patch.json").read_text())
    record = {"python": args.python, "dir": str(work)}
    package = meta.get("package", "")
    where = subprocess.run([args.python, "-c", f"import {package}; print({package}.__file__)"],
                           capture_output=True, text=True, env=_env())  # fmt: skip
    record["package_file"] = where.stdout.strip()  # which build generate.sh is given
    if "FRAMEWORK_PYTHON" not in (work / "generate.sh").read_text():
        # A script that names its interpreter itself runs the build it names, never this one.
        problems, record["unscored"] = ["generate.sh does not run under $FRAMEWORK_PYTHON"], True
    elif where.returncode != 0 or not record["package_file"]:
        problems = [f"{args.python} cannot import {package!r}: {where.stderr.strip()[-300:]}"]
    else:
        out.with_suffix("").mkdir(parents=True, exist_ok=True)
        problems = score(work, meta, args.python, args, record, out.with_suffix(""))
    record["pass"] = not problems
    out.write_text(json.dumps(record, indent=1) + "\n")
    for p in problems:
        print(f"patch score: FAIL {p}")
    if problems and record.get("unscored"):
        return UNSCORED
    if not problems:
        how = (f"tokens within {record['tolerance']} nat of the reference (worst "
               f"{max(record['worst'], default=0)})" if "tolerance" in record
               else record.get("reference") or "output compared")  # fmt: skip
        print(f"patch score: PASS {args.python}: {how} -> {args.out}")
    return 0 if not problems else 1


def run(args) -> int:
    if args.patch_command == "score":
        return rescore(args)
    from xe_forge.skills.upstream import measured_pr

    problems, record = check(args)
    work = Path(args.dir).resolve()
    quality = record.get("quality") or {}
    facts = {
        "repository": record.get("url"), "base": record.get("base"), "head": record.get("head"),
        "files": ", ".join(record.get("files") or []), "reference": record.get("reference"),
        "tolerance (nat)": record.get("tolerance"), "tokens checked per sequence": record.get("required"),
        "worst gap per sequence (nat)": record.get("worst"), "judged in": record.get("judge"),
        "transformers": record.get("transformers"), "quality": quality.get("verdict"),
        "patched run (s)": record.get("patched_s"), "native run (s)": record.get("native_s"),
        "op gain (native / patched)": record.get("op_gain"),
        **_quantized_facts(record.get("quantized")),
        "check": "FAIL" if problems else "PASS",
    }  # fmt: skip
    if record.get("head") and not measured_pr(work / "PR.md", facts):
        problems.append("no PR.md: write the pull-request description beside patch.json")
    record["pass"] = not problems
    (work / "check.json").write_text(json.dumps(record, indent=1) + "\n")
    for p in problems:
        print(f"patch check: FAIL {p}")
    if not problems:
        # The message must only describe checks that actually ran: a record may have no
        # tolerance (the --no-reference path).
        if "tolerance" in record:
            how = (f"tokens within {record['tolerance']} nat of transformers "
                   f"(worst {max(record['worst'])})")  # fmt: skip
        elif record.get("quality"):
            how = f"output matches {record['reference']}" + (
                f"; op gain x{record['op_gain']}" if record.get("op_gain") else ""
            )
        else:
            how = (
                "the model runs under the patch; its output was NOT checked "
                f"({record.get('reference', 'reference skipped')})"
            )
        print(f"patch check: PASS {record['repo']} {record['base'][:12]}..{record['head'][:12]} "
              f"({len(record['files'])} files); {how} -> {record['patch']}")  # fmt: skip
        q = record.get("quantized")
        if q:
            how = "gated on its dequantized weights" if q.get("gated") else "reported, not gated"
            print(f"patch check: {q['quantization']} ({how}): "
                  + (q["error"] if q.get("error") else
                     f"worst gap per sequence {q['worst']} nat, mean {q['mean_gap']}, "
                     f"{q['best_share']:.0%} of tokens the reference's best"))  # fmt: skip
        for k in (record.get("fp32") or {}).get("gaps") or {}:
            print(
                f"patch check: sequence {k} judged in float32 on {record['fp32']['device']} "
                f"({record['judge_first']} {max(record['first_gaps'][int(k)]):.4f} -> {record['worst'][int(k)]}; "
                f"{record['fp32']['excused'][k]} tokens within the reference's own spread)"
            )
    return 0 if not problems else 1
