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
sequence, on the GPU in --precision, bf16 unless the run declares another) -- under patch.json's "reference_python" (the
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
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from xe_forge.frontend import profile
model_id, batch, input_len, n, trust = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5] == "1"
got = json.load(open(sys.argv[6]))
prompt_file, dtype, draws = sys.argv[7] or None, sys.argv[8], int(sys.argv[10])
rows = [int(i) for i in sys.argv[9].split(",")] if sys.argv[9] else range(min(len(got), batch))
dev = profile.device_type()
if dev == "cpu":
    sys.exit("the reference interpreter sees no GPU")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
m = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), trust_remote_code=trust)
try:
    m = m.to(dev).eval()
except RuntimeError as e:  # float32 that does not fit the GPU runs on the CPU
    if dtype != "float32" or "out of memory" not in str(e).lower():
        raise
    m, dev = m.to("cpu").eval(), "cpu"
prompts = profile.token_ids(profile.hf_vocab(tok), batch, input_len, prompt_file)
seed = [None]  # draw k > 0: the embeddings times (1 + 1e-6 * N(0, 1)), seeded k
def perturb(mod, inp, out):
    if seed[0] is None:
        return out
    z = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed[0]))
    return out * (1 + 1e-6 * z.to(out.device, out.dtype))
m.get_input_embeddings().register_forward_hook(perturb)
gaps = {}
for i in rows:
    p, g = prompts[i], got[i][:n]  # one row, the whole sequence's logits: no batch, no slice
    gaps[i] = []
    for k in range(draws):
        seed[0] = k or None
        with torch.no_grad():
            x = torch.tensor([p + g], device=dev)
            lp = m(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
        best = lp.max(-1).values
        gaps[i].append((best - lp[torch.arange(len(g), device=dev),
                                  torch.tensor(g, device=dev)]).tolist())
print(json.dumps({"gaps": gaps, "device": dev,
                  "transformers": [transformers.__version__, transformers.__file__]}))
"""


def _native(work: Path, python: str, out: Path) -> tuple[subprocess.CompletedProcess, float]:
    """generate.sh again, with the framework's own plain-torch path instead of the new op.

    ``XE_FORGE_NATIVE=1`` asks generate.sh for vLLM's ``custom_ops: ["none"]`` and the torch
    attention backend: the same build, weights, seed and precision, with the patched kernels
    unused. So the comparison differs in one thing only -- who runs the op -- which is what a
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


def _by_native(work: Path, python: str, patched: Path, record: dict) -> list[str]:
    """Judge an image or video patch against the framework's own torch path for the same op.

    Writes ``quality.json`` beside the patch: the artifacts generate.sh saved (the prompt
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
        judged = quality.quality(work, {"A": [Path(nat)], "B": [patched]},
                                 {"A": [Path(nat)], "B": [patched]})  # fmt: skip
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
    if problems:
        return problems, record
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
            return [
                f"generate.sh failed (exit {run.returncode}): {run.stderr.strip()[-400:]}"
            ], record
        if not got_path.is_file():
            # No tokens: an image or a video. The reference is then the framework's own
            # plain-torch path for the same op, not transformers -- there is nothing to
            # teacher-force, but there is something to compare, and the op gain to measure.
            if getattr(args, "no_reference", False):
                record.update(reference="skipped (--no-reference)")
                return problems, record
            return _by_native(work, python, Path(out), record), record
        if args.input_len is None:  # tokens are scored on the prompt they continue
            return ["generate.sh wrote tokens.json: pass --input-len to score them"], record
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
            return problems, record

        def score(dtype: str, rows: list[int] | None = None, draws: int = 1):
            """Teacher forcing on the candidate's own tokens, on the GPU."""
            return subprocess.run([
                meta.get("reference_python") or args.reference_python or args.framework_python,
                "-c", _REFERENCE, args.model, str(args.batch), str(args.input_len), str(n),
                "1" if args.trust_remote_code or meta.get("trust_remote_code") else "0",
                str(got_path), prompt, dtype,
                ",".join(map(str, rows)) if rows else "", str(draws)],
                capture_output=True, text=True, env=_env())  # fmt: skip

        ref = score(args.precision)
        if ref.returncode != 0:
            return [
                f"the transformers reference failed ({ref.stderr.strip()[-400:]}); patch.json's "
                '"reference_python" (GPU torch and a transformers that loads the model) and '
                '"trust_remote_code" set it up'
            ], record
        res = json.loads(ref.stdout.strip().splitlines()[-1])
        first = [res["gaps"][k][0] for k in sorted(res["gaps"], key=int)]
        gaps, judge = list(first), [args.precision] * len(first)
        # A sequence the first pass fails: float32 judges all its positions, over DRAWS forwards.
        again = [i for i, g in enumerate(first)
                 if i < len(got) and len(got[i]) >= n and max(g, default=0) > tol]  # fmt: skip
        fp32 = None
        if again:
            ref32 = score("float32", again, DRAWS)
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
    worst = [round(max(r, default=float("inf")), 4) for r in gaps]
    record.update(tokens=got, gaps=gaps, worst=worst, judge=judge, judge_first=args.precision, first_gaps=first, fp32=fp32,
                  tolerance=tol, required=n, transformers=res.get("transformers"),
                  prompt_file=prompt)  # fmt: skip
    if len(gaps) != len(got) or any(len(g) < n for g in got) or any(w > tol for w in worst):
        problems.append(f"tokens fall more than {tol} nat below transformers' best within the "
                        f"first {n}: worst per sequence {worst} ({len(got)} of {args.batch})")  # fmt: skip
    return problems, record


def run(args) -> int:
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
        "check": "FAIL" if problems else "PASS",
    }  # fmt: skip
    if record.get("head") and not measured_pr(work / "PR.md", facts):
        problems.append("no PR.md: write the pull-request description beside patch.json")
    record["pass"] = not problems
    (work / "check.json").write_text(json.dumps(record, indent=1) + "\n")
    for p in problems:
        print(f"patch check: FAIL {p}")
    if not problems:
        # The message must only describe checks that actually ran. Reading
        # record['tolerance'] unconditionally raised KeyError on the --no-reference path --
        # after the patch had PASSED -- and the caller read the crash as a failed gate.
        # Three models' worth of working XPU ops (Z-Image, Qwen-Image, Wan2.2) were
        # discarded by this line reporting a success.
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
        for k in (record.get("fp32") or {}).get("gaps") or {}:
            print(
                f"patch check: sequence {k} judged in float32 on {record['fp32']['device']} "
                f"({record['judge_first']} {max(record['first_gaps'][int(k)]):.4f} -> {record['worst'][int(k)]}; "
                f"{record['fp32']['excused'][k]} tokens within the reference's own spread)"
            )
    return 0 if not problems else 1
