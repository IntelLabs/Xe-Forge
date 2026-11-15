"""``patch check``: the gate on a patch an agent wrote to make a framework run a model (an
architecture or op it lacks). The agent writes, builds and debugs the patch; this decides.

<dir>/patch.json: {"python": <the patched stack's interpreter>, "repo": <clone under --repos,
patch committed on top of "base">, "base": <commit>, "package": <the module it patches>}.
<dir>/generate.sh: runs the model under $FRAMEWORK_PYTHON, greedy, on
``xe_forge.frontend.profile.token_ids``, and writes $OUT/tokens.json ([[generated ids], ...]).

Checks, in order: the patch is git's own diff of base..HEAD and applies to a fresh clone at
base; the patched package imports from outside the framework's venv; then each of generate.sh's
first --min-match tokens of every sequence is within --tolerance nat of the best token of a
Hugging Face transformers reference fed the same prefix (teacher forcing: one forward per
sequence, on the GPU in bf16) -- under patch.json's "reference_python" (the
agent's, when the framework's transformers cannot load the model), else --reference-python,
else the framework's interpreter. The agent owns that environment; the scoring is fixed here.
Exit 0 passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

_REFERENCE = """
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from xe_forge.frontend import profile
model_id, batch, input_len, n, trust = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5] == "1"
got = json.load(open(sys.argv[6]))
dev = profile.device_type()
if dev == "cpu":
    sys.exit("the reference interpreter sees no GPU")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=trust)
m = m.to(dev).eval()
gaps = []
for p, g in zip(profile.token_ids(profile.hf_vocab(tok), batch, input_len), got):
    g = g[:n]  # one row, the whole sequence's logits: no batch, no last-token slice
    with torch.no_grad():
        x = torch.tensor([p + g], device=dev)
        lp = m(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
    best = lp.max(-1).values
    gaps.append((best - lp[torch.arange(len(g), device=dev), torch.tensor(g, device=dev)]).tolist())
print(json.dumps({"gaps": gaps, "transformers": [transformers.__version__, transformers.__file__]}))
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
                f"the patched output does not match the framework's own torch path: "
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
    problems = []
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
    with tempfile.TemporaryDirectory() as out:
        t0 = time.perf_counter()
        run = subprocess.run(["bash", str(work / "generate.sh")], cwd=work,
                             env=_env(FRAMEWORK_PYTHON=python, OUT=out),
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
        # Teacher forcing on the candidate's own tokens, on the GPU in bf16.
        ref = subprocess.run([meta.get("reference_python") or args.reference_python
                              or args.framework_python, "-c", _REFERENCE, args.model,
                              str(args.batch), str(args.input_len), str(args.min_match),
                              "1" if args.trust_remote_code or meta.get("trust_remote_code") else "0",
                              str(got_path)], capture_output=True, text=True, env=_env())  # fmt: skip
    if ref.returncode != 0:
        return [
            f"the transformers reference failed ({ref.stderr.strip()[-400:]}); patch.json's "
            '"reference_python" (GPU torch and a transformers that loads the model) and '
            '"trust_remote_code" set it up'
        ], record
    res = json.loads(ref.stdout.strip().splitlines()[-1])
    gaps, n, tol = res["gaps"], args.min_match, args.tolerance
    worst = [round(max(r, default=float("inf")), 4) for r in gaps]
    record.update(tokens=got, gaps=gaps, worst=worst, tolerance=tol, required=n,
                  transformers=res.get("transformers"))  # fmt: skip
    if len(gaps) != len(got) or any(len(g) < n for g in got) or any(w > tol for w in worst):
        problems.append(f"tokens fall more than {tol} nat below transformers' best within the "
                        f"first {n}: worst per sequence {worst} ({len(got)} of {args.batch})")  # fmt: skip
    return problems, record


def run(args) -> int:
    problems, record = check(args)
    record["pass"] = not problems
    work = Path(args.dir).resolve()
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
            how = (f"output matches {record['reference']}"
                   + (f"; op gain x{record['op_gain']}" if record.get("op_gain") else ""))
        else:
            how = ("the model runs under the patch; its output was NOT checked "
                   f"({record.get('reference', 'reference skipped')})")
        print(f"patch check: PASS {record['repo']} {record['base'][:12]}..{record['head'][:12]} "
              f"({len(record['files'])} files); {how} -> {record['patch']}")  # fmt: skip
    return 0 if not problems else 1
