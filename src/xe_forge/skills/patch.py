"""``patch check``: the gate on a patch an agent wrote to make a framework run a model (an
architecture or op it lacks). The agent writes, builds and debugs the patch; this decides.

<dir>/patch.json: {"python": <the patched stack's interpreter>, "repo": <clone under --repos,
patch committed on top of "base">, "base": <commit>, "package": <the module it patches>}.
<dir>/generate.sh: runs the model under $FRAMEWORK_PYTHON, greedy, on
``xe_forge.frontend.profile.token_ids``, and writes $OUT/tokens.json ([[generated ids], ...]).

Checks, in order: the patch is git's own diff of base..HEAD and applies to a fresh clone at
base; the patched package imports from outside the framework's venv; generate.sh's tokens
equal a Hugging Face transformers reference -- computed here, after them, on the GPU in bf16,
under the framework's interpreter or --reference-python -- for the first --min-match tokens of
every sequence. Exit 0 passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

_REFERENCE = """
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xe_forge.frontend import profile
model_id, batch, input_len, output_len, trust = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5] == "1"
dev = profile.device_type()
if dev == "cpu":
    sys.exit("the reference interpreter sees no GPU")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
ids = torch.tensor(profile.token_ids(profile.hf_vocab(tok), batch, input_len), device=dev)
m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=trust).to(dev)
out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=output_len,
                 min_new_tokens=output_len, do_sample=False)
print(json.dumps(out[:, input_len:].tolist()))
"""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _env(**extra) -> dict:
    src = str(Path(__file__).resolve().parents[2])  # this package, for profile.token_ids
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_"))}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    return {**env, **extra}


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
        run = subprocess.run(["bash", str(work / "generate.sh")], cwd=work,
                             env=_env(FRAMEWORK_PYTHON=python, OUT=out),
                             capture_output=True, text=True)  # fmt: skip
        got_path = Path(out) / "tokens.json"
        if run.returncode != 0 or not got_path.is_file():
            return [
                f"generate.sh failed (exit {run.returncode}): {run.stderr.strip()[-400:]}"
            ], record
        got = json.loads(got_path.read_text())
    # On the GPU, in bf16 -- the framework's interpreter unless it has no XPU torch.
    ref = subprocess.run([args.reference_python or args.framework_python, "-c", _REFERENCE,
                          args.model, str(args.batch),
                          str(args.input_len), str(args.output_len),
                          "1" if args.trust_remote_code else "0"],
                         capture_output=True, text=True, env=_env())  # fmt: skip
    if ref.returncode != 0:
        return [
            f"the transformers reference failed ({ref.stderr.strip()[-400:]}); a "
            "--reference-python with GPU torch and transformers runs it"
        ], record
    want = json.loads(ref.stdout.strip().splitlines()[-1])
    n = args.min_match
    matched = [next((i for i, (a, b) in enumerate(zip(g, w, strict=False)) if a != b), min(len(g), len(w)))
               for g, w in zip(got, want, strict=False)]  # fmt: skip
    record.update(reference=want, tokens=got, matched=matched, required=n)
    if len(got) != len(want) or any(m < n for m in matched):
        problems.append(f"greedy tokens differ from transformers within the first {n}: "
                        f"matched {matched} of {len(want)} sequences")  # fmt: skip
    return problems, record


def run(args) -> int:
    problems, record = check(args)
    record["pass"] = not problems
    work = Path(args.dir).resolve()
    (work / "check.json").write_text(json.dumps(record, indent=1) + "\n")
    for p in problems:
        print(f"patch check: FAIL {p}")
    if not problems:
        print(f"patch check: PASS {record['repo']} {record['base'][:12]}..{record['head'][:12]} "
              f"({len(record['files'])} files); tokens match transformers ({record['matched']}) "
              f"-> {record['patch']}")  # fmt: skip
    return 0 if not problems else 1
