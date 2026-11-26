"""The optional fusion step, after the kernels are optimized: an agent proposes chains of ops
one kernel could replace, ``fuse check`` gates and ranks them, and each one passing goes
through the same manifest, inputs, workspace and benchmark gate as a kernel -- its baseline
the chain, on repositories that already carry the kernels' winning patches, so a fusion is
measured against the optimized parts and its patch applies on top of theirs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from x_forge.frontend.analyze import Analysis, Entry


def _git(*args: str, env: dict | None = None) -> str:
    full = None if env is None else {**os.environ, **env}
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                          env=full).stdout  # fmt: skip


def origin(patch: Path) -> tuple[str, str]:
    """(repository, base commit) the PORT.txt beside a patch names for it."""
    text = (patch.parent / "PORT.txt").read_text()
    block = text.split(f"output/{patch.name}:", 1)[-1]
    repo = re.search(r"repository:[ \t]*(\S.*)", block)
    base = re.search(r"base commit:\s*(\S+)", block)
    if not (repo and base):
        raise ValueError(
            f"{patch.parent / 'PORT.txt'} names no repository and base for {patch.name}"
        )
    return repo.group(1).strip(), base.group(1)


def stack(patches: list[Path], out: Path) -> dict[str, str]:
    """{origin: clone}: each repository the patches touch, cloned at their base commit with
    them applied in order and committed. STACK.json records what each clone is made of."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    made: dict[str, dict] = {}
    for p in patches:
        repo, base = origin(p)
        if repo not in made:
            clone = out / f"{len(made)}-{Path(repo).name}"  # repositories may share a basename
            _git("clone", "--quiet", "--shared", "--no-checkout", repo, str(clone))
            _git("-C", str(clone), "checkout", "--quiet", "--detach", base)
            made[repo] = {"clone": str(clone), "origin": repo, "base": base, "patches": []}
        elif made[repo]["base"] != base:
            raise ValueError(
                f"{p}: base {base}, but earlier patches to {repo} are on {made[repo]['base']}"
            )
        clone = made[repo]["clone"]
        _git("-C", clone, "apply", "--index", str(p))
        # Who and when fixed by the base: the same base and patches rebuild the same commit,
        # so a clone that borrowed it from an earlier build still finds it.
        when = _git("-C", clone, "show", "-s", "--format=%cI", base).strip()
        who = {"GIT_AUTHOR_NAME": "x-forge", "GIT_AUTHOR_EMAIL": "x-forge@localhost",
               "GIT_COMMITTER_NAME": "x-forge", "GIT_COMMITTER_EMAIL": "x-forge@localhost",
               "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}  # fmt: skip
        _git("-C", clone, "-c", "commit.gpgsign=false", "commit", "--quiet", "--no-verify",
             "-m", f"x-forge: {p}", env=who)  # fmt: skip
        made[repo]["patches"].append(str(p))
    for m in made.values():
        m["commit"] = _git("-C", m["clone"], "rev-parse", "HEAD").strip()
    (out / "STACK.json").write_text(json.dumps(list(made.values()), indent=1) + "\n")
    return {repo: m["clone"] for repo, m in made.items()}


def analysis(passed: list[dict], capture, base: Path, top: int, min_gain: float) -> Analysis:
    """The passing fusions as manifest entries on their anchor's workload, the costliest
    ``top`` of those worth ``min_gain`` enabled."""
    owner = {i: w for w in capture.workloads for i in w.ops}
    entries = []
    for rank, f in enumerate(passed, 1):
        e = Entry(f["name"], str((base / f["repo"]).resolve()), f["dsl"], [owner[f["anchor"]]], [])
        e.enabled = rank <= top and f["share"] >= min_gain
        e.priority = f["share"]
        e.reason = (
            f"fusion of {len(f['ops'])} ops, {f['share']:.2%} of device time: {f.get('why', '')}"
        )
        if not e.enabled:
            e.reason += f"; rank {rank}, beyond --fuse-top {top} or below --min-gain {min_gain}"
        e.extra = {
            "reference": str((base / f["reference"]).resolve()),
            "integration_repos": [
                str((base / r).resolve()) for r in f.get("integration_repos") or []
            ],
            "ops": f["ops"],
        }
        entries.append(e)
    run = capture.run
    return Analysis(
        run.get("id", ""), run.get("total_device_us", 0.0), entries, {}, {"fuse_top": top}
    )
