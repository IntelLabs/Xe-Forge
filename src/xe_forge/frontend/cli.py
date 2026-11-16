"""The framework frontend's commands: run, optimize (an existing kernels.yaml again), calibrate.

Anything after ``--`` is passed unchanged to every Xe-Forge run ``optimize`` starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import xe_forge

COMMANDS = ("optimize", "run", "calibrate")
SRC_DIR = Path(xe_forge.__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent


# ── capture ────────────────────────────────────────────────────────────────


def _tool(python: str, *argv: str, problems: list | None = None, **env: str) -> int:
    """An ``xe-forge-skill`` tool under ``python``, with this package on its path; the PROBLEM
    lines it prints are added to ``problems`` when given."""
    full = dict(os.environ, **env)
    full["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), full.get("PYTHONPATH")) if p)
    cmd = [python, "-m", "xe_forge.skills", *argv]
    print(f"run: {shlex.join(cmd)}", flush=True)
    if problems is None:
        return subprocess.call(cmd, env=full)
    proc = subprocess.run(cmd, env=full, stdout=subprocess.PIPE, text=True)
    print(proc.stdout, end="", flush=True)
    problems += [
        ln.split(": PROBLEM ", 1)[1] for ln in proc.stdout.splitlines() if ": PROBLEM " in ln
    ]
    return proc.returncode


def _shapes(ops, top: int | None = 12) -> list[dict]:
    """The op's distinct shapes, most device time first: identical ones merged, so a shape
    that runs rarely (prefill next to many decode steps) is never crowded out."""
    from xe_forge.frontend.canonical import _named_dims as named_dims

    merged: dict[str, dict] = {}
    for o in ops:
        if not o.calls:
            continue
        dims, dtypes = named_dims(o), [a.dtype if a else None for a in o.args]
        m = merged.setdefault(json.dumps([dims, dtypes], sort_keys=True), {"dims": dims,
                              "calls": 0, "us": 0.0, "kernels": {}, "dtype": next((d for d in dtypes
                              if d and d.startswith(("bfloat", "float"))), None)})  # fmt: skip
        m["calls"] += o.calls
        m["us"] += o.device_us
        for k, t in o.kernel_symbols.items():
            m["kernels"][k] = m["kernels"].get(k, 0.0) + t
    return [
        {"dims": m["dims"], "calls": m["calls"], "us_per_call": round(m["us"] / m["calls"], 3),
         "kernels_per_call": {k: round(t / m["calls"], 3) for k, t in m["kernels"].items()},
         "dtype": m["dtype"]}
        for m in sorted(merged.values(), key=lambda m: -m["us"])[:top]
    ]  # fmt: skip


def _agent(step: str, template: str, context: dict, cwd: Path, answer: str, python, turns):
    """One agent step; None when it gave no answer (the reason is printed)."""
    from xe_forge.frontend import agent

    try:
        return agent.run(template, context, cwd, answer, python or sys.executable, turns)
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        return None


# ── the workload ───────────────────────────────────────────────────────────
# What one measured run is. A text decoder is a batch of sequences, a prefill and some
# decode steps; a vision model is some frames at a resolution over some denoising steps --
# one frame being a still image, so Flux and LTX are the same kind.
# Neither set of numbers means anything to the other, so each kind names its own and the
# capture template is told which it is.
#
# Adding a kind: add its fields here and say what they mean in frontend-capture.md.j2.
# Nothing downstream of the capture needs to change -- analyze, inputs, optimize and
# port-back work on ops, shapes and kernels, and the axis a shape varies along is whatever
# the naming agent declares (``token_arg``), not something this file decides.
WORKLOADS = {
    "text": ("batch", "input_len", "output_len"),
    "vision": (
        "batch",
        "steps",
        "height",
        "width",
        "frames",
        "input_images",
        "prompt",
        "negative_prompt",
    ),
}
# The prompt a vision run uses when none is given: one fixed text, so captures of the same
# model on two frameworks, or two days, run the same work.
VISION_PROMPT = "A red fox standing in fresh snow at the edge of a pine forest, morning light"


def _add_workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workload", default="text", choices=sorted(WORKLOADS),
                   help="what one measured run is: a text decoder's prefill and decode "
                   "steps, or a vision model's denoising steps at a resolution")  # fmt: skip
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--quantization", default="auto",
                   help="how the weights are quantized: auto (the checkpoint's own), the "
                   "framework's name for a scheme (fp8, awq, ...), a stored dtype (int8), or "
                   "<scheme>:<dtype> (awq:int4) -- the dtype is what the capture checks the "
                   "GEMMs read. The capture agent never picks this: it "
                   "changes the model's output quality, which only you can judge, and it "
                   "changes which kernels run -- so optimizing a quantized run and serving "
                   "an unquantized one tunes kernels you never execute")  # fmt: skip
    # text
    p.add_argument("--input-len", type=int, default=512, help="prompt tokens [text]")
    p.add_argument("--output-len", type=int, default=32, help="tokens generated [text]")
    # vision
    p.add_argument("--steps", type=int, default=28, help="denoising steps [vision]")
    p.add_argument("--height", type=int, default=1024, help="image height [vision]")
    p.add_argument("--width", type=int, default=1024, help="image width [vision]")
    p.add_argument("--input-image", action="append", default=[], metavar="PATH",
                   help="an image the model takes as input [vision]: the one being edited, "
                   "or the first frame of a video. Repeatable -- some editing models take "
                   "several references. Its size decides the shapes the model runs at, so "
                   "it is part of what a capture is: a capture made with one image is not "
                   "a capture made with another")  # fmt: skip
    p.add_argument("--frames", type=int, default=1,
                   help="frames to generate [vision]; 1 is a still image. A video model "
                   "may round this up -- LTX wants 8k+1 -- so record what actually ran")  # fmt: skip
    p.add_argument("--prompt", help=f"the text prompt [vision], default {VISION_PROMPT!r}. "
                   "Part of what a capture is, as the input images are")  # fmt: skip
    p.add_argument("--negative-prompt", help="the negative prompt [vision], for a pipeline "
                   "that takes one")  # fmt: skip


def workload(a) -> dict:
    """The fields of ``a.workload``'s kind, and nothing else.

    Passing every field of every kind would be worse than useless: the capture agent would
    be told a text model generates 28 denoising steps, and the reuse check below would
    compare an output_len that no vision capture ever recorded.
    """
    out = {}
    for k in WORKLOADS[a.workload]:
        v = getattr(a, k, None)
        if k == "input_images":
            # Resolved and sorted: the same images given in another order, or by a relative
            # path, are the same work. Empty becomes None so a capture that never recorded
            # the field still matches -- the trap quantization already taught us.
            paths = getattr(a, "input_image", []) or []
            v = sorted(str(Path(x).resolve()) for x in paths) or None
        out[k] = v
    return out


def _model_id(name: str) -> str:
    """A model name reduced to what identifies the model, for comparison only.

    A Hugging Face cache path carries the repo id in a ``models--Org--Name`` component and
    ends in a commit hash, so neither the whole string nor its last component is the name:
    ``.../hub/models--Qwen--Qwen3.5-9B/snapshots/c202236...`` is ``Qwen/Qwen3.5-9B``. A
    plain directory (a Flux snapshot) has only its own name, which is the best available.
    """
    name = name.rstrip("/")
    parts = Path(name).parts
    for part in reversed(parts):
        if part.startswith("models--"):
            return part[len("models--") :].replace("--", "/").lower()
    # Only a filesystem path gets reduced to its last component. "Qwen/Qwen3.5-9B" is a
    # repo id, not a path, and cutting it to "Qwen3.5-9B" would lose the organisation.
    if name.startswith(("/", "./", "~")):
        return (parts[-1] if parts else name).lower()
    return name.lower()


def _image_paths(value) -> list[str] | None:
    """The image paths in a recorded or requested ``input_images``, resolved and sorted.

    A launcher is right to record more than a path -- a FLUX editing capture recorded
    ``{"path", "size": [4544, 2805], "mode": "RGB", "note": "the pipeline resizes to
    <=1024*1024 area"}``, which is exactly the kind of thing a reader wants. Comparing the
    whole structure refused that capture after it had already succeeded. The path is what
    identifies the input; the rest is description.
    """
    if not value:
        return None
    out = []
    for item in value if isinstance(value, list) else [value]:
        path = item.get("path") if isinstance(item, dict) else item
        if path:
            out.append(str(Path(str(path)).resolve()))
    return sorted(out) or None


def _same_workload(want: dict, have: dict) -> bool:
    """Whether a capture is of the workload asked for, comparing each field as it means."""
    for key in want:
        if key == "input_images":
            if _image_paths(want[key]) != _image_paths(have.get(key)):
                return False
        elif want[key] != have.get(key):
            return False
    return True


def _same_model(want: str, have: str | None) -> bool:
    """Whether two model names mean the same model.

    They often differ without disagreeing: pass a local snapshot directory and the capture
    agent records the canonical id it resolved to. A 9B run was refused for exactly that,
    after a good capture -- "capture of {'model': 'Qwen/Qwen3.5-9B'}, not
    {'model': '/data/.../models--Qwen--Qwen3.5-9B/snapshots/c202236...'}". A local Flux
    directory has the same shape.
    """
    if not have:
        return False
    # Deliberately strict about the organisation: comparing only the last segment would
    # make "Qwen/Qwen3.5-9B" and "meta-llama/Qwen3.5-9B" the same model. A wrong refusal
    # costs one re-capture; a wrong match optimizes against someone else's capture and says
    # nothing about it.
    return want == have or _model_id(want) == _model_id(have)


def _collect_outputs(run_dir: Path) -> None:
    """Gather what the model produced into ``outputs/``, for a person to look at.

    A vision patch and a vision kernel are both kept on measurements that say nothing about
    the picture, so the picture has to be easy to find. The launcher and each prove arm
    write theirs wherever suits them, which is no use for comparing before against after.

    Only the places a model writes are searched. Walking the run directory instead pulled in
    135 files, nearly all of them documentation images out of cloned kernel repositories --
    CUTLASS logos and oneDNN diagrams -- which is worse than collecting nothing.
    """
    out = run_dir / "outputs"
    places = [run_dir, run_dir / "raw"]
    places += sorted(run_dir.glob("prove/*"))
    places += sorted(run_dir.glob("prove/*/raw"))
    places += sorted(run_dir.glob("results/*/output"))
    seen: set[str] = set()
    for place in places:
        if not place.is_dir():
            continue
        for src in sorted(place.iterdir()):
            if src.suffix.lower() not in {".png", ".jpg", ".jpeg", ".mp4", ".wav"}:
                continue
            if not src.is_file() or src.stat().st_size < 1024:
                continue
            prefix = "" if place == run_dir else f"{place.parent.name}-{place.name}-"
            name = (prefix + src.name).replace(f"{run_dir.name}-", "")
            if name in seen:
                continue
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / name)
            seen.add(name)
    if seen:
        print(
            f"run: {len(seen)} generated output(s) in {out} -- look at these before using "
            "any patch from this run"
        )


def capture(
    a, python: str, run_dir: Path, turns: int, repos: Path, refused: list | None = None
) -> tuple[int, str]:
    """(exit code, the interpreter the run continues under).

    The capture agent writes and runs a launcher; the tool's own verdict decides. When the
    framework lacks the model's architecture or an op, a patch agent establishes the patch
    first and capture runs again on the patched stack (``stack.json``), a few rounds at most.
    ``refused``: the problems the named read found in this launcher's last capture.
    """
    from xe_forge.frontend import CACHE

    # runtime_keys: the launcher must record exactly the fields the reuse check reads back,
    # or a good capture is refused for a field it never wrote.
    context = {"model": a.model, "framework": a.framework, "cache": CACHE,
               "workload": a.workload, "runtime_keys": WORKLOADS[a.workload],
               "framework_notes": getattr(a, "framework_note", []) or [],
               **workload(a), "inputs": getattr(a, "inputs", None) or {},
               "precision": getattr(a, "precision", "auto"),
               "quantization": getattr(a, "quantization", "auto"),
               "kv_cache": getattr(a, "kv_cache", "auto"), "refused": refused or []}  # fmt: skip
    for round_ in range(1, 4):
        got = _agent("capture", "frontend-capture.md.j2", context, run_dir, "captured.json",
                     python, turns)  # fmt: skip
        if got is None:
            return 1, python
        if got.get("fits") is False:
            print(f"capture: does not fit -- {got.get('reason', '')}", file=sys.stderr)
            return 3, python
        if got.get("unsupported") == "precision":  # a format this device has no kernel for
            print(f"capture: {a.framework} cannot run {got.get('what')} on this device -- "
                  f"{got.get('reason', '')}", file=sys.stderr)  # fmt: skip
            return 3, python
        if "unsupported" not in got:
            # What the agent reports is not the verdict: the tool checks raw/ again here.
            raw, out = str(run_dir / "raw"), str(run_dir / "capture.json")
            return _tool(python, "capture", "parse", raw, "-o", out), python
        print(f"capture: {a.framework} lacks {got['unsupported']} {got.get('what')} "
              f"({got.get('reason')}); a patch first", flush=True)  # fmt: skip
        work = run_dir / f"patch-{round_}"
        ctx = {**context, "what": got.get("what"), "reason": got.get("reason"), "repos": repos}
        if (
            _agent("patch", "frontend-patch.md.j2", ctx, work, "patched.json", python, turns)
            is None
        ):
            return 1, python
        check = ["patch", "check", str(work), "--repos", str(repos), "--framework-python",
                 python, "--model", a.model, "--batch", str(a.batch), "--input-len",
                 str(getattr(a, "input_len", 0))]  # fmt: skip
        # An image or a video has no token to teacher-force, but it still has a reference:
        # `patch check` runs generate.sh a second time with XE_FORGE_NATIVE=1 -- the
        # framework's own plain-torch path for the same op -- and compares the two, stage by
        # stage, as well as timing both for the op gain. Pass --no-reference to skip it.
        if getattr(a, "reference_python", None):  # the framework's own has no GPU transformers
            check += ["--reference-python", a.reference_python]
        if "prompts" in context["inputs"]:
            check += ["--prompt-file", context["inputs"]["prompts"]["file"]]
        if str(context["precision"]).startswith(("float", "bfloat")):  # declared, not auto
            check += ["--precision", context["precision"]]
        if _tool(python, *check) != 0:  # the gate is the tool's, not the agent's word
            print(f"capture: the patch in {work} does not pass its check", file=sys.stderr)
            return 1, python
        python = json.loads((work / "patch.json").read_text())["python"]
        stack = run_dir / "stack.json"
        rounds = json.loads(stack.read_text())["patches"] if stack.is_file() else []
        stack.write_text(json.dumps({"python": python, "patches": [*rounds, str(work)]},
                                    indent=1) + "\n")  # fmt: skip
    print("capture: still unsupported after 3 patches", file=sys.stderr)
    return 1, python


def name_and_read(a, cap, run_dir: Path, python: str, turns: int, repos: Path) -> tuple[int, str]:
    """Name the capture's ops, then re-read it with the names: the tool's verdict decides. A
    capture the named read refuses goes back once to the capture agent -- the namer saw those
    problems (bound) and cannot change what was captured -- and its new capture is named again.
    """
    from xe_forge.frontend import ir

    capture_path, names = run_dir / "capture.json", run_dir / "names.json"
    for attempt in (1, 2):
        if names.is_file() and json.loads(names.read_text()).get("capture") != _stamp(cap):
            print(f"run: {names} was written for another capture; naming again", file=sys.stderr)
            names.unlink()
        if not names.is_file():
            # A capture.json a previous names.json annotated would show the namer its own old
            # answers (an op's "gemm" as given) and it would leave them out: name a clean one.
            if any("named" in o.framework_meta for o in cap.ops):
                if _tool(python, "capture", "parse", str(run_dir / "raw"), "-o", str(capture_path)):
                    return 1, python
                cap = ir.load(capture_path)
            name(capture_path, run_dir, python, turns, repos, a.name_coverage, a.calibration,
                 cap=cap)  # fmt: skip
            if not names.is_file():
                print("run: no names.json; the ops stay unnamed", file=sys.stderr)
                return 0, python
        problems: list = []
        parse = ["capture", "parse", str(run_dir / "raw"), "-o", str(capture_path)]
        if _tool(python, *parse, "--names", str(names), problems=problems) == 0:
            return 0, python
        if attempt == 2 or not problems:
            break
        print("run: the named capture is refused; back to the capture agent", file=sys.stderr)
        if (run_dir / "session.log").is_file():  # the refused capture's, kept
            (run_dir / "session.log").replace(run_dir / "session-refused.log")
        code, python = capture(a, python, run_dir, turns, repos, refused=problems)
        names.unlink()  # its stamp does not see a changed launcher: name the new capture
        if code != 0:
            return (code if code > 0 else 1), python
        cap = ir.load(capture_path)
    print("run: re-reading the capture with names.json failed", file=sys.stderr)
    return 1, python


def capture_rows(a, run_dir: Path, python: str) -> tuple[Path | None, str | None]:
    """The run's other rows: the launcher rerun at each, no agent, every capture read with
    the anchor's names.json, then merged (``rows.merge``). (The merged capture, else None;
    the launcher's problem for the capture agent when it did not run at a row's size.)"""
    from xe_forge.frontend import ir, rows
    from xe_forge.skills.capture import _sha256

    launcher, names = run_dir / "launcher" / "run.sh", run_dir / "names.json"
    caps = [ir.load(run_dir / "capture.json")]
    for i, row in enumerate(a.rows[1:], 2):
        out = run_dir / "rows" / f"{i:02d}"
        raw, path = out / "raw", out / "capture.json"
        parse = ["capture", "parse", str(raw), "-o", str(path)]
        parse += ["--names", str(names)] if names.is_file() else []
        # a row this launcher captured at this size is kept; read again, names.json may be new
        run = json.loads((raw / "run.json").read_text()) if (raw / "run.json").is_file() else {}
        rt = run.get("runtime") or {}
        if run.get("launcher_sha256") != _sha256(launcher) or any(
            rt.get(k) != row[k] for k in rows.KEYS
        ):
            env = {f"XF_{k.upper()}": str(row[k]) for k in rows.KEYS}
            if _tool(sys.executable, "capture", "run", str(launcher), "-o", str(raw),
                     FRAMEWORK_PYTHON=python, **env):  # fmt: skip
                print(f"run: row {i} {row} did not capture", file=sys.stderr)
                return None, None
        if _tool(python, *parse):  # the interpreter the anchor was read under: its schemas
            print(f"run: row {i} {row}: its capture is refused", file=sys.stderr)
            return None, None
        caps.append(ir.load(path))
        if not rows.ran(caps[-1], row):
            got = {k: (caps[-1].run.get("runtime") or {}).get(k) for k in rows.KEYS}
            env = " ".join(f"XF_{k.upper()}={row[k]}" for k in rows.KEYS)
            return None, (f"run with {env} set, the launcher recorded runtime {got}: it must "
                          "run at those sizes and record them")  # fmt: skip
    merged = rows.merge(caps, a.rows)
    return ir.dump(merged, run_dir / "rows" / "capture.json"), None


def recapture(run_dir: Path, out_dir: Path, python: str) -> int:
    """Re-run the cached launcher under ``python``'s venv, then parse; no agent. The tools
    run here: the stack's interpreter need not import them."""
    from xe_forge.frontend.prove import PARSE_REFUSED

    launcher = str(run_dir / "launcher" / "run.sh")
    code = _tool(sys.executable, "capture", "run", launcher, "-o", str(out_dir / "raw"),
                 FRAMEWORK_PYTHON=python)  # fmt: skip
    if code:
        return code
    parse = ["capture", "parse", str(out_dir / "raw"), "-o", str(out_dir / "capture.json")]
    return PARSE_REFUSED if _tool(sys.executable, *parse) else 0


def name(
    capture_path: Path,
    run_dir: Path,
    python,
    max_turns: int,
    repos: Path,
    coverage: float,
    calibration: Path | None = None,
    cap=None,
) -> Path | None:
    """Run the naming agent over the costliest ops; the written names.json, or None."""
    from xe_forge.frontend import ir, naming
    from xe_forge.frontend.canonical import _named_dims as named_dims
    from xe_forge.frontend.canonical import arg_label

    cap = cap or ir.load(capture_path)
    work = run_dir / "name"
    work.mkdir(parents=True, exist_ok=True)
    total = sum(op.device_us for op in cap.ops) or 1.0
    by_key: dict[str, list] = {}
    for op in cap.ops:
        by_key.setdefault(op.framework_meta.get("name_key", op.framework_op), []).append(op)
    ranked = sorted(by_key.items(), key=lambda kv: -sum(o.device_us for o in kv[1]))
    listing, covered = {}, 0.0
    for key, ops in ranked:
        if covered >= coverage:
            break
        us = sum(o.device_us for o in ops)
        covered += us / total
        top = max(ops, key=lambda o: o.device_us)
        kernels: dict[str, float] = {}
        for o in ops:
            for sym, k_us in o.kernel_symbols.items():
                kernels[sym] = round(kernels.get(sym, 0.0) + k_us, 3)
        listing[key] = {
            "framework_op": top.framework_op,
            "owner": top.framework_meta.get("owner"),
            "share": round(us / total, 4),
            "calls": sum(o.calls for o in ops),
            "kernels": dict(sorted(kernels.items(), key=lambda kv: -kv[1])),
            "args": [
                {
                    "name": arg_label(top, i),
                    **(
                        {"shape": a.shape, "dtype": a.dtype}
                        if a
                        else {"value": top.scalars[i] if i < len(top.scalars) else None}
                    ),
                }
                for i, a in enumerate(top.args)
            ],
            "call_site": top.call_site,
            **({"gemm": top.framework_meta["gemm"]} if top.framework_meta.get("gemm") else {}),
            "dims": named_dims(top),
            "shapes": _shapes(ops),  # what the namer reads; its costs are checked on all
        }
    (work / "ops.json").write_text(json.dumps(listing, indent=1) + "\n")
    model = cap.run.get("model") or {}
    (work / "config.json").write_text(json.dumps(model.get("config") or {}, indent=1) + "\n")
    from xe_forge.frontend import calibrate
    from xe_forge.models import DSL

    calib = json.loads(calibration.read_text()) if calibration else None
    context = {
        "model": model.get("name"),
        "framework": cap.run.get("framework"),
        "coverage": f"{covered:.0%}",
        "repos": repos,
        "routes": naming.ROUTES,
        "rows": naming.ROWS,
        "dsls": [d.value for d in DSL],
        "calibration": calib and {**calib, "matmul_tflops": calibrate.peaks(calib)[1]},
        "bound_cmd": shlex.join(
            [
                "xe-forge-skill",
                "bound",
                str(capture_path),
                "--names",
                str(work / "names.json"),
                "--ops",
                str(work / "ops.json"),
                "--repos",
                str(repos),
            ]
            + (["--calibration", str(calibration)] if calibration else [])
        ),
    }
    answer = _agent("name", "frontend-name.md.j2", context, work, "names.json", python, max_turns)
    if answer is None:
        return None
    answer, problems = naming.keep_valid(answer, listing, repos)
    if problems:
        what = "not usable" if answer is None else "entries left unnamed or fields dropped"
        print(f"name: {what} ({len(problems)} problems):", file=sys.stderr)
        for line in problems[:20]:
            print(f"  {line}", file=sys.stderr)
    if answer is None:
        return None
    # A stated cost is a lower bound, so one above a measured call is wrong.
    from xe_forge.frontend.analyze import NOISE

    every = {k: {"shapes": _shapes(by_key[k], top=None)} for k in listing}
    wrong = naming.refuted(answer, every, calib, NOISE)
    for k, sub, why in wrong:
        print(f"name: cost dropped for {k}{f' [{sub}]' if sub else ''}: {why}", file=sys.stderr)
    naming.drop_costs(answer, wrong)
    missing = [k for k in listing if k not in answer["ops"]]
    if missing:
        print(f"name: {len(missing)} ops left unnamed: {', '.join(missing[:5])}", file=sys.stderr)
    answer["capture"] = _stamp(cap)
    path = run_dir / "names.json"
    path.write_text(json.dumps(answer, indent=1) + "\n")
    return path


def _stamp(cap) -> dict:
    """The capture's model, framework and runtime, to tie a names.json to it."""
    run = cap.run
    return {
        "model": (run.get("model") or {}).get("name"),
        "framework": run.get("framework"),
        "runtime": {k: v for k, v in (run.get("runtime") or {}).items() if not isinstance(v, dict)},
    }


# ── analyze ────────────────────────────────────────────────────────────────


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--top", type=int, default=3, help="kernels to optimize")
    p.add_argument("--min-gain", type=float, default=0.005, help="smallest share worth it")
    p.add_argument("--calibration", type=Path, help="xe-forge calibrate output")


def analyze(capture_path: Path, out_dir: Path, a: argparse.Namespace, weights=None) -> Path:
    from xe_forge.frontend import analyze as an
    from xe_forge.frontend import ir, manifest

    cap = ir.load(capture_path)
    calib = json.loads(a.calibration.read_text()) if a.calibration else None
    result = an.analyze(cap, top=a.top, min_gain=a.min_gain, calibration=calib)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result.as_dict(), indent=1) + "\n")
    path = manifest.emit(result, cap, capture_path, out_dir, weights)
    enabled = [e for e in result.entries if e.enabled]
    if cap.run.get("counters"):
        print(f"analyze: counters cover {result.options['counters_pct']}% of device time")
    for d in result.options["unpeaked"]:
        print(f"analyze: {d} has no matmul peak in the calibration; bounded at the fastest")
    print(
        f"analyze: {len(cap.workloads)} workloads -> {len(result.entries)} kernels, "
        f"{len(enabled)} enabled ({', '.join(e.name for e in enabled) or 'none'}) -> {path}"
    )
    return path


def inputs(manifest_path: Path, run_dir: Path, a, python, turns, repos) -> dict:
    """{kernel: {shape: weight}} for each enabled kernel whose trial shapes the inputs
    agent chose and ``inputs check`` passed; the others are not optimized."""
    from xe_forge.frontend import manifest
    from xe_forge.skills import inputs as tool

    m = manifest.load(manifest_path)
    kernels = [e for e in m["kernels"] if e.get("enabled") and e.get("record")]
    if not kernels:
        return {}
    work = run_dir / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    check = ["xe-forge-skill", "inputs", "check", str(work / "inputs.json"), "--manifest",
             str(manifest_path)]  # fmt: skip
    ctx = {"model": a.model, "framework": a.framework, "repos": repos,
           "check_cmd": shlex.join(check), "kernels": kernels}  # fmt: skip
    answer = _agent("inputs", "frontend-inputs.md.j2", ctx, work, "inputs.json", python, turns)
    problems, _ = tool.check(answer if isinstance(answer, dict) else {}, m)  # the tool's verdict
    for kernel, bad in problems.items():
        print(f"inputs: {kernel} not optimized: {'; '.join(bad)}", file=sys.stderr)
    passed = [e["name"] for e in kernels if e["name"] not in problems]
    return tool.weights(answer, passed) if passed else {}


# ── optimize ───────────────────────────────────────────────────────────────


def xe_forge_command(entry: dict, variant: str, repo: Path, workspace: Path, opts) -> list[str]:
    cmd = [
        sys.executable, "-m", "xe_forge.cli",
        "--engine", opts.engine,
        "--auto-launch",
        "--dsl", entry["dsl"],
        "--device", opts.device,
        "--workspace", str(workspace),
        "--max-turns", str(opts.max_turns),
        "--name", entry["name"],
        "--kernel-repo", str(repo),
        "--spec", entry["spec"],
        "--variant", variant,
        "--unitrace",  # device timing, ComputeBasic and EU stall per trial; --no-unitrace wins
    ]  # fmt: skip
    if entry.get("record"):
        cmd += ["--dataset-record", entry["record"]]
    if entry.get("target") == "library":
        cmd += ["--target", "library"]
    if entry.get("reference"):  # a fusion: its chain of ops as PyTorch
        cmd += ["--reference", entry["reference"]]
    for r in entry.get("integration_repos") or []:
        cmd += ["--integration-repo", r]
    if getattr(opts, "stack_python", None):
        cmd += ["--stack-python", opts.stack_python]
    if entry["dsl"] == "sycl" and opts.compiler_flags:
        cmd += ["--compiler-flags", opts.compiler_flags.replace("{kernel_repo}", str(repo))]
    return cmd + list(opts.passthrough)


def optimize(manifest_path: Path, run_dir: Path, opts) -> list[tuple[str, str]]:
    """One Xe-Forge run per enabled (kernel, variant), in sequence; returns the pairs."""
    from xe_forge.frontend import manifest

    data = manifest.load(manifest_path)
    repos_dir = Path(opts.kernel_repos).resolve() if opts.kernel_repos else REPO_DIR.parent
    ws_root = run_dir / "workspaces"
    ws_root.mkdir(parents=True, exist_ok=True)
    from xe_forge import agent_cli as agents

    cli = agents.BACKENDS.get(opts.engine)
    if cli and shutil.which(cli.binary) is None:
        raise SystemExit(f"optimize: --engine {opts.engine} needs the {cli.binary!r} CLI on PATH")
    rest = list(opts.passthrough)
    unitrace = rest[rest.index("--unitrace-bin") + 1] if "--unitrace-bin" in rest[:-1] else (
        os.environ.get("UNITRACE_BIN", "unitrace"))  # fmt: skip
    if "--no-unitrace" not in rest and shutil.which(unitrace) is None:
        raise SystemExit(f"optimize: profiling needs unitrace ({unitrace!r} not found); "
                         "set UNITRACE_BIN, or pass -- --no-unitrace")  # fmt: skip

    pairs = []
    for entry in data["kernels"]:
        key = manifest.key(entry)
        if not entry.get("enabled") or (opts.only and not {entry["name"], key} & set(opts.only)):
            continue
        for variant in entry["variants"][:1]:  # the heaviest; the spec's weights score all
            pairs.append((key, variant))
            workspace = ws_root / f"{key}__{variant}"
            status_path = manifest.sidecar(workspace, ".status.json")
            done = json.loads(status_path.read_text()) if status_path.is_file() else {}
            if done.get("exit_code") == 0 and workspace.is_dir():
                print(f"optimize: {entry['name']} ({variant}) ran before in {workspace}; "
                      "kept (remove it to run again)")  # fmt: skip
                continue
            repo = manifest.resolve_repo(entry["repo"], repos_dir)
            if not repo.is_dir():
                status = {"skipped": f"kernel repo not found: {repo}"}
                status_path.write_text(json.dumps(status, indent=1) + "\n")
                print(f"optimize: {entry['name']} ({variant}): {status['skipped']}")
                continue
            cmd = xe_forge_command(entry, variant, repo, workspace, opts)
            log = manifest.sidecar(workspace, ".log")
            print(
                f"optimize: {entry['name']} ({variant}) -> {workspace}\n  {shlex.join(cmd)}",
                flush=True,
            )
            started = time.time()
            with open(log, "w") as f:
                code = subprocess.call(
                    cmd, cwd=REPO_DIR, stdout=f, stderr=subprocess.STDOUT, env=_session_env()
                )
            status = {
                "exit_code": code,
                "seconds": round(time.time() - started, 1),
                "command": shlex.join(cmd),
                "log": str(log),
                "repo": _git_revision(repo),
            }
            status_path.write_text(json.dumps(status, indent=1) + "\n")
            print(
                f"optimize: {entry['name']} ({variant}) exit {code} in {status['seconds']}s; log {log}"
            )
    return pairs


# ── fuse ───────────────────────────────────────────────────────────────────


def fuse(capture_path: Path, run_dir: Path, a, python, turns, repos) -> Path | None:
    """The optional fusion step (frontend/fuse.py), under ``run_dir/fuse``: its own
    manifest, inputs, workspaces and results; returns their SUMMARY.md, or None."""
    from xe_forge.frontend import collect, ir, manifest, prove
    from xe_forge.frontend import fuse as fusion
    from xe_forge.models import DSL
    from xe_forge.skills import fuse as tool

    work = run_dir / "fuse"
    work.mkdir(parents=True, exist_ok=True)
    try:
        stacked = fusion.stack(prove.winners(run_dir / "results"), work / "stacked")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"fuse: the winning patches do not stack: {exc}", file=sys.stderr)
        return None
    results = run_dir / "results" / "summary.json"
    check = ["xe-forge-skill", "fuse", "check", str(work / "fusions.json"), "--capture",
             str(capture_path), "--results", str(results)]  # fmt: skip
    ctx = {"model": a.model, "framework": a.framework, "run_dir": run_dir,
           "capture": capture_path, "stacked": stacked, "repos": repos,
           "dsls": [d.value for d in DSL], "check_cmd": shlex.join(check),
           "rows": len(a.rows) if len(a.rows) > 1 else 0}  # fmt: skip
    answer = _agent("fuse", "frontend-fuse.md.j2", ctx, work, "fusions.json", python, turns)
    if not answer or not answer.get("fusions"):
        print(f"fuse: none proposed: {(answer or {}).get('why', 'no answer')}")
        return None
    cap = ir.load(capture_path)
    problems, passed = tool.check(answer, cap, tool.wins(results), work)  # the tool's verdict
    for p in problems:
        print(f"fuse: dropped: {p}", file=sys.stderr)
    if not passed:
        return None
    found = fusion.analysis(passed, cap, work, a.fuse_top, a.min_gain)
    path = manifest.emit(found, cap, capture_path, work)
    path = manifest.emit(
        found, cap, capture_path, work, inputs(path, work, a, python, turns, repos)
    )
    print(f"fuse: {len(passed)} passed, enabled: "
          f"{', '.join(e.name for e in found.entries if e.enabled) or 'none'} -> {path}")  # fmt: skip
    pairs = optimize(path, work, a)
    return collect.collect(work, manifest.load(path), pairs)


def _session_env() -> dict:
    """This interpreter's bin first on PATH: the session's ``python`` and ``xe-forge-skill``
    are then the ones running it, however ``run`` itself was started."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    return env


def _git_revision(repo: Path) -> dict:
    """Which code a kernel was optimized in: the clone's origin and HEAD, dirty or not."""

    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "path": str(repo),
        "url": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _add_optimize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--engine", default=os.environ.get("XE_FORGE_AGENT") or "claude",
        choices=("claude", "codex", "dspy-agent"),
        help="the agent CLI (claude, codex) for every agent of the run, or dspy-agent for the "
        "kernels (default: $XE_FORGE_AGENT, else claude)",
    )  # fmt: skip
    p.add_argument("--device", default="xpu")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--only", nargs="*", help="kernel names to run (default: every enabled)")
    p.add_argument(
        "--kernel-repos",
        help="directory holding the kernel repositories (default: beside this checkout)",
    )
    p.add_argument(
        "--compiler-flags",
        default=os.environ.get("XE_FORGE_SYCL_FLAGS"),
        help="SYCL compile flags for sycl entries; {kernel_repo} expands (env XE_FORGE_SYCL_FLAGS)",
    )


# ── entry ──────────────────────────────────────────────────────────────────


def _use_agent(a) -> None:
    """``--engine claude|codex`` drives the frontend agents too (they read XE_FORGE_AGENT)."""
    from xe_forge import agent_cli as agents

    if getattr(a, "engine", None) in agents.BACKENDS:
        os.environ["XE_FORGE_AGENT"] = a.engine


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"usage: xe-forge {{{','.join(COMMANDS)}}} ...")
    command, rest = argv[0], argv[1:]
    passthrough: list[str] = []
    if "--" in rest:
        i = rest.index("--")
        rest, passthrough = rest[:i], rest[i + 1 :]

    if command == "calibrate":
        from xe_forge.frontend import calibrate

        p = argparse.ArgumentParser(prog="xe-forge calibrate")
        p.add_argument("-o", "--output", type=Path, help="default: the per-device cache")
        p.add_argument("--read-bytes", type=int, default=1 << 30, help="> the card's caches")
        p.add_argument("--matmul-n", type=int, default=8192, help="square matmul size")
        p.add_argument("--iters", type=int, default=20)
        p.add_argument("--dtypes", nargs="*", help="matmul dtypes to measure (default: all)")
        a = p.parse_args(rest)
        _use_agent(a)
        return calibrate.main(a.output, a.read_bytes, a.matmul_n, a.iters, a.dtypes)

    if command == "optimize":
        from xe_forge.frontend import collect, manifest

        p = argparse.ArgumentParser(prog="xe-forge optimize")
        p.add_argument("manifest", type=Path)
        _add_optimize_args(p)
        a = p.parse_args(rest)
        _use_agent(a)
        a.passthrough = passthrough
        run_dir = a.manifest.resolve().parent
        pairs = optimize(a.manifest.resolve(), run_dir, a)
        summary = collect.collect(run_dir, manifest.load(a.manifest), pairs)
        print(summary.read_text())
        return 0

    # run: capture (agent) -> name (agent) -> analyze -> optimize -> collect [-> fuse] [-> prove]
    from xe_forge.frontend import collect, manifest

    p = argparse.ArgumentParser(prog="xe-forge run", allow_abbrev=False)
    p.add_argument("-o", "--output", type=Path, required=True, help="run directory")
    p.add_argument("--framework", required=True, help="what runs the model (vllm, openvino, ...)")
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--framework-python", required=True, help="the framework's interpreter")
    p.add_argument(
        "--framework-note", action="append", default=[], metavar="TEXT",
        help="a constraint the capture agent cannot discover: a setting this framework "
        "needs on this machine, and why. Repeatable. The code never reads it -- it is "
        "handed to the agent verbatim. Example: LTX on XPU needs "
        "--stage-overrides '{\"0\":{\"extras\":{\"ltx2_use_conv_vae\":true}}}' because the "
        "default VAE decoder loads a NATTEN kernel that has no XPU build, and the failure "
        "lands after the whole denoise is paid for",
    )  # fmt: skip
    p.add_argument(
        "--reference-python",
        help="GPU torch + transformers, for a patch's reference tokens (default: the framework's)",
    )
    _add_workload_args(p)
    # unset, so --rows can refuse them; the defaults are set once parsed
    p.set_defaults(batch=None, input_len=None, output_len=None)
    p.add_argument("--rows", type=Path, help="rows.yaml: several (batch, input_len, output_len, "
                   "weight) served by one run, in place of --batch/--input-len/--output-len "
                   "(frontend/rows.py)")  # fmt: skip
    p.add_argument(
        "--prompt-file",
        help="text the prompts are sliced from, batch x input-len tokens at least (default: "
        "fixed synthetic token ids)",
    )
    p.add_argument(
        "--precision",
        default="auto",
        help="compute dtype: auto (the framework's default) or a torch name / bf16 ...",
    )
    p.add_argument("--kv-cache", default="auto",
                   help="KV cache dtype: auto (the framework's default) or a torch name, e.g. float8_e4m3fn")  # fmt: skip
    _add_analyze_args(p)
    p.add_argument(
        "--name-coverage", type=float, default=0.99,
        help="share of device time whose ops the naming agent places",
    )  # fmt: skip
    p.add_argument("--analyze-only", action="store_true", help="stop after the manifest")
    p.add_argument(
        "--agent-max-turns", type=int,
        help="turn limit of the capture, name and prove agents (default: --max-turns)",
    )  # fmt: skip
    p.add_argument(
        "--prove", action="store_true", help="build the stack with the winning patches and "
        "measure them in the model (prove.py)"
    )  # fmt: skip
    p.add_argument("--prove-repeat", type=int, default=3, help="captures per arm")
    p.add_argument(
        "--fuse", action="store_true", help="after optimizing, fuse chains of ops into one "
        "kernel on the patched repositories (serving-only, as a rule; frontend/fuse.py)"
    )  # fmt: skip
    p.add_argument("--fuse-top", type=int, default=2, help="fusions to optimize")
    p.add_argument(
        "--stack-check", action="store_true", help="each port-back agent builds and tests "
        "its port in the framework's stack (Xe-Forge --stack-python)"
    )  # fmt: skip
    _add_optimize_args(p)
    a = p.parse_args(rest)
    from xe_forge.frontend.dtypes import canonical, parse_quantization

    try:
        a.precision = a.precision if a.precision == "auto" else canonical(a.precision)
        a.kv_cache = a.kv_cache if a.kv_cache == "auto" else canonical(a.kv_cache)
        parse_quantization(a.quantization)
        from xe_forge.frontend import rows

        given = {k: getattr(a, k) for k in rows.KEYS}
        if a.workload == "text" and (a.prompt is not None or a.negative_prompt is not None):
            raise ValueError("--prompt is for --workload vision; a text run's prompts are "
                             "--prompt-file")  # fmt: skip
        if a.workload != "text":
            text_only = {"--rows": a.rows, "--prompt-file": a.prompt_file,
                         "--kv-cache": a.kv_cache != "auto"}  # fmt: skip
            if given := [f for f, v in text_only.items() if v]:
                raise ValueError(f"{', '.join(given)}: for --workload text only")
            a.prompt = VISION_PROMPT if a.prompt is None else a.prompt
        if a.rows and any(v is not None for v in given.values()):
            raise ValueError("--rows replaces --batch, --input-len and --output-len")
        # the anchor row (the largest) is the one the capture agent captures
        a.rows = rows.load(a.rows) if a.rows else []
        first = a.rows[0] if a.rows else {"batch": 1, "input_len": 512, "output_len": 32}
        for k, v in given.items():
            setattr(a, k, first[k] if v is None else v)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    _use_agent(a)
    a.passthrough = passthrough
    turns = a.agent_max_turns or a.max_turns
    run_dir = a.output.resolve()
    repos = Path(a.kernel_repos).resolve() if a.kernel_repos else REPO_DIR.parent
    # Before an agent is started and a model is loaded: a missing input image would
    # otherwise surface as a launcher failure minutes in, blamed on the framework.
    missing = [x for x in (getattr(a, "input_image", []) or []) if not Path(x).is_file()]
    if missing:
        print(f"run: no such input image: {', '.join(missing)}", file=sys.stderr)
        return 1
    stack = run_dir / "stack.json"  # a patched framework, when one was needed
    python = json.loads(stack.read_text())["python"] if stack.is_file() else a.framework_python
    # The run's inputs (runtime.inputs): a copy in the run dir, so every agent and gate reads
    # one stable path, tied by its sha256. Room for other kinds (images, observations).
    a.inputs = {}
    if a.prompt_file:
        from xe_forge.skills.capture import _sha256

        prompts = run_dir / "inputs" / "prompts.txt"
        prompts.parent.mkdir(parents=True, exist_ok=True)
        if not prompts.is_file() or prompts.read_bytes() != Path(a.prompt_file).read_bytes():
            shutil.copyfile(a.prompt_file, prompts)
        a.inputs = {"prompts": {"file": str(prompts), "sha256": _sha256(prompts)}}
    capture_path = run_dir / "capture.json"
    if not capture_path.is_file():
        code, python = capture(a, python, run_dir, turns, repos)
        if code != 0:
            why = f"killed by signal {-code}" if code < 0 else f"exit {code}"
            print(f"run: stopped before analysis ({why})", file=sys.stderr)
            return code if code > 0 else 1
    from xe_forge.frontend import ir

    cap = ir.load(capture_path)  # once; the parse with names below rewrites it for analyze
    rt = cap.run.get("runtime") or {}
    want = {"framework": a.framework, **workload(a), "prompts": a.inputs.get("prompts")}
    # The run's one input is its prompt file; other files the launcher records (a tokenizer)
    # are its own, checked by the parse, not part of which run this is.
    have = {"framework": cap.run.get("framework"), **{k: rt.get(k) for k in workload(a)},
            "prompts": (rt.get("inputs") or {}).get("prompts")}  # fmt: skip
    for k in (
        "precision",
        "quantization",
        "kv_cache",
    ):  # declared ones the launcher writes verbatim
        if getattr(a, k) != "auto":
            want[k], have[k] = getattr(a, k), str(rt.get(k))
    if "quantization" in want and have["quantization"].rpartition(":")[0] == want["quantization"]:
        have["quantization"] = want[
            "quantization"
        ]  # a declared scheme, the stored dtype added to it
    model_was = (cap.run.get("model") or {}).get("name")
    if not _same_workload(want, have) or not _same_model(a.model, model_was):
        print(f"run: {capture_path} is a capture of {have} of {model_was!r}, not {want} of "
              f"{a.model!r}; remove it to capture again", file=sys.stderr)  # fmt: skip
        return 1
    code, python = name_and_read(a, cap, run_dir, python, turns, repos)
    if code != 0:
        return code
    if len(a.rows) > 1:  # analyze and every step after it read the merged capture
        merged, problem = capture_rows(a, run_dir, python)
        if problem:  # a launcher that ignores the rows' sizes goes back once to its agent
            print(f"run: {problem}; back to the capture agent", file=sys.stderr)
            if (run_dir / "session.log").is_file():
                (run_dir / "session.log").replace(run_dir / "session-rows.log")
            code, python = capture(a, python, run_dir, turns, repos, refused=[problem])
            if code == 0:
                code, python = name_and_read(a, ir.load(capture_path), run_dir, python, turns,
                                             repos)  # fmt: skip
            if code != 0:
                return code if code > 0 else 1
            merged, problem = capture_rows(a, run_dir, python)
        if merged is None:
            print(f"run: the rows were not captured{f' ({problem})' if problem else ''}",
                  file=sys.stderr)  # fmt: skip
            return 1
        capture_path = merged
        if a.prove:
            print("run: --prove measures the anchor row only", file=sys.stderr)
    _collect_outputs(run_dir)
    path = analyze(capture_path, run_dir, a)
    path = analyze(capture_path, run_dir, a, inputs(path, run_dir, a, python, turns, repos))
    if a.workload != "text" and not a.analyze_only:
        # Optimizing is allowed here, and nothing is merged by it. A trial is kept only if
        # it is faster AND torch.allclose against the baseline kernel at the spec's
        # tolerance -- a comparison of two tensors, which knows nothing about images. What
        # a vision workload has no answer for is the step after: proving the whole model
        # still produces an acceptable image. A kernel can pass its tolerance and still
        # shift the picture, because the difference compounds over the denoising steps.
        # So a patch is written and left for a person to judge; `mergeable` stays no
        # without a stack check, which is what collect already does.
        print(f"run: {a.workload} workload -- a missing-op patch is checked against the "
              "framework's own torch path (patch-*/quality.json, with the op gain), but an "
              "optimized kernel is not: it is kept on a per-kernel tolerance, and --prove "
              "compares time only. A kernel can pass its tolerance and still shift the "
              "picture over the denoising steps, so look at the images before using any "
              "patch from this run.")  # fmt: skip
    if a.analyze_only:
        return 0
    a.stack_python = python if a.stack_check else None
    pairs = optimize(path, run_dir, a)
    summary = collect.collect(run_dir, manifest.load(path), pairs)
    if a.fuse:
        fused = fuse(capture_path, run_dir, a, python, turns, repos)
        if fused is not None:
            table = fused.read_text().split("\n", 2)[2]  # without its "# Results" title
            summary.write_text(summary.read_text() + "\n## Fusions\n\n" + table)
    if a.prove:
        from xe_forge.frontend import prove

        section = prove.prove(run_dir, recapture, python, a.prove_repeat, turns)
        summary.write_text(summary.read_text() + "\n" + section)
    print(summary.read_text())
    return 0
