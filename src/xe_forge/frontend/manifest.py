"""Write and read the kernel manifest: ``kernels.yaml``, ``specs/<name>.yaml`` and
``records/<name>.json``. Every entry is kept, enabled or not, with its reason.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from xe_forge.frontend.analyze import Analysis, Entry
from xe_forge.frontend.canonical import resolve_dims
from xe_forge.frontend.ir import CaptureRun, Workload

_TOLERANCE = {
    "bfloat16": (0.02, 0.01),
    "float16": (0.01, 0.005),
    "float32": (1e-4, 1e-5),
}

_READ_WITH = """\
`{capture}` is an `xe_forge.capture/v1` file written by `xe-forge-skill capture parse` from a run of
`{model}` under {framework}. Each variant above names a workload in its `workloads` list
by `workload_id`, followed by `@AXIS=value`: the token count that variant was captured at.
The workload's `ops` name rows of the file's `ops` list, whose `args` hold the exact shapes,
dtypes and strides the framework passed, `scalars` the non-tensor arguments, and
`call_site` the module that called it. It is plain JSON:
`python -c "import json; c = json.load(open('{capture}'))"`.
"""


def _dtype(w: Workload) -> str:
    floats = [d for d in w.dtypes if d.startswith(("bfloat", "float"))]
    return floats[0] if floats else (w.dtypes[0] if w.dtypes else "float32")


def _variants(e: Entry) -> list[tuple[str, Workload, str | None, int | None, float]]:
    """(variant, workload, var axis, value, device us) for every captured shape, costliest
    first: which of them a trial is scored on, and how much each counts, is the inputs
    agent's choice (``weights``)."""
    rows = []
    for i, w in enumerate(e.workloads):
        if not w.dims:
            continue
        suffix = f"-w{i}" if len(e.workloads) > 1 else ""
        if not w.var_axes:
            rows.append((f"bench{suffix}", w, None, None, w.device_us))
            continue
        axis, hist = next(iter(w.var_axes.items()))
        for value, h in hist.items():
            rows.append((f"bench-t{value}{suffix}", w, axis, int(value), h["device_us"]))
    rows.sort(key=lambda r: -r[4])
    return rows


def build_spec(
    e: Entry, weights: dict[str, float] | None = None
) -> tuple[dict, dict[str, str]] | None:
    """The spec (every shape a variant, ``weights`` the ones a trial is scored on: the
    given ones, else each shape's share of captured time) and its record."""
    rows = _variants(e)
    if not rows:
        return None
    total = sum(r[4] for r in rows) or 1.0
    chosen = weights or {r[0]: r[4] / total for r in rows if r[4] > 0}
    chosen = dict(sorted(chosen.items(), key=lambda kv: -kv[1]))
    spec: dict = {"default_variant": next(iter(chosen)),
                  "weights": {k: round(v, 4) for k, v in chosen.items()}}  # fmt: skip
    record: dict[str, str] = {}
    smallest = min(rows, key=lambda r: r[3] if r[3] is not None else 0)
    for name, w, axis, value, _ in [("ci", *smallest[1:]), *rows]:
        rtol, atol = _TOLERANCE.get(_dtype(w), (0.05, 0.05))
        spec[name] = [
            {"dtype": _dtype(w), "rtol": rtol, "atol": atol, "dims": resolve_dims(w, value)}
        ]
        if name != "ci":
            record[name] = w.workload_id + (f"@{axis}={value}" if axis else "")
    return spec, record


def emit(
    analysis: Analysis,
    capture: CaptureRun,
    capture_path: Path,
    out_dir: Path,
    weights: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Write kernels.yaml, specs/ and records/ under ``out_dir``; return the manifest.
    ``weights``: the inputs agent's checked shapes per kernel name, once it ran; an enabled
    kernel without them is not optimized."""
    from xe_forge.core.spec_loader import load_spec

    out_dir = out_dir.resolve()
    specs, records = out_dir / "specs", out_dir / "records"
    for d in (specs, records):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    run = capture.run
    model = (run.get("model") or {}).get("name", "?")
    used: set[str] = set()
    kernels = []
    for e in analysis.entries:
        item = {
            "name": e.name,
            "repo": e.repo,
            "dsl": e.dsl,
            "enabled": e.enabled,
            "workload_ids": [w.workload_id for w in e.workloads],
            "priority": e.priority,
            "reason": e.reason,
        }
        if e.target == "library":
            item["target"] = "library"
        if weights is not None and e.enabled and e.name not in weights:
            e.enabled, e.reason = False, f"{e.reason}; no checked trial shapes (inputs/)"
            item.update(enabled=False, reason=e.reason)
        wanted = e.repo and (e.target == "kernel" or e.enabled)
        built = build_spec(e, (weights or {}).get(e.name)) if wanted else None
        if built is not None:
            spec, variants = built
            repo = _flat(e.repo)
            stems = (e.name, f"{e.name}__{repo}", f"{e.name}__{repo}__{e.target}")
            stem = next(s for s in stems if s not in used)
            used.add(stem)
            spec_path = specs / f"{stem}.yaml"
            header = (
                f"# {e.name} at {model} shapes; generated by xe-forge analyze from capture "
                f"{analysis.run_id}.\n# No inputs: the dims go as keyword arguments to the "
                f"baseline's get_init_inputs/get_inputs.\n"
            )
            spec_path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
            load_spec(spec_path)  # fail now rather than in a workspace
            record_path = records / f"{stem}.json"
            record_path.write_text(
                json.dumps(
                    {
                        "dataset": str(capture_path.resolve()),
                        "definition": e.name,
                        "variants": variants,
                        "read_with": _READ_WITH.format(
                            capture=capture_path.resolve(),
                            model=model,
                            framework=run.get("framework"),
                        ),
                        "input_fidelity": "shapes-only",
                    },
                    indent=1,
                )
                + "\n"
            )
            item.update(
                spec=str(spec_path.relative_to(out_dir)),
                record=str(record_path.relative_to(out_dir)),
                variants=list(spec["weights"]),
            )
        elif e.enabled:
            item["enabled"] = False
            item["reason"] += "; no spec could be built"
        kernels.append(item)

    manifest = out_dir / "kernels.yaml"
    manifest.write_text(
        f"# Generated by xe-forge analyze from {capture_path.resolve()} (run {analysis.run_id}).\n"
        "# One Xe-Forge run per kernel, scored on its spec's weighted variants.\n"
        + yaml.safe_dump(
            {"kernels": kernels},
            sort_keys=False,
            width=100,
        )
    )
    return manifest


def load(path: str | Path) -> dict:
    """Read a manifest with absolute ``spec``/``record`` paths; raise on an incomplete entry."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("kernels"), list):
        raise ValueError(f"{path}: expected a mapping with a 'kernels' list")
    for entry in data["kernels"]:
        for key in ("spec", "record"):
            if entry.get(key) and not Path(entry[key]).is_absolute():
                entry[key] = str((path.parent / entry[key]).resolve())
        if entry.get("enabled", True):
            missing = [k for k in ("name", "repo", "dsl", "spec", "variants") if not entry.get(k)]
            if missing:
                raise ValueError(f"{path}: enabled entry {entry.get('name')!r} lacks {missing}")
    return data


def resolve_repo(repo: str, repos_dir: Path) -> Path:
    p = Path(repo).expanduser()
    return p if p.is_absolute() else (repos_dir / p).resolve()


def _flat(repo: str) -> str:
    """A repo path (``vllm/csrc``) as one filename part."""
    return re.sub(r"[^\w.-]+", "-", repo)


def sidecar(workspace: Path, suffix: str) -> Path:
    """``<workspace>.<suffix>`` beside it; ``with_suffix`` would cut a dotted kernel name."""
    return workspace.parent / f"{workspace.name}{suffix}"


def key(entry: dict) -> str:
    """An entry's unique workspace/results name: its spec's stem, else its name."""
    return Path(entry["spec"]).stem if entry.get("spec") else entry["name"]
