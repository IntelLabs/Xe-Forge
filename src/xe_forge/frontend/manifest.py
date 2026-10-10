"""The kernel manifest and its specs: the only thing the frontend hands Xe-Forge.

    kernels.yaml        one entry per kernel: name, repo, dsl, spec, variants, enabled
    specs/<name>.yaml   no-inputs spec: dims go as keyword arguments to the baseline's
                        get_init_inputs/get_inputs, the baseline copy being the oracle
    records/<name>.json DatasetRecord: which captured workload each variant times

The same format ``scripts/nemotron30b/extract.py`` writes by hand and
``scripts/gen_kernel_sbatch.py`` turns into one Slurm job per (kernel, variant). Every
entry is kept, enabled or not, with the reason, so the manifest is the full record of the
decision and an entry can be switched on by hand. A ``variants`` list is ordered by device
time; ``xe-forge optimize`` runs the first unless told to run them all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from xe_forge.frontend.analyze import Analysis, Entry
from xe_forge.frontend.ir import CaptureRun, Workload

COVERAGE = 0.9
MAX_VARIANTS = 4

_TOLERANCE = {
    "bfloat16": (0.02, 0.01),
    "float16": (0.01, 0.005),
    "float32": (1e-4, 1e-5),
}

_READ_WITH = """\
`{capture}` is an `xe_forge.capture/v1` file written by `xe-forge capture` from a run of
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
    """(variant name, workload, var axis, value, device us), costliest first; per workload
    the smallest set of values covering COVERAGE of its time, at most MAX_VARIANTS."""
    rows = []
    for i, w in enumerate(e.workloads):
        if not w.dims:
            continue
        suffix = f"-w{i}" if len(e.workloads) > 1 else ""
        if not w.var_axes:
            rows.append((f"bench{suffix}", w, None, None, w.device_us))
            continue
        axis, hist = next(iter(w.var_axes.items()))
        values = sorted(hist.items(), key=lambda kv: -kv[1]["device_us"])
        covered = 0.0
        for value, h in values[:MAX_VARIANTS]:
            rows.append((f"bench-t{value}{suffix}", w, axis, int(value), h["device_us"]))
            covered += h["device_us"]
            if covered >= COVERAGE * w.device_us:
                break
    rows.sort(key=lambda r: -r[4])
    return rows


def _dims(w: Workload, value: int | None) -> dict:
    """Concrete dims for one captured token count: var dims that follow the token count
    take it; the others take what the heaviest call at that count used."""
    with_ = {}
    if value is not None and w.var_axes:
        with_ = next(iter(w.var_axes.values())).get(str(value), {}).get("with", {})
    return {k: (with_.get(k, value) if v == "var" else v) for k, v in w.dims.items()}


def build_spec(e: Entry) -> tuple[dict, dict[str, str]] | None:
    rows = _variants(e)
    if not rows:
        return None
    spec: dict = {"default_variant": rows[0][0]}
    record: dict[str, str] = {}
    smallest = min(rows, key=lambda r: r[3] if r[3] is not None else 0)
    for name, w, axis, value, _ in [("ci", *smallest[1:]), *rows]:
        rtol, atol = _TOLERANCE.get(_dtype(w), (0.05, 0.05))
        spec[name] = [{"dtype": _dtype(w), "rtol": rtol, "atol": atol, "dims": _dims(w, value)}]
        if name != "ci":
            record[name] = w.workload_id + (f"@{axis}={value}" if axis else "")
    return spec, record


def emit(
    analysis: Analysis,
    capture: CaptureRun,
    capture_path: Path,
    out_dir: Path,
    partition: str = "b70",
) -> Path:
    """Write kernels.yaml, specs/ and records/ under ``out_dir``; return the manifest."""
    from xe_forge.core.spec_loader import load_spec

    out_dir = out_dir.resolve()
    specs, records = out_dir / "specs", out_dir / "records"
    for d in (specs, records):  # generated: a re-analysis must not leave stale entries
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
        built = build_spec(e) if e.repo else None
        if built is not None:
            spec, variants = built
            stem = e.name if e.name not in used else f"{e.name}__{e.repo}"
            used.add(stem)
            spec_path = specs / f"{stem}.yaml"
            header = (
                f"# {e.name} at {model} shapes; generated by xe-forge analyze from capture "
                f"{analysis.run_id}.\n# No inputs: the dims go as keyword arguments to the "
                f"baseline's get_init_inputs/get_inputs.\n"
            )
            spec_path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
            load_spec(spec_path)  # fail here, not in a workspace, on a spec Xe-Forge rejects
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
                variants=list(variants),
            )
        elif e.enabled:
            item["enabled"] = False
            item["reason"] += "; no spec could be built"
        kernels.append(item)

    manifest = out_dir / "kernels.yaml"
    manifest.write_text(
        f"# Generated by xe-forge analyze from {capture_path.resolve()} (run {analysis.run_id}).\n"
        "# One Xe-Forge run per (kernel, variant); set enabled: false or trim variants by hand.\n"
        + yaml.safe_dump({"partition": partition, "kernels": kernels}, sort_keys=False, width=100)
    )
    return manifest


def load(path: str | Path) -> dict:
    """Read a manifest; ``spec`` and ``record`` paths come back absolute (relative ones are
    relative to the manifest). An enabled entry missing what a run needs is an error."""
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
