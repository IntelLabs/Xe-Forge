"""
Typed artifact persistence and replay for `.orbit/runs/<run-id>/`. Every stage reads
and writes typed artifacts; `RunStore.load()` is what makes `--replay` work.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from xe_forge.orbit.models import SCHEMA_VERSION

T = TypeVar("T", bound=BaseModel)

ORBIT_DIR_NAME = ".orbit"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactError(RuntimeError):
    """Raised for a missing, malformed or incompatible artifact."""


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}"


def schema_major(version: str) -> str:
    return version.split(".", 1)[0]


class RunStore:
    """Reads and writes the artifacts of a single Orbit run."""

    def __init__(self, root: Path, run_id: str) -> None:
        if not _RUN_ID_RE.match(run_id):
            raise ArtifactError(f"invalid run id: {run_id!r}")
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, base: Path | None = None, run_id: str | None = None) -> RunStore:
        root = Path(base) if base else Path.cwd() / ORBIT_DIR_NAME
        store = cls(root, run_id or new_run_id())
        store.run_dir.mkdir(parents=True, exist_ok=True)
        return store

    @classmethod
    def open(cls, run_id: str, base: Path | None = None) -> RunStore:
        root = Path(base) if base else Path.cwd() / ORBIT_DIR_NAME
        store = cls(root, run_id)
        if not store.run_dir.is_dir():
            raise ArtifactError(f"no such run: {run_id} (looked in {store.run_dir})")
        return store

    @classmethod
    def latest(cls, base: Path | None = None) -> RunStore:
        root = Path(base) if base else Path.cwd() / ORBIT_DIR_NAME
        runs_dir = root / "runs"
        if not runs_dir.is_dir():
            raise ArtifactError(f"no runs directory at {runs_dir}")
        runs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        if not runs:
            raise ArtifactError(f"no runs recorded under {runs_dir}")
        return cls(root, runs[-1].name)

    @classmethod
    def list_runs(cls, base: Path | None = None) -> list[str]:
        root = Path(base) if base else Path.cwd() / ORBIT_DIR_NAME
        runs_dir = root / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted(d.name for d in runs_dir.iterdir() if d.is_dir())

    # -- paths -------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def subdir(self, *parts: str) -> Path:
        d = self.path(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- typed persistence -------------------------------------------------

    def save(self, name: str, artifact: BaseModel) -> Path:
        """Write a typed artifact. `name` is a path relative to the run directory."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact.model_dump(mode="json")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, name: str, model: type[T]) -> T:
        """Read a typed artifact, validating both schema version and shape."""
        target = self.path(name)
        if not target.is_file():
            raise ArtifactError(
                f"missing artifact {name!r} in run {self.run_id}; "
                f"run the stage that produces it, or pass a run that has it"
            )
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"artifact {name!r} is not valid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise ArtifactError(f"artifact {name!r} must be a JSON object")

        stored = raw.get("schema_version")
        if stored is not None and schema_major(str(stored)) != schema_major(SCHEMA_VERSION):
            raise ArtifactError(
                f"artifact {name!r} has schema version {stored}, incompatible with "
                f"{SCHEMA_VERSION}; re-run the stage that produced it"
            )

        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise ArtifactError(
                f"artifact {name!r} does not match {model.__name__}: {exc}"
            ) from exc

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def save_json(self, name: str, payload: object) -> Path:
        """Escape hatch for raw third-party output (a profiler trace, for example)."""
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def load_json(self, name: str) -> object:
        target = self.path(name)
        if not target.is_file():
            raise ArtifactError(f"missing file {name!r} in run {self.run_id}")
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"file {name!r} is not valid JSON: {exc}") from exc

    def record_stage(self, stage: str) -> None:
        """Append a stage name to the manifest's completion list.

        Creates a minimal manifest when the run has none — a stage can legitimately be
        the first thing to touch a run.
        """
        from xe_forge.orbit.models import RunManifest

        if not self.exists("manifest.json"):
            self.save("manifest.json", RunManifest(run_id=self.run_id))
        manifest = self.load("manifest.json", RunManifest)
        if stage not in manifest.stages_completed:
            manifest.stages_completed.append(stage)
            self.save("manifest.json", manifest)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RunStore(run_id={self.run_id!r}, dir={self.run_dir})"


# Canonical artifact names, so stages never spell them differently.
MANIFEST = "manifest.json"
WORKLOAD = "workload.json"
ENVIRONMENT = "environment.json"
MEASUREMENT = "measurement.json"
TORCH_TRACE = "traces/torch_trace.json"
EVENTS = "traces/events.json"
LAUNCHES = "traces/launches.json"
UNITRACE_DIR = "traces/unitrace"
KERNEL_CATALOG = "kernels/catalog.json"
REGION_CATALOG = "regions/catalog.json"
REPORT = "report.json"
