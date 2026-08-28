"""
Versioned artifact schemas and contract tests (plan §16.2).

A stage is not complete until its input and output artifacts have a committed,
versioned JSON schema. This module is the registry: it names every persisted artifact,
generates its JSON Schema from the pydantic model, and provides the round-trip check
(`schema -> object -> schema`) the contract tests assert.

Schemas are generated rather than hand-written so they cannot drift from the models,
and `export_schemas()` writes them to disk so a schema change shows up as a reviewable
diff rather than a silent behaviour change.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from xe_forge.orbit.models import (
    SCHEMA_VERSION,
    CapturedInvocation,
    ComparisonOutcome,
    EnvironmentInfo,
    KernelBundle,
    KernelCatalog,
    RunManifest,
    WorkloadMatrix,
    WorkloadMeasurement,
    WorkloadSpec,
)
from xe_forge.orbit.profiling.interception import LaunchLog
from xe_forge.orbit.profiling.trace import TraceEvents

# Artifact name -> model. The name is the filename the stage writes under the run
# directory, so this table doubles as the layout contract in §23.
ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "manifest": RunManifest,
    "workload": WorkloadSpec,
    "environment": EnvironmentInfo,
    "measurement": WorkloadMeasurement,
    "events": TraceEvents,
    "launches": LaunchLog,
    "catalog": KernelCatalog,
    "capture": CapturedInvocation,
    "bundle": KernelBundle,
    "matrix": WorkloadMatrix,
    "decision": ComparisonOutcome,
}

SCHEMA_DIR = Path(__file__).parent


def schema_for(name: str) -> dict:
    """JSON Schema for one named artifact."""
    if name not in ARTIFACT_MODELS:
        raise KeyError(f"unknown artifact {name!r}; known: {sorted(ARTIFACT_MODELS)}")
    schema = ARTIFACT_MODELS[name].model_json_schema()
    schema["x-orbit-schema-version"] = SCHEMA_VERSION
    schema["x-orbit-artifact"] = name
    return schema


def all_schemas() -> dict[str, dict]:
    return {name: schema_for(name) for name in ARTIFACT_MODELS}


def export_schemas(target: Path | None = None) -> list[Path]:
    """Write every schema to disk so changes are reviewable in a diff."""
    directory = Path(target) if target else SCHEMA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in all_schemas().items():
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def roundtrip(name: str, payload: dict) -> dict:
    """schema -> object -> schema round trip, the contract test in §16.2 item 3.

    Validating a payload and re-dumping it must be stable. A field that silently
    changes representation on the way through breaks replay, because a stage reading a
    stored artifact would then see something different from what was written.
    """
    model = ARTIFACT_MODELS[name]
    obj = model.model_validate(payload)
    first = obj.model_dump(mode="json")
    again = model.model_validate(first).model_dump(mode="json")
    if first != again:
        raise ValueError(f"artifact {name!r} does not round-trip stably")
    return again


def check_compatibility(name: str, stored_schema: dict) -> list[str]:
    """Report fields a stored schema has that the current model no longer accepts.

    Used by the compatibility test against the previous schema version. Returned as
    reasons rather than a boolean so a failure explains itself.
    """
    current = schema_for(name)
    current_props = set(current.get("properties", {}))
    stored_props = set(stored_schema.get("properties", {}))

    problems: list[str] = []
    for removed in sorted(stored_props - current_props):
        problems.append(f"field {removed!r} was removed from {name}")

    current_required = set(current.get("required", []))
    stored_required = set(stored_schema.get("required", []))
    for added in sorted(current_required - stored_required):
        problems.append(f"field {added!r} became required in {name}, breaking older artifacts")

    return problems
