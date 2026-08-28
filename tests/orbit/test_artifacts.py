"""
Artifact persistence, schema contracts and replay (plan §16.2, §16.3).

Every stage's definition of done requires a published schema, a round-trip contract
test, a golden fixture and at least one negative test that produces a clean typed
failure rather than a crash or a silent default. These are those tests.
"""

from __future__ import annotations

import json

import pytest

from xe_forge.orbit import schemas
from xe_forge.orbit.artifacts import ArtifactError, RunStore
from xe_forge.orbit.models import SCHEMA_VERSION, WorkloadMeasurement, WorkloadSpec
from xe_forge.orbit.stats import estimate


class TestRunStore:
    def test_typed_artifact_round_trips(self, store):
        spec = WorkloadSpec(command=["python", "train.py"], repetitions=7)
        store.save("workload.json", spec)
        reloaded = store.load("workload.json", WorkloadSpec)
        assert reloaded.command == spec.command
        assert reloaded.repetitions == 7

    def test_missing_artifact_is_a_typed_failure(self, store):
        """Never a bare KeyError or a blank default — the message must say what to run."""
        with pytest.raises(ArtifactError, match="missing artifact"):
            store.load("catalog.json", WorkloadSpec)

    def test_malformed_json_is_a_typed_failure(self, store):
        target = store.path("workload.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="not valid JSON"):
            store.load("workload.json", WorkloadSpec)

    def test_wrong_shape_is_a_typed_failure(self, store):
        store.save_json("workload.json", {"schema_version": SCHEMA_VERSION, "nope": 1})
        with pytest.raises(ArtifactError, match="does not match"):
            store.load("workload.json", WorkloadSpec)

    def test_incompatible_schema_version_is_refused(self, store):
        """A stored artifact from an older major version must not be silently reused."""
        store.save_json("workload.json", {"schema_version": "99.0", "command": ["python", "x.py"]})
        with pytest.raises(ArtifactError, match="schema version"):
            store.load("workload.json", WorkloadSpec)

    def test_writes_are_atomic(self, store):
        """A partial write must never leave a half-parsed artifact behind."""
        spec = WorkloadSpec(command=["a"])
        store.save("workload.json", spec)
        assert not list(store.run_dir.glob("*.tmp"))
        assert json.loads(store.path("workload.json").read_text())["command"] == ["a"]

    def test_run_ids_are_validated(self, tmp_path):
        with pytest.raises(ArtifactError, match="invalid run id"):
            RunStore(tmp_path, "../escape")

    def test_opening_an_unknown_run_fails_clearly(self, tmp_path):
        with pytest.raises(ArtifactError, match="no such run"):
            RunStore.open("run-does-not-exist", base=tmp_path / ".orbit")

    def test_latest_finds_the_newest_run(self, tmp_path):
        base = tmp_path / ".orbit"
        first = RunStore.create(base=base, run_id="run-0001")
        second = RunStore.create(base=base, run_id="run-0002")
        first.save("workload.json", WorkloadSpec(command=["a"]))
        second.save("workload.json", WorkloadSpec(command=["b"]))
        assert RunStore.latest(base=base).run_id == "run-0002"

    def test_latest_on_an_empty_root_fails_clearly(self, tmp_path):
        with pytest.raises(ArtifactError):
            RunStore.latest(base=tmp_path / ".orbit")

    def test_replay_reads_back_what_a_stage_wrote(self, store):
        """The core of --replay: a later stage re-reads artifacts, not hardware."""
        measurement = WorkloadMeasurement(wall_time=estimate([1.0, 1.1, 0.9, 1.05, 0.95]))
        store.save("measurement.json", measurement)

        reopened = RunStore.open(store.run_id, base=store.root)
        replayed = reopened.load("measurement.json", WorkloadMeasurement)
        assert replayed.wall_time.n == 5
        assert replayed.wall_time.mean == pytest.approx(measurement.wall_time.mean)


class TestSchemas:
    def test_every_artifact_has_a_schema(self):
        generated = schemas.all_schemas()
        assert set(generated) == set(schemas.ARTIFACT_MODELS)
        for name, schema in generated.items():
            assert schema["x-orbit-artifact"] == name
            assert schema["x-orbit-schema-version"] == SCHEMA_VERSION
            assert "properties" in schema

    def test_round_trip_is_stable(self):
        """schema -> object -> schema must not change representation (§16.2 item 3).

        If it did, a stage reading a stored artifact would see something different
        from what was written, and replay would quietly diverge from the live run.
        """
        assert schemas.roundtrip("workload", {"command": ["python", "x.py"]})

    def test_unknown_artifact_name_is_rejected(self):
        with pytest.raises(KeyError):
            schemas.schema_for("not_an_artifact")

    def test_compatibility_check_flags_a_removed_field(self):
        stored = schemas.schema_for("workload")
        stored["properties"]["legacy_field"] = {"type": "string"}
        problems = schemas.check_compatibility("workload", stored)
        assert any("legacy_field" in p for p in problems)

    def test_compatibility_check_passes_against_itself(self):
        current = schemas.schema_for("catalog")
        assert schemas.check_compatibility("catalog", current) == []

    def test_export_writes_reviewable_files(self, tmp_path):
        written = schemas.export_schemas(tmp_path)
        assert len(written) == len(schemas.ARTIFACT_MODELS)
        for path in written:
            payload = json.loads(path.read_text())
            assert payload["x-orbit-schema-version"] == SCHEMA_VERSION


class TestWorkloadSpec:
    def test_empty_command_is_rejected(self):
        with pytest.raises(ValueError):
            WorkloadSpec(command=[])

    def test_repetition_default_supports_a_decision(self):
        """The default must not be below what §17 requires for accept/reject."""
        assert WorkloadSpec(command=["x"]).repetitions >= 5
