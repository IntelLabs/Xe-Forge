"""The dataset record, and what it adds to a generated workspace.

Two properties are worth a test rather than a reading: a workspace with no record
must render exactly what it rendered before the seam existed, and a workspace with
one must say where the data is *and* say what reading it does not settle. The
second is the whole point -- a session told it may open the dataset, and not told
that an inspection is not a measurement, has been handed a new way to invent a
number.
"""

from __future__ import annotations

import json

import pytest

from xe_forge.claude.generator import generate_workspace
from xe_forge.config import Config
from xe_forge.core.dataset_record import (
    DEFAULT_PROFILE_PATH,
    INPUT_FIDELITIES,
    load_dataset_record,
)


def _record(tmp_path, **overrides):
    payload = {
        "dataset": str(tmp_path / "trace-set"),
        "definition": "moe_double_gemm",
        "variants": {"bench-xpu": "wl-aaa", "bench-xpu-1": "wl-bbb"},
        "read_with": "```python\nfrom somewhere import open_it\n```",
    }
    payload.update(overrides)
    path = tmp_path / "kernel.fib.json"
    path.write_text(json.dumps(payload))
    return path


def _workspace(tmp_path, record_path=None, *, external_benchmark="host-bench {trial}"):
    config = Config()
    config.external.benchmark = external_benchmark
    config.external.dataset_record = str(record_path) if record_path else None
    ws = tmp_path / "ws"
    generate_workspace(ws, config, "moe_double_gemm", "// kernel\n", reference_code="x = 1\n")
    return ws


# --------------------------------------------------------------------- loading


def test_no_record_is_not_an_error(tmp_path):
    assert load_dataset_record(None) is None
    assert load_dataset_record("") is None


def test_record_naming_no_dataset_is_ignored(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"definition": "x", "tolerance": {"atol": 1e-3}}))
    assert load_dataset_record(path) is None


def test_missing_record_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset_record(tmp_path / "absent.json")


def test_malformed_record_raises(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        load_dataset_record(path)


def test_record_fields(tmp_path):
    rec = load_dataset_record(_record(tmp_path))
    assert rec is not None
    assert rec.path == str(tmp_path / "trace-set")
    assert rec.definition == "moe_double_gemm"
    assert rec.variants == {"bench-xpu": "wl-aaa", "bench-xpu-1": "wl-bbb"}
    assert rec.profile_path == DEFAULT_PROFILE_PATH


# ------------------------------------------------------------------- rendering


def test_workspace_without_record_is_unchanged(tmp_path):
    ws = _workspace(tmp_path)
    claude_md = (ws / "CLAUDE.md").read_text()
    assert "THE WORKLOAD DATA" not in claude_md
    assert DEFAULT_PROFILE_PATH not in claude_md
    # Without a dataset record, no workload-inspector agent is generated.
    assert not (ws / ".claude" / "agents" / "workload-inspector.md").exists()


def test_workspace_with_record_names_the_data(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path))
    claude_md = (ws / "CLAUDE.md").read_text()

    assert "THE WORKLOAD DATA" in claude_md
    assert str(tmp_path / "trace-set") in claude_md
    assert "moe_double_gemm" in claude_md
    # The variant->workload map, so a finding names the variant to re-measure on.
    assert "`bench-xpu-1`" in claude_md and "wl-bbb" in claude_md


def test_workspace_with_record_states_the_boundary(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path))
    claude_md = (ws / "CLAUDE.md").read_text()

    assert "read-only" in claude_md.lower()
    assert "inspection is not a measurement" in claude_md.lower()
    assert "special-case" in claude_md
    # CLAUDE.md rule 3 is still rendered: benchmark is the only source of a timing.
    assert "`xe-forge-skill benchmark` is the\n   only source of a timing" in claude_md


def test_profile_is_the_only_new_writable_file(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path))
    claude_md = (ws / "CLAUDE.md").read_text()
    assert "is exactly one exception, and no others" in claude_md
    assert DEFAULT_PROFILE_PATH in claude_md


def test_inspector_agent_carries_the_record(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path))
    agent = (ws / ".claude" / "agents" / "workload-inspector.md").read_text()

    assert agent.startswith("---\nname: workload-inspector\n")
    assert str(tmp_path / "trace-set") in agent
    assert "from somewhere import open_it" in agent  # the host's own reading instructions
    assert DEFAULT_PROFILE_PATH in agent
    assert "Read-only" in agent


def test_slash_command_mirrors_the_workspace(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path))
    cmd = (ws / ".claude" / "commands" / "optimize-kernel.md").read_text()
    assert "workload-inspector" in cmd
    assert DEFAULT_PROFILE_PATH in cmd
    assert "read-only" in cmd


def test_record_renders_without_a_host_reader(tmp_path):
    """A host that can name the dataset and nothing else still gets the facts."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"dataset": str(tmp_path / "trace-set")}))
    ws = _workspace(tmp_path, path)
    assert "THE WORKLOAD DATA" in (ws / "CLAUDE.md").read_text()
    agent = (ws / ".claude" / "agents" / "workload-inspector.md").read_text()
    assert "## How to open it" not in agent


# --------------------------------------------------------------- input fidelity


def test_fidelity_is_unset_by_default(tmp_path):
    """A host that says nothing about its tensors makes no claim about them."""
    rec = load_dataset_record(_record(tmp_path))
    assert rec.input_fidelity is None
    ws = _workspace(tmp_path, _record(tmp_path))
    assert "Input fidelity" not in (ws / "CLAUDE.md").read_text()


@pytest.mark.parametrize("fidelity", INPUT_FIDELITIES)
def test_fidelity_round_trips(tmp_path, fidelity):
    rec = load_dataset_record(_record(tmp_path, input_fidelity=fidelity))
    assert rec.input_fidelity == fidelity


def test_unrecognized_fidelity_is_refused(tmp_path):
    """Rendering a value nothing here understands would state it as a fact."""
    with pytest.raises(ValueError, match="input_fidelity"):
        load_dataset_record(_record(tmp_path, input_fidelity="real-ish"))


def test_captured_is_rendered_as_evidence(tmp_path):
    ws = _workspace(tmp_path, _record(tmp_path, input_fidelity="captured"))
    claude_md = (ws / "CLAUDE.md").read_text()
    agent = (ws / ".claude" / "agents" / "workload-inspector.md").read_text()
    assert "`captured`" in claude_md and "the model ran on" in claude_md
    assert "`captured`" in agent


def test_shapes_only_says_what_the_values_do_not_settle(tmp_path):
    """The distinction is the point: a generated value is not a workload fact."""
    ws = _workspace(tmp_path, _record(tmp_path, input_fidelity="shapes-only"))
    claude_md = (ws / "CLAUDE.md").read_text()
    agent = (ws / ".claude" / "agents" / "workload-inspector.md").read_text()
    assert "`shapes-only`" in claude_md
    assert "property of the generator" in claude_md
    assert "not a reason to specialize" in claude_md
    assert "synthetic" in agent
