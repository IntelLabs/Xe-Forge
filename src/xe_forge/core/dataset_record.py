"""The host's record of the data behind this workspace's benchmark variants.

Xe-Forge measures a kernel at shapes. Where those shapes came from -- which model
ran, how often each combination occurred, how the tensors are distributed -- is a
property of the host's dataset, not of Xe-Forge. A session that cannot see it is
optimizing an average of shapes it was handed, which is how a kernel ends up tuned
for a combination the model never runs.

So a host that has such a dataset attaches a JSON record naming it, instead of
Xe-Forge learning to read any particular dataset format. Xe-Forge renders the
structured fields it understands -- where the data is, which definition it belongs
to, which workload each benchmark variant times, and whether the tensors behind those
workloads are the model's own or generated at its shapes -- plus a markdown fragment the
host wrote saying how to open it. The reading is the host's business; the rules
about what an inspection may and may not settle are Xe-Forge's, and live in the
workspace template beside the rendered facts.

Unset, nothing renders and the workspace is exactly what it was.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_PROFILE_PATH", "INPUT_FIDELITIES", "DatasetRecord", "load_dataset_record"]

# Durable, and one per definition. The dataset does not change under the session,
# so the profile is written once and re-read, not regenerated per trial.
DEFAULT_PROFILE_PATH = "experiments/workload_profile.md"

# What the tensors behind the variants are. "captured" means they were recorded from
# the model this definition came from and are the values it ran on; "shapes-only"
# means they are generated at the recorded shapes, so the shapes are real and the
# values are not.
INPUT_FIDELITIES = ("captured", "shapes-only")


@dataclass(frozen=True)
class DatasetRecord:
    """What the workspace may say about the data behind its variants.

    Every field but ``path`` is optional: a host that can name the dataset and
    nothing else still gives the session more than it had.
    """

    path: str
    definition: str | None = None
    # variant name -> the host's identifier for the workload it times. Xe-Forge
    # does not interpret the identifier; it renders it so a session can ask the
    # host's own tooling about a specific variant rather than matching on shapes.
    variants: dict[str, str] = field(default_factory=dict)
    # Markdown, written by the host, saying how to open the dataset. This is the
    # one part Xe-Forge cannot supply: it names an API, an interpreter and a set
    # of conventions that belong to the project that generated the workspace.
    read_with: str = ""
    profile_path: str = DEFAULT_PROFILE_PATH
    # One of INPUT_FIDELITIES, or None where the host did not say. None renders
    # nothing: a session that is not told must not assume either answer.
    input_fidelity: str | None = None


def load_dataset_record(record_path: str | Path | None) -> DatasetRecord | None:
    """Read a dataset record, or ``None`` when no usable one was named.

    A record that names no dataset is not an error -- a host may write one
    sidecar for several purposes and only some of them describe data. A record
    that cannot be parsed is: the host named a file and got it wrong, and
    silently dropping it would leave the session with a workspace that looks
    complete and says nothing about its own workloads.
    """
    if not record_path:
        return None
    path = Path(record_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset record not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"dataset record {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"dataset record {path} must hold a JSON object")

    dataset = raw.get("dataset")
    if not dataset:
        return None

    variants = raw.get("variants") or {}
    if not isinstance(variants, dict):
        raise ValueError(f"dataset record {path}: 'variants' must map variant name to workload")

    fidelity = raw.get("input_fidelity") or None
    if fidelity is not None and fidelity not in INPUT_FIDELITIES:
        raise ValueError(
            f"dataset record {path}: 'input_fidelity' is {fidelity!r}, expected one of "
            f"{', '.join(INPUT_FIDELITIES)}. An unrecognized value would render as a claim "
            f"about the data that nothing here can check; omit the field instead."
        )

    return DatasetRecord(
        path=str(dataset),
        definition=raw.get("definition") or None,
        variants={str(k): str(v) for k, v in variants.items()},
        read_with=str(raw.get("read_with") or ""),
        profile_path=str(raw.get("profile_path") or DEFAULT_PROFILE_PATH),
        input_fidelity=fidelity,
    )
