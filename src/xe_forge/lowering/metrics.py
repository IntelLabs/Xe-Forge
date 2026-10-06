"""Research records for lowering runs: one JSON line per run.

Per-attempt lines are written by ``visa-verify`` (``metrics.jsonl`` in the run
directory) as each attempt is evaluated.

The fields answer the experiment's questions directly: did it compile, run and
verify; after how many attempts and how much time; how many tokens; how the
result performs; which knowledge the prompt contained.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any



@dataclass
class LoweringRunRecord:
    run_id: str
    kernel: str
    family: str | None
    target: str
    mode: str
    mode_letter: str
    model: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    attempts: int = 0
    correct: bool = False
    attempts_to_valid: int | None = None
    time_to_valid_s: float | None = None
    categories: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, Any] = field(default_factory=dict)
    best_speedup: float | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    grf: int | None = None
    spills: int | None = None
    binary_size: int | None = None
    knowledge_chars: int = 0
    doc_ids: list[str] = field(default_factory=list)
    example_ids: list[str] = field(default_factory=list)
    corpus_ids: list[str] = field(default_factory=list)
    held_out: list[str] = field(default_factory=list)
    screened_out: dict[str, float] = field(default_factory=dict)
    contract_chars: int = 0
    stopped: str = ""
    error: str = ""


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record) if hasattr(record, "__dataclass_fields__") else record, default=str) + "\n")
