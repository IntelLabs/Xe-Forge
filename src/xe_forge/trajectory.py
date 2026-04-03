"""Trajectory collection and storage for CoVeR training data.

A trajectory records one complete optimization attempt on one kernel at one stage.
Trajectories are saved as a side effect of normal pipeline runs when
``--collect-trajectories`` is passed on the CLI.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    """One optimization attempt on one kernel at one stage."""

    # --- Inputs (what the LLM saw) ---
    original_code: str = ""
    current_code: str = ""
    stage: str = ""
    issues: str = ""
    xpu_config: str = ""
    problem_context: str = ""
    performance_context: str = ""
    vtune_report: str = ""
    knowledge_base_context: str = ""
    # Extra fields present only for specific stages
    pytorch_code: str = ""  # algorithmic stage
    suggested_autotune_configs: str = ""  # autotuning stage
    problem_shapes: str = ""  # autotuning stage

    # --- Outputs (what the LLM produced) ---
    optimized_code: str = ""
    next_thought: str = ""  # first reasoning step from CoVeR

    # --- Metadata ---
    success: bool = False
    iterations_used: int = 0
    speedup: Optional[float] = None
    tflops: Optional[float] = None
    kernel_name: str = ""
    kernel_path: str = ""
    spec_path: str = ""

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryRecord":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_trajectories(directory: str | Path, success_only: bool = True) -> List[TrajectoryRecord]:
    """Load all trajectory JSON files from a directory."""
    directory = Path(directory)
    records = []
    for p in sorted(directory.glob("*.json")):
        try:
            rec = TrajectoryRecord.load(p)
            if success_only and not rec.success:
                continue
            records.append(rec)
        except Exception as e:
            logger.warning(f"Failed to load trajectory {p}: {e}")
    logger.info(f"Loaded {len(records)} trajectories from {directory}")
    return records
