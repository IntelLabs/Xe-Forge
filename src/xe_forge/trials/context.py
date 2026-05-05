"""Context object passed to searches and writers during a trial run."""

from dataclasses import dataclass, field
from typing import Any

from xe_forge.config import Config


@dataclass
class TrialContext:
    """Everything a search/writer might need, in one place.

    The runner builds this once at startup and passes the same instance to
    every search.propose() and writer.write() call. Mutable state (like
    profiler_cache) is updated by the runner between trials.
    """

    kernel_name: str
    config: Config
    spec: Any                       # xe_forge.core.spec_loader.Spec
    variant_type: str               # e.g. "bench-gpu"
    input_shapes: dict[str, Any]
    flop: float | None
    dtype: Any                      # torch.dtype
    init_args: list | None
    xpu_config: dict | None
    knowledge_base: Any = None      # xe_forge.knowledge.loader.KnowledgeBase | None
    pytorch_code: str = ""          # empty when baseline is Triton
    pytorch_path: str | None = None
    static_analysis: dict = field(default_factory=dict)  # analyze_pytorch_kernel output

    # Mutable state the runner updates between trials.
    profiler_cache: dict[str, str] = field(default_factory=dict)
    # parent_id -> vtune report (markdown) captured while that trial was the best
