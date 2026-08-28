"""
Knowledge Base Loader — loads optimization patterns from YAML files.

Key design:
- Entirely optional: if no KB directory is provided or found, everything
  falls back gracefully to LLM built-in knowledge.
- Stage-scoped: format_for_stage() returns only what is relevant to the
  requested stage, keeping context windows lean.
- Robust parser: handles all field variants found in the KB YAML files,
  including entries with severity instead of stage, stream_k → persistent_kernel
  remapping, notes fields, and correctness-only constraint files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from xe_forge.models import OptimizationStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage aliases
# ---------------------------------------------------------------------------
_STAGE_ALIASES: dict[str, str] = {
    "memory": "memory_access",
    "memory_patterns": "memory_access",
    "block_ptr": "block_pointers",
    "block_pointer": "block_pointers",
    "dtype": "dtype_fix",
    "dtypes": "dtype_fix",
    "fuse": "fusion",
    "persist": "persistent_kernel",
    "persistent": "persistent_kernel",
    "xpu_specific": "device_specific",
    "cuda": "device_specific",
    "nvidia": "device_specific",
    "sycl": "device_specific",
    "gemm": "device_specific",
    "autotune": "autotuning",
    "stream_k": "persistent_kernel",
    "discovery": "discovery",
    "open_ended": "discovery",
}

_CONSTRAINT_STAGE_HINTS: dict[str, OptimizationStage] = {
    "streamk": OptimizationStage.PERSISTENT_KERNEL,
    "stream_k": OptimizationStage.PERSISTENT_KERNEL,
    "int64": OptimizationStage.DEVICE_SPECIFIC,
    "descriptor": OptimizationStage.BLOCK_POINTERS,
    "block_ptr": OptimizationStage.BLOCK_POINTERS,
    "boundary_check": OptimizationStage.BLOCK_POINTERS,
    "autotune": OptimizationStage.AUTOTUNING,
    "fuse": OptimizationStage.FUSION,
    "dtype": OptimizationStage.DTYPE_FIX,
    "device_to_host": OptimizationStage.MEMORY_ACCESS,
    "contiguous": OptimizationStage.MEMORY_ACCESS,
    "tl_multiple_of": OptimizationStage.MEMORY_ACCESS,
    "repack": OptimizationStage.DEVICE_SPECIFIC,
    "gemm2": OptimizationStage.DEVICE_SPECIFIC,
    "packed_weight": OptimizationStage.DEVICE_SPECIFIC,
    "grid": OptimizationStage.DEVICE_SPECIFIC,
    "grf": OptimizationStage.DEVICE_SPECIFIC,
    "sigmoid": OptimizationStage.DEVICE_SPECIFIC,
    "warp_sweep": OptimizationStage.DEVICE_SPECIFIC,
    "warp_count": OptimizationStage.DEVICE_SPECIFIC,
    "exp2": OptimizationStage.DEVICE_SPECIFIC,
    "open_ended": OptimizationStage.DISCOVERY,
    "discovery": OptimizationStage.DISCOVERY,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    id: str
    name: str
    stage: OptimizationStage
    pattern_before: str
    pattern_after: str
    description: str = ""
    rationale: str = ""
    expected_speedup: str | None = None
    applies_to: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    notes: str = ""
    examples: list[dict] = field(default_factory=list)


@dataclass
class KnowledgeConstraint:
    id: str
    name: str
    description: str
    severity: str = "critical"
    stages: list[OptimizationStage] = field(default_factory=list)


@dataclass
class KnowledgeExample:
    id: str
    name: str
    description: str
    stages: list[OptimizationStage]
    optimizations_applied: list[str] = field(default_factory=list)
    expected_speedup: str = ""
    unoptimized_code: str = ""
    optimized_code: str = ""


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    def __init__(self) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}
        self._by_stage: dict[OptimizationStage, list[KnowledgeEntry]] = {
            s: [] for s in OptimizationStage
        }
        self._constraints: list[KnowledgeConstraint] = []
        self._examples: dict[str, KnowledgeExample] = {}
        self.skipped: list[dict] = []

    def add_entry(self, entry: KnowledgeEntry) -> None:
        self._entries[entry.id] = entry
        self._by_stage[entry.stage].append(entry)

    def add_constraint(self, constraint: KnowledgeConstraint) -> None:
        self._constraints.append(constraint)

    def add_example(self, example: KnowledgeExample) -> None:
        self._examples[example.id] = example

    def get_by_stage(self, stage: OptimizationStage) -> list[KnowledgeEntry]:
        return self._by_stage.get(stage, [])

    def constraints_for_stage(self, stage: OptimizationStage) -> list[KnowledgeConstraint]:
        return [c for c in self._constraints if not c.stages or stage in c.stages]

    def examples_for_stage(self, stage: OptimizationStage) -> list[KnowledgeExample]:
        return [e for e in self._examples.values() if stage in e.stages]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    @property
    def example_count(self) -> int:
        return len(self._examples)

    def format_for_stage(self, stage: OptimizationStage) -> str:
        """
        Return a context string containing only what is relevant for *stage*.

        Section order:
        1. Stage-scoped critical constraints
        2. Optimization patterns (before/after pairs)
        3. Full code examples
        """
        parts: list[str] = []

        # 1. Constraints
        constraints = self.constraints_for_stage(stage)
        if constraints:
            lines = [
                "=" * 60,
                f"CRITICAL CONSTRAINTS FOR {stage.value.upper()}",
                "=" * 60,
                "",
            ]
            for c in constraints:
                lines += [
                    f"### {c.name}",
                    f"Severity: {c.severity}",
                    "",
                    c.description.strip(),
                    "",
                ]
            parts.append("\n".join(lines))

        # 2. Patterns
        entries = self.get_by_stage(stage)
        if entries:
            lines = [
                f"OPTIMIZATION PATTERNS FOR {stage.value.upper()}",
                "=" * 60,
            ]
            for entry in entries:
                lines += [
                    f"\n## {entry.name}",
                    f"Description: {entry.description}",
                    f"Rationale: {entry.rationale.strip()}",
                ]
                if entry.pattern_before.strip():
                    lines += ["", "### Before:", "```python", entry.pattern_before.strip(), "```"]
                if entry.pattern_after.strip():
                    lines += ["", "### After:", "```python", entry.pattern_after.strip(), "```"]
                if entry.expected_speedup:
                    lines.append(f"\nExpected speedup: {entry.expected_speedup}")
                if entry.notes.strip():
                    lines += ["", f"Notes: {entry.notes.strip()}"]
                if entry.examples:
                    lines.append("\n### Inline examples:")
                    for i, ex in enumerate(entry.examples, 1):
                        if "before" in ex or "after" in ex:
                            lines.append(f"\nExample {i}:")
                        if "before" in ex:
                            lines += ["Before:", "```python", ex["before"].strip(), "```"]
                        if "after" in ex:
                            lines += ["After:", "```python", ex["after"].strip(), "```"]
                lines.append("")
            parts.append("\n".join(lines))
        else:
            parts.append(f"No YAML patterns loaded for {stage.value} — relying on LLM knowledge.")

        # 3. Full code examples
        examples = self.examples_for_stage(stage)
        logger.debug(
            "format_for_stage(%s): %d constraints, %d patterns, %d examples",
            stage.value,
            len(constraints),
            len(entries),
            len(examples),
        )
        if examples:
            lines = [
                "=" * 60,
                f"FULL CODE EXAMPLES FOR {stage.value.upper()}",
                "=" * 60,
                "",
            ]
            for ex in examples:
                lines += [f"## {ex.name}", f"Description: {ex.description.strip()}", ""]
                if ex.optimizations_applied:
                    lines.append("Optimizations applied:")
                    for opt in ex.optimizations_applied:
                        lines.append(f"  - {opt}")
                if ex.expected_speedup:
                    lines.append(f"Expected speedup: {ex.expected_speedup}")
                if ex.unoptimized_code:
                    lines += [
                        "",
                        "### Unoptimized:",
                        "```python",
                        ex.unoptimized_code.strip(),
                        "```",
                    ]
                if ex.optimized_code:
                    lines += ["", "### Optimized:", "```python", ex.optimized_code.strip(), "```"]
                lines.append("")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def summary(self) -> str:
        by_stage = {s.value: len(e) for s, e in self._by_stage.items() if e}
        return (
            f"{self.entry_count} patterns ({by_stage}), "
            f"{self.constraint_count} constraints, "
            f"{self.example_count} examples"
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_knowledge_base(
    knowledge_dir: str | Path,
    dsl: str = "triton",
    device_type: str = "xpu",
) -> KnowledgeBase:
    kb = KnowledgeBase()
    kp = Path(knowledge_dir)

    if not kp.exists():
        logger.warning("Knowledge directory not found: %s — KB disabled", kp)
        return kb

    yaml_files = _collect_yaml_files(kp, dsl, device_type)
    if not yaml_files:
        logger.warning("No YAML files found in %s for dsl=%s device=%s", kp, dsl, device_type)
        return kb

    for yf in yaml_files:
        _load_yaml_file(kb, yf)

    for examples_dir in _collect_examples_dirs(kp, dsl, device_type):
        _load_examples(kb, examples_dir)

    logger.info("Knowledge base loaded (dsl=%s, device=%s): %s", dsl, device_type, kb.summary())

    if kb.skipped:
        logger.warning("Skipped %d patterns (unknown/unmappable stage):", len(kb.skipped))
        for s in kb.skipped[:5]:
            logger.warning("  %s / %s  stage=%r", s["file"], s.get("id", "?"), s.get("stage"))
        if len(kb.skipped) > 5:
            logger.warning("  … and %d more", len(kb.skipped) - 5)

    return kb


def _collect_yaml_files(kp: Path, dsl: str, device_type: str) -> list[Path]:
    """Collect YAML files in priority order: common → dsl-common → dsl/device.

    Falls back to flat *.yaml in the root if no subdirectory structure is found.
    """
    common_dir = kp / "common"
    dsl_dir = kp / dsl
    device_dir = dsl_dir / device_type

    has_subdirs = common_dir.is_dir() or dsl_dir.is_dir()

    if not has_subdirs:
        return sorted(kp.glob("*.yaml")) + sorted(kp.glob("*.yml"))

    files: list[Path] = []
    if common_dir.is_dir():
        files.extend(sorted(common_dir.glob("*.yaml")) + sorted(common_dir.glob("*.yml")))
    dsl_common = dsl_dir / "common"
    if dsl_common.is_dir():
        files.extend(sorted(dsl_common.glob("*.yaml")) + sorted(dsl_common.glob("*.yml")))
    if device_dir.is_dir():
        files.extend(sorted(device_dir.glob("*.yaml")) + sorted(device_dir.glob("*.yml")))
    return files


def _collect_examples_dirs(kp: Path, dsl: str, device_type: str) -> list[Path]:
    """Return examples directories that exist, in priority order."""
    candidates = [
        kp / "common" / "examples",
        kp / dsl / "common" / "examples",
        kp / dsl / device_type / "examples",
        kp / "examples",  # legacy flat layout
    ]
    return [d for d in candidates if d.is_dir()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml_file(kb: KnowledgeBase, path: Path) -> None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return

    if not data or not isinstance(data, dict):
        return

    fname = path.name

    for raw in data.get("constraints", []):
        c = _parse_constraint(raw, fname)
        if c:
            kb.add_constraint(c)

    for raw in data.get("patterns", []):
        if not isinstance(raw, dict):
            continue
        entry = _parse_entry(raw, fname)
        if entry:
            kb.add_entry(entry)
        else:
            kb.skipped.append(
                {
                    "file": fname,
                    "id": raw.get("id", "?"),
                    "name": raw.get("name", "?"),
                    "stage": raw.get("stage", "?"),
                }
            )


def _parse_constraint(data: dict, source: str) -> KnowledgeConstraint | None:
    try:
        cid = data.get("id", "")
        stages = _infer_constraint_stages(cid)
        return KnowledgeConstraint(
            id=cid,
            name=data.get("name", cid),
            description=data.get("description", ""),
            severity=data.get("severity", "critical"),
            stages=stages,
        )
    except Exception as exc:
        logger.debug("Failed to parse constraint in %s: %s", source, exc)
        return None


def _infer_constraint_stages(cid: str) -> list[OptimizationStage]:
    cid_lower = cid.lower()
    matched = [stage for keyword, stage in _CONSTRAINT_STAGE_HINTS.items() if keyword in cid_lower]
    seen: set[OptimizationStage] = set()
    result: list[OptimizationStage] = []
    for s in matched:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _parse_entry(data: dict, source: str) -> KnowledgeEntry | None:
    if "severity" in data and "stage" not in data:
        return None

    stage_str = data.get("stage", "")
    if not stage_str:
        return None

    stage = _normalize_stage(stage_str)
    if stage is None:
        return None

    return KnowledgeEntry(
        id=data.get("id", f"entry_{hash(str(data))}"),
        name=data.get("name", "Unnamed Pattern"),
        stage=stage,
        pattern_before=data.get("pattern_before", data.get("before", "")),
        pattern_after=data.get("pattern_after", data.get("after", "")),
        description=data.get("description", ""),
        rationale=data.get("rationale", ""),
        expected_speedup=data.get("expected_speedup"),
        applies_to=data.get("applies_to", []),
        prerequisites=data.get("prerequisites", []),
        notes=str(data.get("notes", "")),
        examples=data.get("examples", []),
    )


def _normalize_stage(stage_str: str) -> OptimizationStage | None:
    if not stage_str:
        return None
    s = stage_str.strip().lower()
    if s in _STAGE_ALIASES:
        s = _STAGE_ALIASES[s]
    try:
        return OptimizationStage(s)
    except ValueError:
        pass
    for stage in OptimizationStage:
        if s in stage.value or stage.value in s:
            return stage
    logger.debug("Cannot map stage %r to any OptimizationStage", stage_str)
    return None


def _load_examples(kb: KnowledgeBase, examples_dir: Path) -> None:
    """Load full code examples from the examples/ subdirectory."""
    index_file = examples_dir / "index.yaml"
    if not index_file.exists():
        logger.debug("No examples/index.yaml found at %s", examples_dir)
        return

    try:
        with open(index_file) as f:
            index: dict = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to read examples index: %s", exc)
        return

    for raw in index.get("critical_constraints", []):
        c = _parse_constraint(raw, "examples/index.yaml")
        if c:
            kb.add_constraint(c)

    loaded = 0
    for ex_meta in index.get("examples", []):
        ex_id = ex_meta.get("id", "")
        if not ex_id:
            continue

        # Prefer explicit stages: field; fall back to keyword inference
        if "stages" in ex_meta:
            stages: list[OptimizationStage] = []
            seen_s: set[OptimizationStage] = set()
            for s in ex_meta["stages"]:
                stage = _normalize_stage(str(s))
                if stage is not None and stage not in seen_s:
                    seen_s.add(stage)
                    stages.append(stage)
            if not stages:
                stages = _infer_example_stages(ex_meta)
        else:
            stages = _infer_example_stages(ex_meta)

        example = KnowledgeExample(
            id=ex_id,
            name=ex_meta.get("name", ex_id),
            description=ex_meta.get("description", ""),
            stages=stages,
            optimizations_applied=ex_meta.get("optimizations_applied", []),
            expected_speedup=ex_meta.get("expected_speedup", ""),
        )

        # Load code files.
        # Priority: unoptimized → optimized_code, optimized → optimized_code, file → optimized_code
        # A "file" key means there is only one file (the optimized version).
        unopt_fname = ex_meta.get("unoptimized")
        opt_fname = ex_meta.get("optimized") or ex_meta.get("file")

        if unopt_fname:
            fp = examples_dir / unopt_fname
            if fp.exists():
                try:
                    example.unoptimized_code = fp.read_text()
                except Exception as exc:
                    logger.warning("Could not read %s: %s", fp, exc)

        if opt_fname:
            fp = examples_dir / opt_fname
            if fp.exists():
                try:
                    example.optimized_code = fp.read_text()
                except Exception as exc:
                    logger.warning("Could not read %s: %s", fp, exc)

        kb.add_example(example)
        loaded += 1
        logger.debug(
            "Loaded example %s for stages %s (unopt=%s opt=%s)",
            ex_id,
            [s.value for s in stages],
            bool(example.unoptimized_code),
            bool(example.optimized_code),
        )

    logger.info(
        "Loaded %d/%d examples from %s", loaded, len(index.get("examples", [])), examples_dir
    )


_EXAMPLE_STAGE_KEYWORDS: list[tuple[str, OptimizationStage]] = [
    ("block pointer", OptimizationStage.BLOCK_POINTERS),
    ("make_block_ptr", OptimizationStage.BLOCK_POINTERS),
    ("tensor descriptor", OptimizationStage.BLOCK_POINTERS),
    ("make_tensor_descriptor", OptimizationStage.BLOCK_POINTERS),
    ("stream k", OptimizationStage.PERSISTENT_KERNEL),
    ("stream_k", OptimizationStage.PERSISTENT_KERNEL),
    ("persistent", OptimizationStage.PERSISTENT_KERNEL),
    ("autotune", OptimizationStage.AUTOTUNING),
    ("fusion", OptimizationStage.FUSION),
    ("fuse", OptimizationStage.FUSION),
    ("swizzl", OptimizationStage.DEVICE_SPECIFIC),
    ("grf_mode", OptimizationStage.DEVICE_SPECIFIC),
    ("warp", OptimizationStage.DEVICE_SPECIFIC),
    ("tile", OptimizationStage.DEVICE_SPECIFIC),
    ("dtype", OptimizationStage.DTYPE_FIX),
    ("float16", OptimizationStage.DTYPE_FIX),
    ("bfloat16", OptimizationStage.DTYPE_FIX),
    ("memory", OptimizationStage.MEMORY_ACCESS),
    ("coalesc", OptimizationStage.MEMORY_ACCESS),
    ("liveness", OptimizationStage.MEMORY_ACCESS),
    ("attention", OptimizationStage.DEVICE_SPECIFIC),
    ("flash", OptimizationStage.DEVICE_SPECIFIC),
    ("exp2", OptimizationStage.DEVICE_SPECIFIC),
    ("sigmoid", OptimizationStage.DEVICE_SPECIFIC),
    ("gelu", OptimizationStage.FUSION),
]


def _infer_example_stages(meta: dict) -> list[OptimizationStage]:
    opts = []
    for item in meta.get("optimizations_applied", []):
        if isinstance(item, str):
            opts.append(item)
        elif isinstance(item, dict):
            opts.extend(str(v) for v in item.values())

    text = " ".join(
        [
            meta.get("description", ""),
            " ".join(opts),
            meta.get("name", ""),
        ]
    ).lower()

    seen: set[OptimizationStage] = set()
    result: list[OptimizationStage] = []
    for keyword, stage in _EXAMPLE_STAGE_KEYWORDS:
        if keyword in text and stage not in seen:
            seen.add(stage)
            result.append(stage)

    if not result:
        result.append(OptimizationStage.DEVICE_SPECIFIC)

    return result
