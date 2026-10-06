"""Choose the vISA knowledge one kernel's prompt receives.

Knowledge modes (the experiment's independent variable):

    none            A   the contract only
    contracts       B'  rules only (entries with kind: contract)
    docs            B   all documentation entries
    docs+examples   C   docs + hand-written verified examples
    docs+corpus     D   docs + compiler-generated corpus of generic kernels
    all             C+D docs + both kinds of examples

Selection is driven by what the kernel uses: ``tl.dot`` pulls DPAS, loads and
stores pull LSC and addressing, masks pull predicates, reductions pull the
cross-lane material. Examples are ranked by topic overlap and held out when they
belong to -- or are related to -- the target's family. Corpus examples are further
screened against the target's own compiler output (which exists only in a sealed
directory), so a generic kernel that happens to compile to nearly the same vISA is
never shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from xe_forge.lowering.context import SemanticSummary
from xe_forge.lowering.visa.knowledge import DocEntry, ExampleEntry, VisaKnowledgeBase

MODES = ("none", "contracts", "docs", "docs+examples", "docs+corpus", "all")
MODE_LETTER = {"none": "A", "contracts": "B'", "docs": "B", "docs+examples": "C", "docs+corpus": "D", "all": "C+D"}

ALWAYS = {"syntax", "declarations", "kernel_args_abi", "exec_widths"}
# Topics nearly every kernel touches; they do not distinguish one example from another.
GENERIC = {"arithmetic_fp", "arithmetic_int", "control_flow", "lsc", "memory_addressing", "cache_controls"}

OP_TOPICS = {
    "dot": {"dpas", "bf16_fp16", "grf"},
    "load": {"lsc", "memory_addressing", "cache_controls"},
    "store": {"lsc", "memory_addressing"},
    "where": {"predicates_execmask"},
    "sum": {"reduction", "simd", "predicates_execmask", "regions"},
    "max": {"reduction", "simd", "regions", "arithmetic_fp"},
    "min": {"reduction", "simd", "regions", "arithmetic_fp"},
    "reduce": {"reduction", "simd", "regions"},
    "exp": {"math"},
    "exp2": {"math"},
    "log": {"math"},
    "sqrt": {"math"},
    "rsqrt": {"math"},
    "sigmoid": {"math"},
    "maximum": {"arithmetic_fp"},
    "minimum": {"arithmetic_fp"},
    "program_id": {"kernel_args_abi"},
    "arange": {"kernel_args_abi"},
    "zeros": {"arithmetic_fp"},
}


def wanted_topics(semantics: SemanticSummary) -> set[str]:
    topics = set(ALWAYS)
    for op in semantics.ops:
        topics |= OP_TOPICS.get(op.split(".")[-1], set())
    if semantics.masks:
        topics |= {"predicates_execmask"}
    if semantics.dtype_casts:
        topics |= {"conversions", "bf16_fp16"}
    if semantics.has_reduction:
        topics |= {"reduction", "simd", "regions"}
    if semantics.has_transcendental:
        topics |= {"math"}
    if semantics.has_dot:
        topics |= {"dpas"}
    topics |= {"arithmetic_fp", "arithmetic_int", "control_flow"}
    return topics


# -- similarity screen ----------------------------------------------------------

_VAR = re.compile(r"\bV\d+\b|\bP\d+\b|\b_\w+\b")


def _shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    lines = []
    for line in text.splitlines():
        line = line.split("///")[0].split("//")[0].strip()
        if not line or line.startswith((".decl", ".input", ".kernel", ".version", ".function")):
            continue
        lines.append(_VAR.sub("V", re.sub(r"\s+", " ", line)))
    return {tuple(lines[i : i + n]) for i in range(max(0, len(lines) - n + 1))}


def containment(candidate: str, target: str, n: int = 4) -> float:
    """Fraction of the candidate's instruction 4-grams that also occur in the target."""
    a = _shingles(candidate, n)
    if not a:
        return 0.0
    return len(a & _shingles(target, n)) / len(a)


@dataclass
class Retrieved:
    mode: str
    text: str
    doc_ids: list[str] = field(default_factory=list)
    example_ids: list[str] = field(default_factory=list)
    corpus_ids: list[str] = field(default_factory=list)
    held_out: list[str] = field(default_factory=list)
    screened_out: dict[str, float] = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)


class VisaRetriever:
    def __init__(
        self,
        kb: VisaKnowledgeBase,
        *,
        char_budget: int = 60000,
        doc_share: float = 0.5,
        max_examples: int = 2,
        max_corpus: int = 2,
        screen_threshold: float = 0.5,
    ):
        self.kb = kb
        self.char_budget = char_budget
        self.doc_share = doc_share
        self.max_examples = max_examples
        self.max_corpus = max_corpus
        self.screen_threshold = screen_threshold

    def retrieve(
        self,
        semantics: SemanticSummary,
        mode: str,
        *,
        family: str | None = None,
        target_visa: list[str] | None = None,
    ) -> Retrieved:
        if mode not in MODES:
            raise ValueError(f"unknown knowledge mode {mode!r}; one of {MODES}")
        out = Retrieved(mode=mode, text="")
        if mode == "none":
            return out
        topics = wanted_topics(semantics)
        with_examples = mode in ("docs+examples", "docs+corpus", "all")
        # Leave room for examples when the mode includes them.
        budget = int(self.char_budget * (self.doc_share if with_examples else 1.0))
        parts: list[str] = []

        docs = [d for d in self.kb.docs if mode != "contracts" or d.kind == "contract"]
        for d in sorted(docs, key=lambda d: (-self._doc_score(d, topics), d.id)):
            if self._doc_score(d, topics) <= 0:
                continue
            text = d.render()
            if len(text) > budget:
                continue
            parts.append(text)
            out.doc_ids.append(d.id)
            budget -= len(text)

        budget += self.char_budget - int(self.char_budget * (self.doc_share if with_examples else 1.0))
        pools: list[tuple[list[ExampleEntry], int, list[str]]] = []
        if mode in ("docs+examples", "all"):
            pools.append((self.kb.examples, self.max_examples, out.example_ids))
        if mode in ("docs+corpus", "all"):
            pools.append((self.kb.corpus, self.max_corpus, out.corpus_ids))
        for pool, limit, chosen in pools:
            for ex in sorted(pool, key=lambda e: (-self._example_score(e, topics), len(e.visa), e.id)):
                if len(chosen) >= limit:
                    break
                if family and (ex.family == family or family in ex.related):
                    out.held_out.append(ex.id)
                    continue
                if ex.origin == "corpus" and target_visa:
                    score = max(containment(ex.visa, t) for t in target_visa)
                    if score > self.screen_threshold:
                        out.screened_out[ex.id] = round(score, 3)
                        continue
                text = ex.render()
                if len(text) > budget:
                    continue
                parts.append(text)
                chosen.append(ex.id)
                budget -= len(text)
        out.text = "\n".join(parts)
        return out

    @staticmethod
    def _example_score(ex: ExampleEntry, topics: set[str]) -> int:
        """Topics particular to this kernel count three times the ones every kernel has."""
        specific = topics - ALWAYS - GENERIC
        mine = set(ex.topics)
        return 3 * len(mine & specific) + len(mine & topics)

    @staticmethod
    def _doc_score(d: DocEntry, topics: set[str]) -> int:
        overlap = len(set(d.topics) & topics)
        if d.kind == "contract":
            overlap += 2  # rules come before descriptions
        return overlap
