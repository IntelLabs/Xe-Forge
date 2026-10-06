"""Load the vISA knowledge base: doc entries, hand-written examples, compiler corpus.

Layout (``knowledge_base/visa``)::

    common/*.yaml, xe2/*.yaml   entries: id, title, topics, kind (doc|contract), source, body
    examples/index.yaml         hand-written, device-verified kernels (+ .visaasm, Triton twin)
    corpus/index.yaml           IGC output for generic MLIR kernels (tools/visa_corpus)

Every example and corpus entry names a ``family``; corpus entries also list the
ladder families they are ``related`` to. Retrieval uses both for hold-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "knowledge_base" / "visa"


@dataclass(frozen=True)
class DocEntry:
    id: str
    title: str
    topics: tuple[str, ...]
    body: str
    source: str = ""
    kind: str = "doc"  # "doc" | "contract"
    scope: str = "common"  # "common" | "xe2"

    def render(self) -> str:
        return f"### {self.title}\n{self.body.rstrip()}\n"


@dataclass(frozen=True)
class ExampleEntry:
    id: str
    family: str
    topics: tuple[str, ...]
    summary: str
    visa: str
    origin: str  # "hand" | "corpus"
    related: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    triton: str | None = None

    def render(self) -> str:
        what = "hand-written, verified on device" if self.origin == "hand" else "compiler-generated for a generic kernel"
        parts = [f"### Example `{self.id}` ({what}): {self.summary}"]
        if self.triton:
            parts.append("Triton source it implements:\n```python\n" + self.triton.strip() + "\n```")
        parts.append("```\n" + self.visa.strip() + "\n```")
        return "\n".join(parts) + "\n"


@dataclass
class VisaKnowledgeBase:
    docs: list[DocEntry] = field(default_factory=list)
    examples: list[ExampleEntry] = field(default_factory=list)
    corpus: list[ExampleEntry] = field(default_factory=list)
    root: Path | None = None


def _triton_kernel_text(py: Path) -> str | None:
    """The @triton.jit function(s) of an example twin, without the host wrapper."""
    if not py.exists():
        return None
    text = py.read_text()
    start = text.find("@triton.jit")
    end = text.find("\nclass Model")
    return text[start:end].strip() if start >= 0 else None


def load_knowledge(root: str | Path | None = None, dirs: tuple[str, ...] = ("common", "xe2")) -> VisaKnowledgeBase:
    root = Path(root) if root else DEFAULT_ROOT
    kb = VisaKnowledgeBase(root=root)
    for scope in dirs:
        for path in sorted((root / scope).glob("*.yaml")):
            for e in (yaml.safe_load(path.read_text()) or {}).get("entries", []):
                kb.docs.append(DocEntry(
                    id=e["id"], title=e["title"], topics=tuple(e.get("topics", [])), body=e["body"],
                    source=e.get("source", ""), kind=e.get("kind", "doc"), scope=scope,
                ))
    ex_index = root / "examples" / "index.yaml"
    if ex_index.exists():
        for e in yaml.safe_load(ex_index.read_text())["examples"]:
            stem = root / "examples" / e["id"]
            kb.examples.append(ExampleEntry(
                id=e["id"], family=e["family"], topics=tuple(e.get("topics", [])), summary=e["summary"],
                visa=stem.with_suffix(".visaasm").read_text(), origin="hand",
                related=tuple(e.get("related", [])), triton=_triton_kernel_text(stem.with_suffix(".py")),
            ))
    corpus_index = root / "corpus" / "index.yaml"
    if corpus_index.exists():
        for e in yaml.safe_load(corpus_index.read_text())["entries"]:
            kb.corpus.append(ExampleEntry(
                id=e["id"], family=e["family"], topics=tuple(e.get("topics", [])), summary=e["summary"],
                visa=(root / "corpus" / e["visa"]).read_text(), origin="corpus",
                related=tuple(e.get("related", [])), features=tuple(e.get("features", [])),
            ))
    return kb
