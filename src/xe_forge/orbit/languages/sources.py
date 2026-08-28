"""Multi-framework SYCL source registry: resolves kernels to files in checked-out
source trees (torch-xpu-ops, vllm-xpu-kernels, sgl-kernel-xpu, IPEX); absent trees are
reported as absent rather than silently reducing coverage. Design rationale:
docs/DESIGN.md."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Kernel source locations are per-framework knowledge, kept as YAML data files.
KNOWLEDGE_DIR = Path("knowledge_base/common")
FRAMEWORK_GLOB = "framework_*.yaml"


def load_known_trees(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Read kernel source locations from the framework knowledge files, falling back
    to the built-in table when the knowledge base is not on disk."""
    import yaml

    directory = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
    if not directory.is_dir():
        for parent in Path(__file__).resolve().parents:
            candidate = parent / KNOWLEDGE_DIR
            if candidate.is_dir():
                directory = candidate
                break

    if not directory.is_dir():
        return dict(_BUILTIN_SOURCE_TREES)

    trees: dict[str, dict] = {}
    for path in sorted(directory.glob(FRAMEWORK_GLOB)):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        for entry in document.get("kernel_sources") or []:
            name = entry.get("name")
            if not name or name in trees:
                continue
            trees[name] = {
                "repo": entry.get("repo", ""),
                "provider": entry.get("provider", "sycl"),
                "registers": entry.get("registers", ""),
                "subdir": entry.get("subdir", "src"),
                "framework": document.get("framework", path.stem),
                "namespaces": list(entry.get("namespaces") or []),
                # A tree is identified by URL AND revision; an empty pin is legal but
                # reported as unpinned, never silently treated as matching.
                "revision": str(entry.get("revision") or ""),
            }

    return trees or dict(_BUILTIN_SOURCE_TREES)


# Fallback used only when no knowledge base is reachable (wheel-only installs).
_BUILTIN_SOURCE_TREES = {
    "torch-xpu-ops": {
        "repo": "https://github.com/intel/torch-xpu-ops",
        "provider": "sycl",
        "registers": "aten::* on the XPU dispatch key",
        "subdir": "src",
    },
    "vllm-xpu-kernels": {
        "repo": "https://github.com/vllm-project/vllm-xpu-kernels",
        "provider": "custom",
        "registers": "torch.ops._C.* / _moe_C.*",
        "subdir": "src",
    },
    "sgl-kernel-xpu": {
        "repo": "https://github.com/sgl-project/sgl-kernel-xpu",
        "provider": "custom",
        "registers": "torch.ops.sgl_kernel.*",
        "subdir": "src",
    },
    "intel-extension-for-pytorch": {
        "repo": "https://github.com/intel/intel-extension-for-pytorch",
        "provider": "ipex",
        "registers": "torch.ops.torch_ipex.*",
        "subdir": "csrc",
    },
}

# Resolved once at import from the knowledge base, with the built-in table as fallback.
KNOWN_SOURCE_TREES = load_known_trees()

# Where to look for checked-out trees, in priority order. The environment variable wins
# so a user can point at a workspace without moving anything.
SOURCE_ROOT_ENV = "ORBIT_SYCL_SOURCES"
DEFAULT_SOURCE_ROOTS = (
    "~/.cache/orbit-dev",
    "~/src",
    "/opt/intel-sources",
)


@dataclass
class SourceTree:
    name: str
    path: Path
    provider: str
    registers: str
    # Which framework's knowledge file declared this tree.
    framework: str = ""
    # C++ namespaces this tree owns; used to break ties between trees.
    namespaces: list[str] = field(default_factory=list)
    symbols: dict[str, Path] = field(default_factory=dict)
    # The revision the knowledge file pins (matching the installed wheel) and the
    # revision actually checked out; skew between them is surfaced, not discovered
    # later as a compile error.
    revision: str = ""
    checkout_revision: str = ""

    def claims(self, demangled: str) -> bool:
        """Whether this tree's namespace appears in the demangled symbol."""
        return any(f"{ns}::" in demangled for ns in self.namespaces)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def pin_state(self) -> str:
        """One of: unpinned | ok | skew | unverified. `unverified` means a pin is
        declared but the checkout's revision could not be read."""
        if not self.revision:
            return "unpinned"
        if not self.checkout_revision:
            return "unverified"
        short = min(len(self.revision), len(self.checkout_revision))
        if self.revision[:short].lower() == self.checkout_revision[:short].lower():
            return "ok"
        return "skew"


@dataclass
class SourceRegistry:
    """Every Intel kernel source tree Orbit can currently see."""

    trees: list[SourceTree] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def lookup(self, symbol: str, demangled: str = "") -> tuple[Path, SourceTree] | None:
        """Find the tree defining `symbol`, preferring the one whose declared
        namespace appears in the demangled name — identifiers collide across trees."""
        hits = [(tree.symbols[symbol], tree) for tree in self.trees if symbol in tree.symbols]
        if not hits:
            return None
        if demangled and len(hits) > 1:
            for path, tree in hits:
                if tree.claims(demangled):
                    return path, tree
        return hits[0]

    def resolve(
        self, demangled: str, agent: object | None = None
    ) -> tuple[Path, SourceTree, str] | None:
        """Find the file defining the most specific identifier in a kernel name.

        The exact symbol index is tried first; when it misses, the question goes to
        a `RepoAgent` rather than to a looser pattern.
        """
        from xe_forge.orbit.languages.sycl_backend import identifying_symbols

        for symbol in identifying_symbols(demangled):
            found = self.lookup(symbol, demangled)
            if found is not None:
                path, tree = found
                return path, tree, symbol

        if agent is None:
            return None
        return self.resolve_with_agent(demangled, agent)

    def resolve_with_agent(
        self, demangled: str, agent: object
    ) -> tuple[Path, SourceTree, str] | None:
        """Ask a RepoAgent which file defines this kernel, and verify the answer:
        the file must exist and sit inside one of the indexed trees."""
        from xe_forge.orbit.agents.base import AgentTask

        for tree in self.trees:
            task = AgentTask(
                question=(
                    "Find the source file that DEFINES the SYCL kernel or functor in "
                    f"this compiled symbol: {demangled}"
                ),
                search_root=tree.path,
                context={"framework": tree.name, "registers": tree.registers},
            )
            try:
                answer = agent.ask(task)
            except Exception:
                # An unreachable provider must not abort resolution.
                continue

            if not answer.resolved or answer.value is None:
                continue

            path = Path(answer.value)
            try:
                path.relative_to(tree.path)
            except ValueError:
                continue

            symbol = next(
                (e for e in answer.evidence if e and "/" not in e and ":" not in e),
                demangled,
            )
            return path, tree, symbol
        return None

    @property
    def total_symbols(self) -> int:
        return sum(t.symbol_count for t in self.trees)

    def format(self) -> str:
        lines = [
            f"{'SOURCE TREE':<30} {'SYMBOLS':>8}  {'PIN':<10} {'FRAMEWORK':<14} REGISTERS",
            "-" * 96,
        ]
        for tree in self.trees:
            lines.append(
                f"{tree.name:<30} {tree.symbol_count:>8}  {tree.pin_state:<10} "
                f"{(tree.framework or tree.provider):<14} {tree.registers}"
            )
        for name in self.missing:
            meta = KNOWN_SOURCE_TREES[name]
            lines.append(f"{name:<30} {'absent':>8}  {'-':<10} {'-':<14} clone {meta['repo']}")
        lines.append("-" * 88)
        lines.append(
            f"{len(self.trees)} tree(s) indexed, {self.total_symbols} symbols; "
            f"{len(self.missing)} absent"
        )
        if self.missing:
            lines.append(
                "An absent tree is not a kernel without source — it is source we have not "
                "been shown. Those two are reported differently on purpose."
            )
        for tree in self.trees:
            if tree.pin_state == "skew":
                lines.append(
                    f"SKEW: {tree.name} is pinned to {tree.revision} but the checkout is at "
                    f"{tree.checkout_revision[:12]} — a bundle built from it is a different "
                    f"kernel than the one the wheel runs. Check out the pinned "
                    f"revision before extracting."
                )
            elif tree.pin_state == "unpinned":
                lines.append(
                    f"UNPINNED: {tree.name} declares no revision in its knowledge file. "
                    f"A tree is identified by URL *and* revision; add `revision:` "
                    f"to its kernel_sources entry, matched to the installed wheel."
                )
            elif tree.pin_state == "unverified":
                lines.append(
                    f"UNVERIFIED: {tree.name} is pinned to {tree.revision} but the "
                    f"checkout's revision could not be read; the pin was not confirmed."
                )
        return "\n".join(lines)


def checkout_revision(checkout: Path) -> str:
    """The revision a checkout is actually at, or "" when it cannot be read.

    Read from `.git` directly so no git binary is needed. Handles a detached HEAD,
    a symbolic ref, packed refs, and a `.git` file (worktrees).
    """
    git = checkout / ".git"
    try:
        if git.is_file():
            # A worktree: `.git` is a file naming the real git directory.
            content = git.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return ""
            git = Path(content.split(":", 1)[1].strip())
            if not git.is_absolute():
                git = (checkout / git).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD: the hash itself
        ref = head.split(":", 1)[1].strip()
        ref_file = git / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.endswith(ref) and " " in line:
                    return line.split(" ", 1)[0]
    except OSError:
        return ""
    return ""


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get(SOURCE_ROOT_ENV)
    if override:
        roots.extend(Path(p).expanduser() for p in override.split(os.pathsep) if p)
    roots.extend(Path(p).expanduser() for p in DEFAULT_SOURCE_ROOTS)
    return [r for r in roots if r.is_dir()]


def discover(roots: list[Path] | None = None) -> SourceRegistry:
    """Index whichever known kernel trees are checked out under the search roots."""
    from xe_forge.orbit.languages.sycl_backend import index_source_tree

    registry = SourceRegistry()
    search = roots if roots is not None else candidate_roots()

    for name, meta in KNOWN_SOURCE_TREES.items():
        located: Path | None = None
        for root in search:
            candidate = root / name
            if candidate.is_dir():
                located = candidate
                break
        if located is None:
            registry.missing.append(name)
            continue

        # Index the kernel subdirectory when it exists; some trees keep sources at the
        # root, and indexing the whole checkout would sweep in tests and third-party code.
        target = located / str(meta["subdir"])
        if not target.is_dir():
            target = located

        registry.trees.append(
            SourceTree(
                name=name,
                path=target,
                provider=str(meta["provider"]),
                registers=str(meta["registers"]),
                framework=str(meta.get("framework", "")),
                namespaces=[str(n) for n in (meta.get("namespaces") or [])],
                symbols=index_source_tree(target),
                revision=str(meta.get("revision", "")),
                checkout_revision=checkout_revision(located),
            )
        )

    return registry
