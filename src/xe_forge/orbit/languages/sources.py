"""
Multi-framework SYCL source registry (plan §11.2, §12.5).

§11.2's table names where Intel's kernels actually live: `torch-xpu-ops` for ATen XPU
operators, `vllm-xpu-kernels` for vLLM's custom ops, `sgl-kernel-xpu` for SGLang's, plus
IPEX and sycl-tla. All of them are open source SYCL C++ registered as dispatcher ops —
which is exactly what makes the P1 override rung work for them (§11.8).

The practical obstacle is not licensing or language, it is *packaging*. Every one of
these installs as a wheel containing a compiled shared object; the sources are in a
separate repository. So on a normal machine there is no build database and no source
tree, and a resolver that requires either reports E3 forever — not because the kernel is
tangled, but because nobody checked out the code.

This module closes that gap: point Orbit at the trees that are present, and kernels from
any of them resolve to a file. Trees that are absent are reported as absent rather than
silently reducing coverage, because "we could not find the source" and "this kernel has
no source" are different findings and only one of them is about the kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Where the knowledge files live. §10.6 requires per-framework knowledge to be *data*:
# "most of a new framework is the YAML file", with code supplying only what genuinely
# needs code. Kernel source locations are knowledge about a framework, so they belong
# here rather than in a Python literal — adding SGLang's kernels became a YAML file, not
# an edit to this module.
KNOWLEDGE_DIR = Path("knowledge_base/common")
FRAMEWORK_GLOB = "framework_*.yaml"


def load_known_trees(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Read kernel source locations from the framework knowledge files (§10.6).

    Falls back to the built-in table when the knowledge base is not on disk — Orbit has
    to work from a wheel with no knowledge directory, and a missing YAML should reduce
    what is known, never break resolution outright.
    """
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
                # §12.10: a tree is identified by URL *and* revision. An empty pin is
                # legal but reported as unpinned, never silently treated as matching.
                "revision": str(entry.get("revision") or ""),
            }

    return trees or dict(_BUILTIN_SOURCE_TREES)


# Fallback used only when no knowledge base is reachable, so a wheel-only install still
# resolves the trees §11.2 names.
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
    # Which framework's knowledge file declared this tree, so a resolved kernel can be
    # attributed to a framework rather than only to a directory.
    framework: str = ""
    # C++ namespaces this tree owns, from its knowledge file. Used to break ties when
    # more than one tree defines an identifier.
    namespaces: list[str] = field(default_factory=list)
    symbols: dict[str, Path] = field(default_factory=dict)
    # §12.10: the revision the knowledge file pins this tree to — the one matching the
    # installed wheel — and the revision actually checked out. Version skew was measured
    # as the single largest source of unverified bundles, so the mismatch is surfaced
    # here rather than discovered later as a compile error with a misleading message.
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
        """One of: unpinned | ok | skew | unverified.

        `unverified` means a pin is declared but the checkout's revision could not be
        read (no git, no .git directory). That is reported as its own state because
        "we could not check" and "it matches" are different claims, and folding the
        first into the second is how a skewed tree gets trusted.
        """
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
        """Find the tree defining `symbol`, preferring the one the namespace names.

        Identifiers collide across frameworks: `rms_norm_kernel` is defined by both
        torch-xpu-ops and vllm-xpu-kernels. Scanning in registry order resolved
        `vllm::rms_norm_kernel` to torch-xpu-ops' `LayerNormKernels.cpp` — the wrong
        framework's kernel, chosen by list position. The demangled name carries the
        namespace, so the ambiguity is only real if nothing declares that namespace.
        """
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

        The symbol index answers the easy majority exactly and for free. What it cannot
        do is parse C++ — it is a pattern scan, and a pattern scan over mangled names
        fails in ways that are hard to predict and silent when they happen. On real
        Intel kernel trees it recovered `GeluErfFunctor` and destroyed `IgammaFunctor`,
        because the identifier contains the character the template-mangling pattern was
        keyed on.

        So when the index misses, the question goes to a `RepoAgent` (§3, §6) rather
        than to a cleverer regular expression. That is the right split: the deterministic
        path keeps its exactness and costs nothing, and the genuinely ambiguous residue
        gets something that can actually read the code.
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
        """Ask a RepoAgent which file defines this kernel, and verify the answer.

        The agent's reply is checked twice before it is believed: the file must exist
        (enforced by the protocol's parser) and it must sit inside one of the indexed
        trees. An agent that names a plausible path outside the trees we are analysing
        has not answered the question that was asked.
        """
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
                # A provider that is unreachable must not abort resolution; the caller
                # still gets an honest "unresolved".
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
                "been shown. Those two are reported differently on purpose (§12.5)."
            )
        for tree in self.trees:
            if tree.pin_state == "skew":
                lines.append(
                    f"SKEW: {tree.name} is pinned to {tree.revision} but the checkout is at "
                    f"{tree.checkout_revision[:12]} — a bundle built from it is a different "
                    f"kernel than the one the wheel runs (§12.10). Check out the pinned "
                    f"revision before extracting."
                )
            elif tree.pin_state == "unpinned":
                lines.append(
                    f"UNPINNED: {tree.name} declares no revision in its knowledge file. "
                    f"A tree is identified by URL *and* revision (§12.10); add `revision:` "
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

    Read from `.git` directly rather than by running git: discovery runs on every
    `xe-orbit sources` call and must not depend on a git binary being present. Handles
    a detached HEAD, a symbolic ref, packed refs, and a `.git` *file* (worktrees).
    Any unreadable state returns "" — reported upstream as `unverified`, never as a
    match.
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
