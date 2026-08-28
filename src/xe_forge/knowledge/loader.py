"""Knowledge Base Loader — delegates to intel_kernel_kb when available.

The public API (load_knowledge_base, KnowledgeBase type alias) is preserved
unchanged so all callers in pipeline.py, agents, etc. continue to work.
Falls back gracefully to the internal YAML loader when intel-kernel-kb is
not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from intel_kernel_kb.compat.xe_forge import (
        XeForgeKnowledgeBase as KnowledgeBase,
        load_knowledge_base as _ikb_load,
    )

    _INTEL_KB_AVAILABLE = True
    logger.debug("intel-kernel-kb available; using compat KB loader")
except ImportError:
    _INTEL_KB_AVAILABLE = False
    from xe_forge.knowledge._legacy_loader import KnowledgeBase  # type: ignore[assignment]

    logger.debug("intel-kernel-kb not installed; using internal YAML loader")


def load_knowledge_base(
    knowledge_dir: str | Path,
    dsl: str = "triton",
    device_type: str = "xpu",
) -> KnowledgeBase:
    """Load the kernel optimization knowledge base.

    Delegates to intel_kernel_kb.compat.xe_forge when available,
    falling back to the internal YAML loader otherwise.
    The returned object is duck-type compatible in both cases.
    """
    if _INTEL_KB_AVAILABLE:
        return _ikb_load(knowledge_dir, dsl=dsl, device_type=device_type)
    from xe_forge.knowledge._legacy_loader import load_knowledge_base as _legacy_load

    return _legacy_load(knowledge_dir, dsl=dsl, device_type=device_type)
