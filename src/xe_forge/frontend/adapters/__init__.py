"""Framework adapters. Each one only runs its framework and leaves artefacts in ``raw/``.

An adapter module provides::

    add_arguments(parser)            # its own flags, on top of the common ones
    capture(args, raw_dir) -> dict   # run the framework; return the run metadata

and, when its artefacts are not torch.profiler traces, ``parse(raw_dir) -> TraceSummary``.
Everything after that -- canonical workloads, ranking, the manifest -- is shared.
"""

from __future__ import annotations

import importlib

ADAPTERS = {
    "torch": "xe_forge.frontend.adapters.pytorch",
    "vllm": "xe_forge.frontend.adapters.vllm",
    "sglang": "xe_forge.frontend.adapters.sglang",
    "openvino": "xe_forge.frontend.adapters.openvino",
}


def get(name: str):
    if name not in ADAPTERS:
        raise SystemExit(f"unknown framework {name!r}; one of: {', '.join(ADAPTERS)}")
    return importlib.import_module(ADAPTERS[name])
