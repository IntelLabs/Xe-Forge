"""Framework adapters: ``add_arguments(parser)``, ``capture(args, raw_dir) -> dict``, and
optionally ``parse(raw_dir) -> TraceSummary`` when artefacts are not torch.profiler traces.
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
