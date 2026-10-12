"""Read a model's metadata without knowing where its family keeps it.

Adapters store the framework's model config whole (``run.model.config``). Multimodal and
hybrid models nest the language model's config (``text_config``, ``language_config``,
``llm_config``), so a value is looked up there too, depth first. Names are the config's
own keys; there is no generic ``top_k``, which would match sampling configuration.
"""

from __future__ import annotations

import json

NESTED = ("config", "text_config", "language_config", "llm_config")


def _configs(cfg: dict):
    yield cfg
    for key in NESTED:
        child = cfg.get(key)
        if isinstance(child, dict):
            yield from _configs(child)


def model_value(model: dict | None, *names: str):
    """The first value any of ``names`` has in the model metadata or a nested config."""
    for cfg in _configs(model or {}):
        for name in names:
            value = cfg.get(name)
            if value is not None:
                return value
    return None


def moe_topk(model: dict | None) -> int | None:
    return model_value(
        model, "num_experts_per_tok", "experts_per_token", "num_experts_per_token", "moe_topk"
    )


def moe_experts(model: dict | None) -> int | None:
    """Routed experts in the whole model (global, before any expert parallelism)."""
    return model_value(model, "num_local_experts", "num_experts", "n_routed_experts")


def stored(config: dict) -> dict:
    """A model config as JSON stores it: whole, so nothing has to be chosen in advance."""
    return json.loads(json.dumps(config, default=str))
