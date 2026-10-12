"""The model's config, kept whole in the capture (``run.model.config``).

Nothing here reads keys out of it: what a family calls its expert count or top-k, and
where a multimodal model nests its language model, is the naming agent's to read
(``names.json`` -> ``run.model_facts``)."""

from __future__ import annotations

import json


def stored(config: dict) -> dict:
    """A model config as JSON stores it: whole, so nothing has to be chosen in advance."""
    return json.loads(json.dumps(config, default=str))
