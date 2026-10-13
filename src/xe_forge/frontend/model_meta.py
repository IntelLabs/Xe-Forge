"""The model's config, stored whole in the capture; the naming agent reads facts from it."""

from __future__ import annotations

import json


def stored(config: dict) -> dict:
    """The config made JSON-serializable."""
    return json.loads(json.dumps(config, default=str))
