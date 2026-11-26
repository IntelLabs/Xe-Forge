"""Framework frontend: capture a model under its framework, rank its workloads, write a manifest.

Capture runs under the framework's interpreter and imports only ``ir``, ``torch_trace``,
``canonical``, ``naming`` and the adapter from this package.
"""

import os
from pathlib import Path

CACHE = Path(os.environ.get("X_FORGE_CACHE", Path.home() / ".cache" / "x_forge"))
