"""Framework frontend: from a model running under its framework to a kernel manifest.

    xe-forge capture   run the framework, record what it executed  -> capture.json
    xe-forge analyze   rank the captured workloads                 -> kernels.yaml + specs/
    xe-forge optimize  one ordinary Xe-Forge run per manifest entry -> workspaces/
    xe-forge run       all of the above, then collect the winners   -> results/

The frontend decides *what* to optimize and at *which* shapes. It never writes kernel or
reference code: the manifest names a kernel (symbol, repository, DSL) and its shapes, and
Xe-Forge's own workspace -- the kernel-locator, the baseline-as-oracle, the engines, the
trial tree and port-back -- does the rest, exactly as for a hand-written manifest.

Capture runs under the framework's interpreter (e.g. the vLLM venv) and imports nothing
from this package but ``ir``, ``torch_trace``, ``canonical``, ``naming`` and the adapter.
"""
