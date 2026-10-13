"""``sglang.benchmark.one_batch`` with the shared prompt's token ids.

one_batch takes a prompt only as text lines, which it re-tokenizes and does not cut to
``--input-len``; its synthetic inputs are random. Its input builder takes ready ids, so this
runs its CLI with that builder handed the ids ``frontend.prompt`` made:

    python -m xe_forge.frontend.adapters.sglang_one_batch <ids.json> <one_batch args...>
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    ids = json.loads(open(sys.argv[1]).read())
    import inspect

    import sglang
    import sglang.benchmark.one_batch as one_batch

    build = getattr(one_batch, "prepare_synthetic_inputs_for_latency_test", None)
    params = list(inspect.signature(build).parameters) if build else []
    if params[:3] != ["batch_size", "input_len", "custom_inputs"] or not hasattr(
        one_batch, "cli_main"
    ):
        raise SystemExit(
            f"sglang {getattr(sglang, '__version__', '?')}: one_batch no longer has the input "
            f"builder this wrapper hands token ids to ({params}); update sglang_one_batch.py"
        )
    one_batch.prepare_synthetic_inputs_for_latency_test = (
        lambda batch_size, input_len, custom_inputs=None: build(batch_size, input_len, ids)
    )
    sys.argv = ["one_batch", *sys.argv[2:]]
    one_batch.cli_main()


if __name__ == "__main__":
    main()
