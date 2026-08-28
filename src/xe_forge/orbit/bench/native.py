"""
Framework-native measurement (plan §5.4, §10.3, §17).

Xe-Forge times kernels through `ai_bench`, and that is correct for a *standalone
extracted* kernel: a bundle has no framework to ask. The in-place path (§13.6) never
extracts, so the thing under test is reachable through the framework's own entry
points — and the framework already ships a benchmark for it, written by the people who
know what its performance means.

Using it instead of a bespoke timer is not a convenience. A hand-rolled harness measures
something adjacent to the workload: it picks its own warmup, its own batching, its own
definition of a token, and any of those can drift from what the framework does in
production. When the number is going to be compared against a published figure, or
against what the user sees, the framework's own harness is the only one that is
measuring the same thing.

AMD's Hyperloom takes the same position — its `integrate` step re-runs the framework's
own `bench_serving` / `benchmark_serving.py` rather than a harness of its own, and its
FLOP figures come from analytic per-architecture models, never from a bolted-on counter.

Three layers, all native:

* **Workload throughput** — the framework's declared benchmark (`vllm bench throughput`,
  SGLang's `bench_offline_throughput`). Resolved per framework, because which harness is
  authoritative is knowledge about that framework (§10.6), not a constant.
* **Kernel timing** — `torch.utils.benchmark.Timer`, PyTorch's equivalent of Triton's
  `do_bench`. It handles warmup, replicates and device synchronisation, and returns a
  distribution rather than one scalar — which is what §17 needs and what a naive
  `time.perf_counter()` around a launch cannot give.
* **FLOPs** — `torch.utils.flop_counter.FlopCounterMode`, which counts through the
  dispatcher and so sees what actually ran, including whatever the backend substituted.

Nothing here imports torch at module scope: Orbit's analysis path runs in CPU-only CI
without it (§15.3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Each framework's own benchmark, as a command. The framework knows what its performance
# means; we do not get to redefine it. Overridden from the adapter/knowledge file where
# one is declared (§10.3) — this table is the fallback for a framework we have not been
# told about.
NATIVE_WORKLOAD_HARNESS: dict[str, list[str]] = {
    "vllm": ["vllm", "bench", "throughput"],
    "sglang": ["python", "-m", "sglang.bench_offline_throughput"],
}

# Timer needs enough replicates for a dispersion estimate; below this the interval it
# reports is not meaningful.
MIN_REPLICATES = 5


@dataclass
class KernelTiming:
    """A kernel's measured time as a distribution, never a scalar."""

    samples_us: list[float] = field(default_factory=list)
    label: str = ""
    harness: str = "torch.utils.benchmark.Timer"

    @property
    def median_us(self) -> float:
        if not self.samples_us:
            return 0.0
        ordered = sorted(self.samples_us)
        return ordered[len(ordered) // 2]

    @property
    def usable(self) -> bool:
        """Whether there are enough samples for §17 to say anything."""
        return len(self.samples_us) >= MIN_REPLICATES

    def format(self) -> str:
        if not self.samples_us:
            return f"{self.label}: no samples"
        return (
            f"{self.label}: median {self.median_us:.2f} us over {len(self.samples_us)} "
            f"replicates (via {self.harness})"
        )


def time_kernel(
    fn: Callable[[], object],
    label: str = "",
    min_run_time: float = 0.2,
    replicates: int = MIN_REPLICATES,
    # 100 minimum on Intel hardware: the first many launches carry SPIR-V JIT,
    # autotune selection, memory-pool growth and clock ramp, not the kernel — a
    # 29% first-position effect was measured on this stack (§17.5), and 3 warmup
    # passes measured the transient. Matches the core executor's 200 default.
    warmup: int = 100,
) -> KernelTiming:
    """Time a callable with PyTorch's own benchmark harness, as a distribution.

    `Timer` is used rather than a loop around `perf_counter` because it synchronises the
    device and sizes its inner loop to the measured cost; a hand-rolled loop on a GPU
    usually times the launch rather than the kernel, and does so without saying it did.

    Two details `blocked_autorange` alone gets wrong for §17's purposes:

    * It returns one entry per *block*, and when `min_run_time` is satisfied by a single
      block that is one sample. A single sample has no dispersion, so no interval can be
      computed from it — the first version of this function silently produced exactly
      that. Calling it `replicates` times and taking each block's median gives genuinely
      independent samples.
    * It does not warm up. On a JIT backend the first call compiles, and that cost lands
      in the first sample: one measurement here came back at 220 ms for a 512x512 matmul.
      The explicit warmup runs are discarded before anything is recorded.
    """
    from torch.utils.benchmark import Timer

    timer = Timer(stmt="fn()", globals={"fn": fn}, label=label or "kernel")
    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = []
    for _ in range(max(1, replicates)):
        measurement = timer.blocked_autorange(min_run_time=min_run_time)
        samples.append(measurement.median * 1e6)
    return KernelTiming(samples_us=samples, label=label or "kernel")


@dataclass
class FlopResult:
    """Measured FLOPs for one invocation, and what they imply."""

    total_flops: int = 0
    by_operator: dict[str, int] = field(default_factory=dict)
    counted: bool = False

    def tflops_at(self, seconds: float) -> float | None:
        """Achieved TFLOP/s, or None when there is nothing to divide.

        Returning None rather than 0.0 keeps "we did not measure this" distinct from
        "this kernel did no arithmetic" — a distinction a roofline plot destroys if the
        two arrive as the same number.
        """
        if not self.counted or seconds <= 0:
            return None
        return self.total_flops / seconds / 1e12


def count_flops(fn: Callable[[], object]) -> FlopResult:
    """Count FLOPs through the dispatcher with PyTorch's own counter.

    Dispatcher-level counting sees what actually ran, including a backend substituting
    its own implementation — which is exactly the case Orbit cares about, since the
    kernel that executed is frequently not the one the source suggests.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return FlopResult(counted=False)

    counter = FlopCounterMode(display=False)
    try:
        with counter:
            fn()
    except Exception:
        # A workload that cannot run under the counter is a fact about the counter, not
        # a zero-FLOP kernel. Report it as uncounted.
        return FlopResult(counted=False)

    by_operator: dict[str, int] = {}
    for module_counts in counter.get_flop_counts().values():
        for op, count in module_counts.items():
            by_operator[str(op)] = by_operator.get(str(op), 0) + int(count)

    return FlopResult(
        total_flops=int(counter.get_total_flops()),
        by_operator=by_operator,
        counted=True,
    )


def native_harness_for(framework: str, declared: list[str] | None = None) -> list[str] | None:
    """The framework's own benchmark command.

    `declared` comes from the adapter or its knowledge file and wins, so adding a
    framework's harness is a YAML change rather than an edit here (§10.6). Returning
    None means we have not been told what this framework's benchmark is — which is
    reported as a gap rather than papered over with a generic timer that would measure
    something else.
    """
    if declared:
        return list(declared)
    return list(NATIVE_WORKLOAD_HARNESS.get(framework, [])) or None


def describe_provenance(framework: str, harness: list[str] | None) -> str:
    """Say which harness produced a number, so a reader can reproduce it.

    A throughput figure with no named source cannot be checked, and the difference
    between "the framework's benchmark" and "ours" is exactly what a reader needs to
    know before comparing it to anything published.
    """
    if not harness:
        return (
            f"no native benchmark declared for {framework!r}; throughput was not "
            f"measured by the framework's own harness"
        )
    return f"measured by {framework}'s own harness: {' '.join(harness)}"
