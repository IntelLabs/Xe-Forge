"""
Handing Orbit's measurements to Xe-Forge's optimizer as knowledge (plan §9.5, §9.9).

Orbit and Xe-Forge already meet at a directory: emit writes a `Model` plus a spec, and
`optimize_kernel_dir` hands it to `XeForgePipeline`. That seam passes the *code* and
throws away everything Orbit learned getting there — which provider produced the kernel,
which template instantiation actually ran, what share of GPU time it holds, what its
Amdahl ceiling is, which shapes the workload really used, and which compiler axes are
worth sweeping before an agent is invoked at all.

All of that is exactly what an optimizer needs and cannot derive from a source file. A
kernel at 0.5% of GPU time and a kernel at 40% deserve different effort; a kernel whose
ceiling is 2% should not be rewritten at all (§18); a SYCL kernel running in the wrong
GRF mode wants a flag, not an algorithm (§11.7). Passing the file alone means the
optimizer re-derives what it can and guesses the rest.

Xe-Forge already has the mechanism for this. `knowledge/loader.py` collects
`common/*.yaml`, then `<dsl>/common/`, then `<dsl>/<device>/`, and scopes entries by
optimization stage — so a measured fact becomes available to the right stage without
any new plumbing. This module renders a `KernelRecord` into that schema.

Two rules keep it honest:

* **Only what was measured.** Every entry cites the run it came from. A knowledge base
  that mixes measurements with plausible-sounding advice is worse than one with fewer
  entries, because a reader cannot tell which is which.
* **A measurement is not a recommendation.** Entries carry the numbers and their
  consequences ("this kernel holds 1.8% of GPU time, so its ceiling is 1.8%"), and stop
  short of asserting what will be faster. Xe-Forge's stages decide that; inventing a
  `pattern_after` we never benchmarked would put a guess into the corpus in the format
  reserved for verified transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xe_forge.orbit.models import KernelRecord, Provider

# Where Orbit writes its contributions. `common/` because these are facts about a
# measured workload rather than about a language, and the loader reads `common/` for
# every DSL and device (§9.5).
ORBIT_KB_SUBDIR = Path("common")
ORBIT_KB_FILENAME = "orbit_measured.yaml"

# Which Xe-Forge stage each kind of finding belongs to, so the loader scopes it to the
# stage that can act on it rather than pasting everything into every prompt.
STAGE_ANALYSIS = "analysis"
STAGE_DEVICE = "device_specific"
STAGE_AUTOTUNING = "autotuning"
STAGE_MEMORY = "memory_access"

# Providers whose kernels have no editable source, so a rewrite stage cannot act on
# them however attractive the numbers look (§7.2).
OPAQUE_PROVIDERS = {Provider.ONEDNN, Provider.ONEMKL, Provider.RUNTIME}


@dataclass
class MeasuredFact:
    """One thing Orbit measured, in Xe-Forge's knowledge-entry shape."""

    id: str
    name: str
    stage: str
    description: str
    rationale: str
    applies_to: list[str] = field(default_factory=list)
    notes: str = ""

    def to_entry(self) -> dict[str, Any]:
        # `pattern_before`/`pattern_after` are deliberately empty: this is a measurement,
        # not a transformation. The loader accepts an entry without them; inventing a
        # transformation we never benchmarked would file a guess as a verified pattern.
        return {
            "id": self.id,
            "name": self.name,
            "stage": self.stage,
            "description": self.description,
            "rationale": self.rationale,
            "applies_to": self.applies_to,
            "notes": self.notes,
        }


def facts_for_kernel(kernel: KernelRecord, run_id: str = "") -> list[MeasuredFact]:
    """Everything Orbit established about one kernel, as scoped knowledge entries."""
    facts: list[MeasuredFact] = []
    tag = kernel.id
    cite = f"measured in Orbit run {run_id}" if run_id else "measured by Orbit"
    applies = [
        p for p in (kernel.provider.value, (kernel.language.value if kernel.language else "")) if p
    ]

    # -- what it costs, which is what decides how much effort it deserves ----
    share = kernel.gpu_time_share * 100
    # `max_e2e_gain` is ALREADY a percentage — `amdahl_ceiling` multiplies by 100 before
    # returning, so it compares directly against the MDE. Scaling it again here printed
    # "at most 3774.87%" for a kernel holding 93% of GPU time, which is impossible on
    # its face and was caught by reading the output rather than by a test.
    ceiling = kernel.max_e2e_gain
    facts.append(
        MeasuredFact(
            id=f"orbit_{tag}_budget",
            name=f"{_short(kernel)}: measured cost and ceiling",
            stage=STAGE_ANALYSIS,
            description=(
                f"This kernel holds {share:.2f}% of GPU time across {kernel.calls} calls "
                f"({kernel.avg_time_us:.1f} us each). Making it infinitely fast improves "
                f"end-to-end time by at most {ceiling:.2f}%."
            ),
            rationale=(
                "The Amdahl ceiling bounds what any rewrite of this kernel can be worth, "
                "however large its microbenchmark speedup. A kernel whose ceiling sits "
                "below the workload's minimum detectable effect cannot be shown to have "
                "helped even if it did."
            ),
            applies_to=applies,
            notes=cite,
        )
    )

    # -- which specialization ran, without which a rewrite may target the wrong one --
    if kernel.source_symbol or kernel.demangled_name:
        symbol = kernel.source_symbol or kernel.demangled_name or ""
        if "<" in symbol:
            facts.append(
                MeasuredFact(
                    id=f"orbit_{tag}_instantiation",
                    name=f"{_short(kernel)}: the instantiation that actually ran",
                    stage=STAGE_ANALYSIS,
                    description=f"Observed instantiation: {symbol}",
                    rationale=(
                        "Template arguments are the only thing separating two kernels "
                        "that share an entry symbol. Optimizing a different "
                        "instantiation produces a real speedup on code the workload "
                        "never executes."
                    ),
                    applies_to=applies,
                    notes=cite,
                )
            )

    # -- the real shapes, which beat synthetic ones ---------------------------
    if kernel.shapes:
        observed = kernel.shapes[0]
        dims = ", ".join(f"{k}={v}" for k, v in observed.dims.items())
        dtypes = ", ".join(f"{k}:{v}" for k, v in observed.dtypes.items())
        facts.append(
            MeasuredFact(
                id=f"orbit_{tag}_shapes",
                name=f"{_short(kernel)}: shapes observed in the workload",
                stage=STAGE_AUTOTUNING,
                description=f"dims {dims or 'unrecorded'}; dtypes {dtypes or 'unrecorded'}",
                rationale=(
                    "Tuning against a synthetic shape optimizes a kernel the workload "
                    "does not run. These came from the trace."
                ),
                applies_to=applies,
                notes=cite,
            )
        )

    # -- compiler axes first, because they are cheap and deterministic (§11.7) --
    if kernel.language and kernel.language.value in ("sycl", "sycl_tla"):
        facts.append(
            MeasuredFact(
                id=f"orbit_{tag}_compiler_axes",
                name=f"{_short(kernel)}: sweep compiler options before rewriting",
                stage=STAGE_DEVICE,
                description=(
                    "GRF mode, sub-group size, AOT target versus SPIR-V JIT, and the "
                    "floating-point contract are all available on this kernel and are "
                    "swept deterministically."
                ),
                rationale=(
                    "These cost a rebuild rather than an agent call, and they are the "
                    "correct first move. A kernel asked to be rewritten when it is "
                    "simply running in the wrong GRF mode yields an expensive, "
                    "complicated, worse answer."
                ),
                applies_to=applies,
                notes=cite,
            )
        )

    # -- and the case where a rewrite is not the action at all ----------------
    if kernel.provider in OPAQUE_PROVIDERS:
        facts.append(
            MeasuredFact(
                id=f"orbit_{tag}_opaque",
                name=f"{_short(kernel)}: no editable source",
                stage=STAGE_ANALYSIS,
                description=(
                    f"Provider {kernel.provider.value} ships this kernel as a compiled "
                    f"library primitive. There is no source to rewrite at any extraction "
                    f"level."
                ),
                rationale=(
                    "Not unactionable — region fusion, backend change, layout change and "
                    "library configuration all still apply — but a source-rewrite stage "
                    "cannot act here and should not be asked to."
                ),
                applies_to=applies,
                notes=cite,
            )
        )

    return facts


def render_document(kernels: list[KernelRecord], run_id: str = "") -> dict[str, Any]:
    """Render measured facts for a whole catalog into one loadable YAML document."""
    entries: list[dict[str, Any]] = []
    for kernel in kernels:
        entries.extend(f.to_entry() for f in facts_for_kernel(kernel, run_id))
    return {
        "version": "1.0",
        "source": "xe-orbit",
        "run_id": run_id,
        "description": (
            "Facts measured by Xe-Orbit on a real workload. Every entry cites the run "
            "it came from. Nothing here is a recommendation; the numbers and their "
            "consequences only."
        ),
        "patterns": entries,
    }


def write_knowledge(
    kernels: list[KernelRecord],
    knowledge_dir: Path,
    run_id: str = "",
) -> Path:
    """Write Orbit's measurements where Xe-Forge's loader will find them (§9.5).

    Under `common/` because the loader collects `common/` for every DSL and device; a
    file at the knowledge-base root is silently ignored.
    """
    import yaml

    target_dir = Path(knowledge_dir) / ORBIT_KB_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ORBIT_KB_FILENAME
    target.write_text(
        yaml.safe_dump(render_document(kernels, run_id), sort_keys=False),
        encoding="utf-8",
    )
    return target


def _short(kernel: KernelRecord) -> str:
    """A readable kernel label, since demangled SYCL names run to hundreds of chars."""
    name = kernel.source_symbol or kernel.demangled_name or kernel.runtime_name
    name = name.split("<")[0].split("(")[0]
    return name.rsplit("::", 1)[-1][:48] or kernel.id


# ---------------------------------------------------------------------------
# Hardware counters in the agent's context (§9.5, §11.7, §18)
# ---------------------------------------------------------------------------


def profiler_context(properties, gpu_busy_us=None, total_time_us=None, launch_gap_us=None) -> str:
    """Render unitrace's per-kernel device metrics for an optimizer prompt.

    Orbit collects these and, until now, told the agent none of them — so the agent
    reasoned about occupancy and register pressure from first principles and got both
    wrong. Every number below answers a question it was otherwise guessing at:

    * **spills** settle whether a larger block costs register pressure. The agent argued
      that raising `BLOCK_SIZE` from 1024 to 8192 would amortize per-program setup; the
      spill count is the direct evidence for or against, and it measured 2x slower.
    * **SIMD width** is what the kernel was *compiled* for, not what the ISA supports.
      The agent asserted "Intel XPU uses a 16-wide sub-group" as a fact about the vendor;
      it is a fact about this compilation.
    * **GRF per thread** is the §11.7 axis — the single largest lever on Xe, and not
      something to propose blind.
    * **AOT versus JIT** changes both performance and how a rebuild behaves (§11.4).
    * **launch gap against GPU busy** decides whether the kernel is even the problem. A
      workload whose device is idle most of the wall clock is launch-bound, and no amount
      of kernel work will show up end to end.
    """
    lines: list[str] = []

    if gpu_busy_us is not None and total_time_us:
        busy = gpu_busy_us / total_time_us * 100.0
        lines.append(f"  GPU busy: {busy:.1f}% of wall clock (measured, not estimated)")
        if busy < 50.0:
            # Worth stating outright: it inverts what is worth proposing.
            lines.append(
                "    The device is idle for most of the run, so this workload is "
                "launch- or host-bound. A faster kernel cannot show up end to end until "
                "that is addressed."
            )
    if launch_gap_us:
        lines.append(f"  total launch gap: {launch_gap_us:.0f} us of dead time between kernels")

    for prop in properties or []:
        detail = [f"  {prop.name[:56]}"]
        if prop.simd:
            detail.append(f"    compiled for SIMD{prop.simd} (this build, not the ISA maximum)")
        if prop.grf_per_thread:
            detail.append(f"    GRF: {prop.grf_per_thread} per thread")
        if prop.spill_per_thread:
            detail.append(
                f"    SPILLS: {prop.spill_per_thread} bytes per thread — the register file "
                f"is already oversubscribed, so anything widening the tile makes this worse"
            )
        elif prop.grf_per_thread:
            detail.append("    no spills at this configuration")
        if prop.slm_per_group:
            detail.append(f"    SLM: {prop.slm_per_group} bytes per work-group")
        if prop.compiled:
            detail.append(f"    {prop.compiled}-compiled")
        lines.extend(detail)

    if not lines:
        return (
            "  no device counters available (unitrace not run) — occupancy, register "
            "pressure and launch overhead are unmeasured, so any claim about them below "
            "is inference rather than evidence"
        )
    return "\n".join(lines)


def occupancy_context(result) -> str:
    """Render VTune's occupancy findings for an optimizer prompt.

    Only the limiter is actionable, so it leads. Occupancy alone is a number an agent
    will reason around; the limiter is the lever. Both of the agent's failed proposals
    on this hardware — a larger reduction tile and more warps — were occupancy arguments
    made without knowing which constraint bound, and both measured roughly 2x slower.

    A kernel already at full occupancy is stated as such rather than omitted, because
    "nothing is limiting this" is exactly the fact that stops an agent proposing an
    occupancy fix for a kernel that does not have an occupancy problem.
    """
    if result is None or not getattr(result, "available", False):
        reason = getattr(result, "reason", "") if result is not None else "not collected"
        return (
            f"  no GPU occupancy counters ({reason}) — occupancy and its limiting factor "
            f"are unmeasured, so any claim about them is inference rather than evidence"
        )

    lines: list[str] = []
    for kernel in result.kernels:
        if kernel.occupancy_percent is None:
            continue
        if kernel.low_occupancy:
            lines.append(
                f"  {kernel.name[:52]}: occupancy {kernel.occupancy_percent:.0f}%, "
                f"limited by {kernel.limiter}"
            )
            if kernel.global_size:
                lines.append(
                    f"      work size global {kernel.global_size}, local {kernel.local_size}"
                    f"; SIMD{kernel.simd_width}, spills {kernel.spill_bytes} bytes"
                )
        else:
            lines.append(
                f"  {kernel.name[:52]}: occupancy {kernel.occupancy_percent:.0f}% — "
                f"{kernel.limiter}; occupancy is not this kernel's problem"
            )
    return "\n".join(lines) if lines else "  VTune reported no kernels with occupancy data"
