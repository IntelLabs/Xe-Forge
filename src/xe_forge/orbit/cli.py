#!/usr/bin/env python3
"""
The `xe-orbit` command line (plan §21).

Every stage is a subcommand, every stage reads and writes typed artifacts, and every
stage accepts `--replay <run-id>` to re-run from stored artifacts instead of live
hardware (§16.3). Replay is what makes this testable on CPU-only CI, and it is also
the debugging tool: when a run produces a surprising decision, the whole downstream
chain can be re-run against the exact artifacts that produced it.

`xe-orbit optimize` plans by default and runs the §13.5 loop only under `--apply`,
with an operator-supplied correctness harness — nothing autonomous ships by accident.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from xe_forge.orbit.artifacts import (
    ENVIRONMENT,
    EVENTS,
    KERNEL_CATALOG,
    LAUNCHES,
    MANIFEST,
    MEASUREMENT,
    TORCH_TRACE,
    WORKLOAD,
    ArtifactError,
    RunStore,
)
from xe_forge.orbit.models import RunManifest, WorkloadMeasurement, WorkloadSpec

logger = logging.getLogger("xe_orbit")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _orbit_base(args: argparse.Namespace) -> Path:
    return Path(args.orbit_dir) if args.orbit_dir else Path.cwd() / ".orbit"


def _resolve_store(args: argparse.Namespace, create: bool = False) -> RunStore:
    """Open the run a stage should operate on.

    `--replay <id>` and `--run <id>` both name an existing run; they differ only in
    intent, so both resolve the same way. Without either, a stage that produces new
    artifacts creates a run and a stage that consumes them uses the most recent.
    """
    base = _orbit_base(args)
    run_id = getattr(args, "replay", None) or getattr(args, "run", None)
    if run_id:
        return RunStore.open(run_id, base=base)
    if create:
        return RunStore.create(base=base)
    return RunStore.latest(base=base)


def _workload_from_args(args: argparse.Namespace) -> WorkloadSpec:
    if not args.command:
        raise SystemExit(
            "no workload command given. Put it after `--`, e.g.\n  xe-orbit run -- python train.py"
        )
    env = {}
    for item in args.env or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return WorkloadSpec(
        command=list(args.command),
        cwd=Path(args.cwd) if args.cwd else Path.cwd(),
        env=env,
        framework=args.framework if args.framework != "auto" else None,
        warmup_iterations=args.warmup,
        repetitions=args.repetitions,
        timeout_s=args.timeout,
    )


def _emit(payload: object, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif text:
        print(text)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_frameworks(args: argparse.Namespace) -> int:
    """List adapters, tiers and declared capabilities (§10)."""
    from xe_forge.orbit.adapters import describe_adapters

    rows = describe_adapters()
    if args.json:
        _emit(rows, True)
        return 0

    print(f"{'ADAPTER':<20} {'TIER':>4}  {'METRICS':<32} CAPABILITIES")
    print("-" * 92)
    for row in rows:
        if row.get("tier") == -1:
            print(f"{row['name']:<20} {'ERR':>4}  {row.get('error', '')}")
            continue
        caps = []
        if row["can_reset_state"]:
            caps.append("reset_state")
        if row["can_construct_single_layer"]:
            caps.append("single_layer(E3)")
        metrics = ",".join(row["metrics"])  # type: ignore[arg-type]
        print(f"{row['name']:<20} {row['tier']:>4}  {metrics:<32} {','.join(caps) or '-'}")
    print()
    print("Tier 0 degrades gracefully: an unfamiliar framework still gets kernel")
    print("discovery, provenance, capture and a wall-clock delta.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Baseline: environment, versions, device, timing with repetitions (§PR 2)."""
    from xe_forge.orbit.adapters import resolve_adapter
    from xe_forge.orbit.adapters.base import LoadSpec
    from xe_forge.orbit.executor import LocalExecutor
    from xe_forge.orbit.runtime import environment

    spec = _workload_from_args(args)
    store = RunStore.create(base=_orbit_base(args))
    executor = LocalExecutor()

    adapter = resolve_adapter(spec, args.framework)
    env_info = environment.capture(cwd=spec.cwd, preferred_device=args.device)

    prepared = adapter.prepare(spec)
    handle = adapter.launch(prepared.spec, executor)
    try:
        adapter.warmup(handle)
        measurement = adapter.benchmark(
            handle, LoadSpec(repetitions=spec.repetitions, warmup=spec.warmup_iterations)
        )
    finally:
        adapter.teardown(handle)

    measurement.clock_samples = env_info.clock_samples
    manifest = RunManifest(
        run_id=store.run_id,
        workload=spec,
        environment=env_info,
        framework=spec.framework,
        adapter=adapter.name,
        adapter_tier=getattr(adapter, "tier", 0),
    )

    store.save(MANIFEST, manifest)
    store.save(WORKLOAD, spec)
    store.save(ENVIRONMENT, env_info)
    store.save(MEASUREMENT, measurement)
    store.record_stage("run")

    if args.json:
        _emit(measurement.model_dump(mode="json"), True)
        return 0

    wall = measurement.wall_time
    print(f"run: {store.run_id}")
    print(f"  adapter:  {adapter.name} (Tier {getattr(adapter, 'tier', 0)})")
    print(f"  device:   {env_info.device_name} ({env_info.device_count} device(s))")
    print(
        f"  wall:     {wall.mean:.4f}s  95% CI [{wall.ci95_low:.4f}, {wall.ci95_high:.4f}]  n={wall.n}"
    )
    print(f"  MDE:      {measurement.minimum_detectable_effect:.2f}%")
    if prepared.notes:
        print()
        for note in prepared.notes:
            print(f"  note: {note}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """torch.profiler + unitrace + launch interception (§PR 3, §12.4)."""
    from xe_forge.orbit.executor import LocalExecutor
    from xe_forge.orbit.profiling import trace as trace_mod
    from xe_forge.orbit.profiling import unitrace as unitrace_mod

    if args.replay:
        store = RunStore.open(args.replay, base=_orbit_base(args))
        events = store.load(EVENTS, trace_mod.TraceEvents)
        print(f"replayed trace from {store.run_id}: {len(events.kernels)} kernel events")
        return 0

    if args.from_trace:
        store = RunStore.create(base=_orbit_base(args))
        events = trace_mod.ingest_trace_file(Path(args.from_trace))
        store.save(EVENTS, events)
        store.record_stage("trace")
        print(f"run: {store.run_id}")
        print(f"  ingested {len(events.kernels)} kernel events from {args.from_trace}")
        for warning in events.warnings:
            print(f"  warning: {warning}")
        return 0

    spec = _workload_from_args(args)
    store = RunStore.create(base=_orbit_base(args))
    executor = LocalExecutor()

    # unitrace first: its GPU-busy and launch-gap numbers decide whether the rest of
    # the pipeline is even worth running (§18).
    unitrace_result = unitrace_mod.run(
        spec.command,
        executor=executor,
        cwd=spec.cwd,
        env=spec.env,
        output_dir=store.subdir("traces", "unitrace"),
        timeout=spec.timeout_s,
    )
    store.save_json("traces/unitrace_summary.json", unitrace_result.model_dump(mode="json"))

    if not unitrace_result.available:
        print(f"unitrace: unavailable — {unitrace_result.reason}")

    # The workload runs as a subprocess, so in-process torch profiling and launch
    # interception do not apply here; a trace file is the handoff. With --wrap, the
    # handoff is arranged for the workload: the command runs through the shipped
    # profiler wrapper, which writes the trace exactly where this stage looks — the
    # point-and-start path for single-process torch workloads. Framework engines
    # that move GPU work into a subprocess (vLLM, SGLang) need their own profiler
    # hook, and the wrapper says so rather than emitting an empty trace silently.
    run_command, run_env = spec.command, spec.env
    if getattr(args, "wrap", False):
        import os

        src_root = str(Path(trace_mod.__file__).resolve().parents[3])
        run_env = dict(spec.env)
        prior = run_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        run_env["PYTHONPATH"] = src_root + (os.pathsep + prior if prior else "")
        run_command = [
            spec.command[0],
            "-m",
            "xe_forge.orbit.profiling.wrap",
            "--out",
            str(store.path(TORCH_TRACE)),
            "--",
            *spec.command,
        ]

    result = executor.run(run_command, env=run_env, cwd=spec.cwd, timeout=spec.timeout_s)
    if not result.ok:
        print(f"workload failed: exit {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-1500:], file=sys.stderr)
        # The failure is an enablement finding, not only a dead end (§5.6): classify
        # it and name the rung, the same wiring BenchRunner.measure carries.
        from xe_forge.orbit.enablement import diagnose

        for gap in diagnose(result.returncode, result.stdout, result.stderr):
            print(gap.format(), file=sys.stderr)
        return 1

    trace_path = store.path(TORCH_TRACE)
    if trace_path.is_file():
        events = trace_mod.ingest_trace_file(trace_path)
    else:
        events = trace_mod.TraceEvents(
            warnings=[
                f"no trace file at {trace_path}. The workload must export a Chrome "
                f"trace there, or pass --from-trace to ingest one it wrote elsewhere."
            ]
        )

    store.save(EVENTS, events)
    store.record_stage("trace")

    print(f"run: {store.run_id}")
    print(f"  kernel events: {len(events.kernels)}")
    if unitrace_result.gpu_busy_percent is not None:
        print(f"  GPU busy: {unitrace_result.gpu_busy_percent:.1f}% (unitrace)")
    for warning in events.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_kernels(args: argparse.Namespace) -> int:
    """Kernel catalog with gating and ranking (§PR 4, §18)."""
    from xe_forge.orbit.analysis.catalog import build_catalog, format_catalog
    from xe_forge.orbit.profiling.trace import TraceEvents

    store = _resolve_store(args)
    events = store.load(EVENTS, TraceEvents)

    mde = 0.0
    if store.exists(MEASUREMENT):
        measurement = store.load(MEASUREMENT, WorkloadMeasurement)
        mde = measurement.minimum_detectable_effect

    gpu_busy = None
    gaps = None
    summary_path = store.path("traces/unitrace_summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("available"):
            busy_us = summary.get("gpu_busy_us")
            total_us = summary.get("total_time_us")
            if busy_us and total_us:
                gpu_busy = min(100.0, busy_us / total_us * 100.0)
            gaps = summary.get("launch_gap_total_us")

    catalog = build_catalog(
        events,
        run_id=store.run_id,
        gpu_busy_percent=gpu_busy,
        launch_gap_total_us=gaps,
        minimum_detectable_effect=mde,
    )
    store.save(KERNEL_CATALOG, catalog)
    store.record_stage("kernels")

    if args.json:
        _emit(catalog.model_dump(mode="json"), True)
        return 0

    print(format_catalog(catalog, limit=args.limit))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Provenance, shapes, headroom and extraction level for one kernel (§21)."""
    from xe_forge.orbit.languages import resolve_backend
    from xe_forge.orbit.models import KernelCatalog

    store = _resolve_store(args)
    catalog = store.load(KERNEL_CATALOG, KernelCatalog)

    matches = [k for k in catalog.kernels if k.id == args.kernel_id]
    if not matches:
        print(f"no kernel {args.kernel_id!r} in run {store.run_id}", file=sys.stderr)
        print(f"known ids: {', '.join(k.id for k in catalog.kernels[:20])}", file=sys.stderr)
        return 1
    kernel = matches[0]

    if args.json:
        _emit(kernel.model_dump(mode="json"), True)
        return 0

    print(f"{kernel.id}  {kernel.runtime_name}")
    print(f"  provider:     {kernel.provider.value}")
    print(f"  language:     {kernel.language.value if kernel.language else 'unknown'}")
    print(f"  extraction:   {kernel.extraction_level.value if kernel.extraction_level else '--'}")
    print(f"  calls:        {kernel.calls}")
    print(
        f"  GPU share:    {kernel.gpu_time_share * 100:.2f}%  ({kernel.total_time_us:.0f}us total)"
    )
    print(
        f"  Amdahl ceil:  {kernel.max_e2e_gain:.2f}%  (vs MDE {catalog.minimum_detectable_effect:.2f}%)"
    )
    print(f"  confidence:   {kernel.describe_confidence()}  [{kernel.resolution_method.value}]")
    print(f"  actions:      {', '.join(a.value for a in kernel.actions_available) or 'none'}")
    print(f"  dispatch:     {' -> '.join(kernel.dispatch_chain) or 'unknown'}")
    if kernel.skip_reason:
        print(f"  SKIPPED:      {kernel.skip_reason}")

    backend, score = resolve_backend(kernel.runtime_name)
    if backend is not None:
        print(f"  language backend: {backend.name} (confidence {score:.2f})")
        print(f"    iteration cost: ~{backend.cost_profile.iteration_seconds:.0f}s per candidate")
        axes = backend.option_axes()
        if axes:
            print(f"    compiler axes:  {', '.join(a.name for a in axes)}")
    if kernel.shapes:
        print("  shapes:")
        for shape in kernel.shapes[:5]:
            dims = ", ".join(f"{k}={v}" for k, v in list(shape.dims.items())[:6])
            print(f"    x{shape.count:<5} {dims}")
    return 0


def _load_kernel(store: RunStore, kernel_id: str):
    """Fetch one kernel from a run's catalog, or exit with a useful message."""
    from xe_forge.orbit.models import KernelCatalog

    catalog = store.load(KERNEL_CATALOG, KernelCatalog)
    for kernel in catalog.kernels:
        if kernel.id == kernel_id:
            return kernel
    print(f"no kernel {kernel_id!r} in run {store.run_id}", file=sys.stderr)
    print(f"known ids: {', '.join(k.id for k in catalog.kernels[:20])}", file=sys.stderr)
    raise SystemExit(1)


def _launch_map(store: RunStore, kernels) -> dict:
    """Match intercepted launch records to catalog kernels by name."""
    from xe_forge.orbit.profiling.interception import LaunchLog

    if not store.exists(LAUNCHES):
        return {}
    log = store.load(LAUNCHES, LaunchLog)
    mapping = {}
    for kernel in kernels:
        for record in log.records:
            if kernel.runtime_name in record.fq_name or record.fq_name in kernel.runtime_name:
                mapping[kernel.id] = record
                break
    return mapping


def cmd_extract_all(args: argparse.Namespace) -> int:
    """Extract every kernel in the run and report coverage by GPU time (§12.3)."""
    from xe_forge.orbit.extract import extract_all
    from xe_forge.orbit.models import KernelCatalog

    store = _resolve_store(args)
    catalog = store.load(KERNEL_CATALOG, KernelCatalog)

    agent = None
    if not getattr(args, "no_agent", False):
        from xe_forge.orbit.agents import default_agent

        agent = default_agent()

    coverage = extract_all(
        catalog,
        output_root=store.subdir("bundles"),
        launches=_launch_map(store, catalog.kernels),
        level=args.level,
        agent=agent,
    )

    for item in coverage.extractions:
        if item.result is not None:
            store.save(f"bundles/{item.kernel.id}/manifest.json", item.result.bundle)
    store.record_stage("extract-all")

    if args.json:
        _emit(
            {
                "run_id": coverage.run_id,
                "standalone_share": coverage.standalone_share,
                "rewritable_share": coverage.rewritable_share,
                "verified": coverage.verified_count,
                "kernels": [
                    {
                        "id": e.kernel.id,
                        "name": e.kernel.runtime_name,
                        "level": e.level.value if e.level else None,
                        "gpu_share": e.share,
                        "verified": e.verified,
                        "error": e.error,
                    }
                    for e in coverage.extractions
                ],
            },
            True,
        )
        return 0

    print(coverage.format())
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Build a KernelBundle, downgrading rather than guessing (§12)."""
    from xe_forge.orbit.extract import Extractor
    from xe_forge.orbit.models import CapturedInvocation
    from xe_forge.orbit.profiling.interception import LaunchLog

    if getattr(args, "all", False):
        return cmd_extract_all(args)

    store = _resolve_store(args)
    kernel = _load_kernel(store, args.kernel_id)

    launch = None
    if store.exists(LAUNCHES):
        log = store.load(LAUNCHES, LaunchLog)
        for record in log.records:
            if kernel.runtime_name in record.fq_name or record.fq_name in kernel.runtime_name:
                launch = record
                break

    inputs = None
    capture_manifest = f"captures/{kernel.id}/invocation.json"
    if store.exists(capture_manifest):
        inputs = store.load(capture_manifest, CapturedInvocation)

    agent = None
    if not args.no_agent:
        from xe_forge.orbit.agents import default_agent

        agent = default_agent()
    extractor = Extractor(output_root=store.subdir("bundles"), agent=agent)
    result = extractor.extract(kernel, launch=launch, inputs=inputs, level=args.level)
    store.save(f"bundles/{kernel.id}/manifest.json", result.bundle)
    store.record_stage("extract")

    if args.json:
        _emit(result.bundle.model_dump(mode="json"), True)
        return 0

    print(f"{kernel.id}: extracted at {result.level.value}")
    if result.downgraded:
        print(f"  downgraded from {result.downgraded_from.value} because:")
        for reason in result.reasons:
            print(f"    - {reason}")
    print(f"  entrypoint:  {result.bundle.entrypoint}")
    print(f"  closure:     {len(result.bundle.closure)} file(s)")
    if result.bundle.data_deps:
        print(f"  data deps:   {len(result.bundle.data_deps)}")
    if result.bundle.build:
        print(f"  build:       {result.bundle.build.compiler}")
    print(f"  bundle:      {store.path(f'bundles/{kernel.id}')}")
    print()
    print(f"  Not yet verified. Run: xe-orbit bundle test {kernel.id}")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """Prove a bundle is the kernel that actually ran (§12.10, §12.12)."""
    from xe_forge.orbit.extract import verify_bundle
    from xe_forge.orbit.models import KernelBundle

    store = _resolve_store(args)
    name = f"bundles/{args.kernel_id}/manifest.json"
    bundle = store.load(name, KernelBundle)

    if args.action == "show":
        _emit(bundle.model_dump(mode="json"), True)
        return 0

    report = verify_bundle(bundle)
    bundle.verification = report.to_extraction_check()
    store.save(name, bundle)

    if args.json:
        _emit(
            {
                "kernel_id": report.kernel_id,
                "level": report.level.value,
                "passed": report.passed,
                "checks": [c.__dict__ for c in report.checks],
            },
            True,
        )
    else:
        print(report.format())
        if not report.passed:
            print()
            print("An unverified bundle is never optimized. Fix the closure or accept a")
            print("downgrade to E3, where the framework drives the real dispatch.")
    return 0 if report.passed else 1


def cmd_emit(args: argparse.Namespace) -> int:
    """Write the Model + spec contract Xe-Forge consumes (§8, §PR 9)."""
    from xe_forge.orbit.emit import emit_candidate
    from xe_forge.orbit.models import KernelBundle

    store = _resolve_store(args)
    kernel = _load_kernel(store, args.kernel_id)
    bundle = store.load(f"bundles/{args.kernel_id}/manifest.json", KernelBundle)

    if not bundle.verification.verified and not args.force:
        print(
            f"bundle {args.kernel_id} is not verified; refusing to emit.\n"
            f"Run `xe-orbit bundle test {args.kernel_id}` first, or pass --force to\n"
            f"emit anyway (which risks optimizing a specialization the workload never ran).",
            file=sys.stderr,
        )
        return 1

    target = store.subdir("candidates", args.kernel_id)
    summary = emit_candidate(kernel, bundle, target, tolerance=(args.rtol, args.atol))
    store.record_stage("emit")

    if args.json:
        _emit(summary, True)
        return 0

    print(f"{kernel.id}: candidate written to {target}")
    print(f"  spec:      {summary['spec_path']}")
    print(f"  variants:  {summary['variants']}")
    print(f"  coverage:  {summary['coverage'] * 100:.1f}% of observed calls")
    if summary["dropped_shapes"]:
        print(f"  dropped:   {summary['dropped_shapes']} tail shape(s) below the weight floor")
    print(f"  reference: {summary['reference_path']} (stub — supply the eager equivalent)")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run the agentic in-place optimization loop for one kernel (§13.5).

    Orbit owns the loop; the agent answers the one question it is better at — what to
    try. Apply, verify, measure and revert are all programmatic, so the result is
    reproducible regardless of what the model said.
    """
    from xe_forge.orbit.knowledge import facts_for_kernel
    from xe_forge.orbit.models import KernelCatalog
    from xe_forge.orbit.optimize.proposer import ClaudeProposer

    store = _resolve_store(args)
    if args.apply and not args.harness:
        print(
            "--apply requires --harness: a loop without a correctness gate is what "
            "§19 forbids. Provide a script exiting 0 (correct) / 1 (wrong) / 2 "
            "(could not check).",
            file=sys.stderr,
        )
        return 2

    catalog = store.load(KERNEL_CATALOG, KernelCatalog)
    kernel = next((k for k in catalog.kernels if k.id == args.kernel_id), None)
    if kernel is None:
        print(f"no kernel {args.kernel_id!r} in run {store.run_id}")
        return 1

    print(f"target: {kernel.id} ({kernel.runtime_name[:60]})")
    print(
        f"  {kernel.gpu_time_share * 100:.2f}% of GPU time, Amdahl ceiling "
        f"{kernel.max_e2e_gain:.2f}%, MDE {catalog.minimum_detectable_effect:.2f}%"
    )
    if kernel.max_e2e_gain < catalog.minimum_detectable_effect:
        # Say this before any work is done, not in a footnote afterwards. §18 is explicit
        # that a ceiling below the MDE means the win cannot be demonstrated end to end
        # even if it is real.
        print(
            "  NOTE: the ceiling is below the minimum detectable effect, so an "
            "end-to-end verdict\n        will be INCONCLUSIVE however good the kernel "
            "gets. Kernel-level only."
        )

    if not kernel.source_file:
        # Provenance says who produced a kernel; the language backend says where it
        # lives. Resolving lazily here rather than at catalog time keeps a filesystem
        # scan out of the common path — the catalog is built for every run, and only a
        # kernel actually being optimized needs its file located.
        from xe_forge.orbit.languages import get_backend

        language = kernel.language.value if kernel.language else "triton"
        try:
            located = get_backend(language).resolve_source(kernel.runtime_name)
        except Exception:
            located = None
        if located is not None and located.file:
            kernel.source_file = located.file
            # Persist the tier alongside the path (§5.6 / G2): the record now says
            # *how* the file was found, and its confidence keeps the None-for-exact
            # semantics rather than being flattened back into a float.
            kernel.resolution_method = located.method
            kernel.provenance_confidence = located.confidence
            print(
                f"  resolved source: {located.file}"
                + (f":{located.line}" if located.line else "")
                + f"  [{located.method.value}, {located.describe_confidence()}]"
            )

    if not kernel.source_file:
        print("  no resolved source file; nothing to patch in place")
        return 1

    knowledge = "\n".join(
        f"  - {fact.description}" for fact in facts_for_kernel(kernel, store.run_id)
    )
    print("\nmeasured context handed to the proposer:")
    print(knowledge)

    if args.no_agent:
        print("\n--no-agent: no proposals generated; the gates were not exercised.")
        return 0

    proposer = ClaudeProposer()
    if not proposer.available():
        print("\nclaude is not on PATH; cannot propose (§6)")
        return 1

    # §13.5's session memory, across CLI invocations: without this, a second
    # `optimize` run re-proposes exactly what the first run measured and reverted —
    # the loop rejected them correctly and the agent never found out. The device
    # facts travel too, for the same measured reason: the first live run's failures
    # were sound reasoning applied to a device the agent had not been told about.
    history = _prior_trials(store, kernel.id)
    if history:
        knowledge += (
            "\n\nWHAT A PREVIOUS RUN ALREADY TRIED AND MEASURED ON THIS EXACT DEVICE"
            " (measurements, not opinions — do not re-propose these; a direction that"
            " measured worse is evidence the opposite direction may be the move):\n" + history
        )
    try:
        from xe_forge.orbit.device import probe_device

        facts = probe_device()
        if facts.available:
            knowledge += "\n\nTHE DEVICE:\n" + facts.describe()
    except Exception:
        pass

    source = Path(kernel.source_file).read_text(encoding="utf-8")
    print(f"\nplanning up to {args.trials} candidates...")
    proposals = proposer.plan(source, knowledge, args.trials, kernel.runtime_name[:40])
    if not proposals:
        print("  the proposer returned no usable candidates")
        return 1
    for i, proposal in enumerate(proposals):
        print(f"  {i + 1}. {proposal.title}")
        print(f"     {proposal.rationale}")

    if not args.apply:
        print(
            "\nProposals only. Pass --apply (with --harness, and --measure to compare) "
            "to run the §13.5 loop: implement, apply, verify, measure, keep or revert."
        )
        return 0

    # The knowledge string travels into the loop because rounds after the first
    # re-plan with it: same measured context, plus what this session has since learned.
    return _run_optimize_loop(args, store, kernel, proposer, proposals, knowledge=knowledge)


def _prior_trials(store, kernel_id: str) -> str:
    """Render the previous loop run's verdicts for the proposer, or "".

    Reads the `loop_result.json` the last `--apply` run persisted. Only the most
    recent round is carried — the file is overwritten per run — which keeps the
    prompt bounded and the freshest evidence in front of the model.
    """
    path = store.subdir("experiments", kernel_id) / "loop_result.json"
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    lines = []
    for trial in payload.get("trials", []):
        delta = trial.get("delta_percent")
        measured = f"{delta:+.2f}%" if delta is not None else "not measured"
        lines.append(f"  - {trial.get('title')}: {trial.get('verdict')} at {measured}")
        if trial.get("reason"):
            lines.append(f"      {trial['reason']}")
    return "\n".join(lines)


def _implement_round(
    proposer, proposals, target, experiments: Path, harness: Path, round_index: int = 0
) -> int:
    """IMPLEMENT one round's proposals, each in its own workspace *copy* (§13.5).

    A session that goes wrong cannot leave the tree broken, and the agent sees its own
    tracebacks. Round 1 keeps the historical `proposal_<i>` workspace names so a
    single-round invocation is unchanged on disk; later rounds are namespaced per round
    so a re-planned session cannot overwrite the evidence of an earlier one.
    """
    prefix = "proposal" if round_index == 0 else f"round{round_index + 1}_proposal"
    implemented = 0
    for index, proposal in enumerate(proposals):
        workspace = experiments / f"{prefix}_{index}"
        print(f"\nimplementing {index + 1}/{len(proposals)}: {proposal.title}")
        proposal.new_source = proposer.implement(proposal, target, workspace, harness=harness)
        if proposal.new_source is None:
            print("  no edit produced; the loop will refuse this proposal")
        else:
            implemented += 1
    return implemented


def _plan_next_round(
    proposer,
    kernel,
    target: Path,
    knowledge: str,
    history,
    round_index: int,
    rounds: int,
    count: int,
):
    """PLAN a round after the first: the measured context, plus what this session learned.

    The history travels through `plan()`'s `knowledge` parameter, which the proposer
    interpolates into the prompt as MEASURED CONTEXT — and measured context is exactly
    what a round's verdicts are. Feeding the seam that already exists keeps the
    proposer unchanged and the layering honest (§13.7).
    """
    print(f"\n--- round {round_index + 1}/{rounds} ---")
    learned = history.render_for_knowledge()
    context = knowledge.rstrip()
    round_knowledge = f"{context}\n\n{learned}" if context else learned
    # Re-read rather than reuse: the file on disk after a round of reverts is the
    # loop's ground truth, and the source the next plan sees must be the same one.
    source = target.read_text(encoding="utf-8")
    print(f"planning up to {count} candidates from this session's measurements...")
    proposals = proposer.plan(source, round_knowledge, count, kernel.runtime_name[:40])
    for i, proposal in enumerate(proposals):
        print(f"  {i + 1}. {proposal.title}")
        print(f"     {proposal.rationale}")
    return proposals


def _session_summary(history, stopped_because: str) -> str:
    """The whole session in three lines: what ran, why it stopped, what won."""
    lines = [
        f"session: {len(history.rounds)} round(s) run",
        f"  stopped: {stopped_because}",
    ]
    best = history.best_so_far
    if best is not None and best.delta_percent is not None:
        lines.append(
            f"  best measured: {best.proposal.title} at {best.delta_percent:+.2f}% "
            f"(verdict {best.verdict.value})"
        )
    else:
        lines.append("  best measured: nothing produced a usable measurement")
    return "\n".join(lines)


def _run_optimize_loop(args, store, kernel, proposer, proposals, knowledge: str = "") -> int:
    """Drive §13.5's loop from the CLI (gap G1), for one round or several.

    The agent has proposed; everything from here is programmatic. Each proposal is
    implemented in an isolated workspace *copy*, then applied through the journalled
    patcher and driven through the gates — novelty, sandbox, apply, correctness,
    measure — with Orbit owning every verdict. Only the winner stays on disk.

    With `--rounds N` above 1 the plan -> implement -> trial cycle repeats, and each
    round's measured verdicts — with their reasons — go back to the proposer as
    session history (§13.7). That closes the "the agent never found out" seam
    in-process, where `_prior_trials` closes it only across separate invocations. The
    patcher and the novelty ledger are created once and shared by every round: novelty
    is memory, and memory that resets each round would readmit in round 2 exactly what
    round 1 already measured (§20.4).
    """
    from xe_forge.orbit.novelty import NoveltyLedger
    from xe_forge.orbit.optimize.harness import run_harness
    from xe_forge.orbit.optimize.loop import OptimizationLoop
    from xe_forge.orbit.optimize.session import RoundOutcome, SessionHistory
    from xe_forge.orbit.patch.inplace import InPlacePatcher

    target = Path(kernel.source_file)
    harness = Path(args.harness).resolve()
    if not harness.is_file():
        print(f"--harness {harness} does not exist", file=sys.stderr)
        return 1

    # Callers that construct the namespace directly may omit `rounds`; that means the
    # single-round default, not an error.
    rounds = max(1, int(getattr(args, "rounds", 1) or 1))
    multi_round = rounds > 1

    experiments = store.subdir("experiments", kernel.id)
    if multi_round:
        print(f"\n--- round 1/{rounds} ---")
    if not _implement_round(proposer, proposals, target, experiments, harness):
        print("\nno proposal produced an edit; nothing to trial")
        return 1

    sandbox = [target.parent] + [Path(p) for p in (args.sandbox or [])]
    patcher = InPlacePatcher(journal_dir=store.run_dir, sandbox_roots=sandbox)

    measure = (lambda: _measure_once(args.measure)) if args.measure else (lambda: None)
    measure_samples = None
    if args.measure and args.samples >= 5:

        def measure_samples():
            values: list[float] = []
            for _ in range(args.samples):
                value = _measure_once(args.measure)
                if value is not None:
                    values.append(value)
            return values

    elif args.measure and 0 < args.samples < 5:
        print(
            f"\n--samples {args.samples} is below the 5-repetition floor (§17); "
            "falling back to a single measurement and the fixed floor"
        )
    if not args.measure:
        print(
            "\nno --measure command: candidates will be checked for correctness, but "
            "without a comparison none can be accepted"
        )

    loop = OptimizationLoop(
        target=target,
        patcher=patcher,
        check=lambda: run_harness(harness),
        measure=measure,
        ledger=NoveltyLedger(),
        min_improvement_percent=args.min_improvement,
        measure_samples=measure_samples,
    )

    history = SessionHistory()
    result = None
    stopped_because = ""
    for round_index in range(rounds):
        if round_index > 0:
            proposals = _plan_next_round(
                proposer, kernel, target, knowledge, history, round_index, rounds, args.trials
            )
            if not proposals:
                stopped_because = f"the proposer returned no proposals for round {round_index + 1}"
                print(f"\n{stopped_because}; stopping the session")
                break
            if not _implement_round(proposer, proposals, target, experiments, harness, round_index):
                stopped_because = f"round {round_index + 1} produced no implementable edit"
                print(f"\n{stopped_because}; stopping the session")
                break

        result = loop.run(proposals)
        store.record_stage("optimize")
        history.rounds.append(RoundOutcome(index=round_index, trials=result.trials))
        if multi_round:
            _write_loop_result(experiments / f"loop_result_round{round_index + 1}.json", result)

        print()
        print(result.format())
        if result.kept:
            # The loop's own semantics leave the winner applied; a further round would
            # only be spending budget to beat a result the operator has not yet seen.
            stopped_because = f"round {round_index + 1} produced a KEPT verdict"
            break
    else:
        stopped_because = f"all {rounds} round(s) exhausted without an accepted candidate"

    # The final `loop_result.json` is the last round that ran, so `_prior_trials` — the
    # cross-invocation memory — keeps reading the freshest evidence unchanged.
    result_path = experiments / "loop_result.json"
    _write_loop_result(result_path, result)

    print(f"\ntrial record: {result_path}")
    if result.kept:
        print(
            "the accepted patch is applied and journalled: `xe-orbit patch status` "
            "shows it, `xe-orbit patch recover` restores the original"
        )
    if multi_round:
        print()
        print(_session_summary(history, stopped_because))
    return 0


def _measure_once(command: str, timeout_s: float = 600.0) -> float | None:
    """Run a measurement command and read a duration in microseconds from it.

    Contract: the command prints a float — the last whitespace-separated token of its
    last non-empty stdout line — in microseconds, lower is better. A failed run
    returns None and is dropped upstream: a failed measurement is missing data, not a
    fast one.
    """
    import subprocess

    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    for line in reversed((completed.stdout or "").splitlines()):
        tokens = line.split()
        if not tokens:
            continue
        try:
            return float(tokens[-1])
        except ValueError:
            return None
    return None


def _write_loop_result(path: Path, result) -> None:
    """Persist the loop's verdicts as an artifact (§23), without the source bytes."""
    payload = {
        "kernel_id": result.kernel_id,
        "baseline_us": result.baseline_us,
        "accepted": result.accepted.proposal.title if result.accepted else None,
        "notes": result.notes,
        "trials": [
            {
                "index": t.index,
                "title": t.proposal.title,
                "verdict": t.verdict.value,
                "reason": t.reason,
                "accuracy": t.accuracy,
                "baseline_us": t.baseline_us,
                "candidate_us": t.candidate_us,
                "delta_percent": t.delta_percent,
            }
            for t in result.trials
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cmd_patch(args: argparse.Namespace) -> int:
    """Show or undo in-place edits left in the installed tree (§13.6).

    An orphaned journal entry is the only on-disk evidence that a previous run modified
    an installed tree and never put it back. Left alone, the next run measures a patched
    baseline and reports it as clean — a wrong number produced by a correct-looking
    pipeline, which is exactly what this project refuses to ship.
    """
    from xe_forge.orbit.patch.inplace import InPlacePatcher, RecoveryOutcome

    store = _resolve_store(args)
    patcher = InPlacePatcher(journal_dir=store.run_dir)

    if args.action == "status":
        print(patcher.format())
        return 0

    outcomes = patcher.recover(force=args.force)
    if not outcomes:
        print("nothing to recover; the installed tree is unmodified")
        return 0

    conflicts = 0
    for target, outcome in outcomes.items():
        print(f"  [{outcome.value:>13}] {target}")
        if outcome is RecoveryOutcome.CONFLICT:
            conflicts += 1

    if conflicts:
        print(
            f"\n{conflicts} file(s) changed since we patched them, so restoring would "
            f"discard that change. Inspect them, then re-run with --force if the change "
            f"was ours."
        )
        return 1
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Patch a candidate back into the workload, highest rung first (§13)."""
    from xe_forge.orbit.adapters import resolve_adapter
    from xe_forge.orbit.models import WorkloadSpec
    from xe_forge.orbit.patch import PatchError, apply_patch

    store = _resolve_store(args)
    kernel = _load_kernel(store, args.kernel_id)

    spec = (
        store.load(WORKLOAD, WorkloadSpec)
        if store.exists(WORKLOAD)
        else WorkloadSpec(command=["true"])
    )
    adapter = resolve_adapter(spec, args.framework)
    points = adapter.patch_points(kernel)

    try:
        record = apply_patch(
            kernel,
            points,
            candidate_module=args.module or f"candidates.{kernel.id}.kernel",
            output_dir=store.subdir("patches", kernel.id),
        )
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    store.save(f"patches/{kernel.id}/patch.json", record)
    store.record_stage("apply")

    if args.json:
        _emit(record.model_dump(mode="json"), True)
        return 0

    print(f"{kernel.id}: rung {record.rung} — {record.mechanism}")
    print(f"  target:   {record.target}")
    if record.module_path:
        print(f"  module:   {record.module_path}")
    print(f"  applied:  {record.applied}")
    print(f"  revert:   {record.revert_procedure}")
    for note in record.notes:
        print(f"  note:     {note}")
    print()
    print("  Not yet verified. A patch is not applied until re-profiling shows the new")
    print("  kernel present AND the old one absent — run `xe-orbit compare` (§13).")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run the correctness ladder and the acceptance decision (§17, §19, §14.3)."""
    from xe_forge.orbit.compare import decide_matrix, run_ladder
    from xe_forge.orbit.models import KernelBundle, WorkloadMatrix

    store = _resolve_store(args)

    if args.matrix:
        matrix = WorkloadMatrix.model_validate_json(Path(args.matrix).read_text(encoding="utf-8"))
        samples_path = Path(args.samples) if args.samples else None
        if samples_path is None or not samples_path.is_file():
            print(
                "error: --matrix requires --samples pointing at a JSON object mapping\n"
                "       profile id -> [baseline_samples, candidate_samples]",
                file=sys.stderr,
            )
            return 1
        raw = json.loads(samples_path.read_text(encoding="utf-8"))
        samples = {k: (v[0], v[1]) for k, v in raw.items()}
        decision = decide_matrix(matrix, samples, regression_threshold_percent=args.regression)
        if args.json:
            _emit(
                {
                    "decision": decision.decision.value,
                    "weighted_improvement": decision.weighted_improvement,
                    "reason": decision.reason,
                    "profiles": [o.__dict__ for o in decision.outcomes],
                },
                True,
            )
        else:
            print(decision.format())
        return 0 if decision.decision.value == "ACCEPT" else 1

    kernel = _load_kernel(store, args.kernel_id)
    bundle_name = f"bundles/{kernel.id}/manifest.json"
    verified = False
    detail = "no bundle extracted"
    if store.exists(bundle_name):
        bundle = store.load(bundle_name, KernelBundle)
        verified = bundle.verification.verified
        detail = (
            "bundle verified"
            if verified
            else "; ".join(bundle.verification.failures) or "bundle unverified"
        )

    ladder = run_ladder(
        kernel_id=kernel.id,
        build_ok=True,
        build_detail="candidate present",
        extraction_verified=verified,
        extraction_detail=detail,
        correctness_ok=None,
        kernel_samples=None,
        e2e_samples=None,
        reprofile_changed=None,
    )

    if args.json:
        _emit(
            {
                "kernel_id": ladder.kernel_id,
                "passed": ladder.passed,
                "failed_at": ladder.failed_at.value if ladder.failed_at else None,
                "gates": [
                    {
                        "gate": r.gate.value,
                        "passed": r.passed,
                        "skipped": r.skipped,
                        "detail": r.detail,
                    }
                    for r in ladder.results
                ],
            },
            True,
        )
    else:
        print(ladder.format())
    return 0 if ladder.passed else 1


def cmd_arena(args: argparse.Namespace) -> int:
    """A/B engines on a shared task set, outside the optimization loop (§5.3, §5.4).

    Discovery, availability, execution and reporting are all honest about their
    limits: an unavailable contestant is skipped with the reason, a task without
    held-out variants says its generalization gap is unmeasurable, and the
    leaderboard never ranks a measured mean against an unmeasured one.
    """
    from xe_forge.orbit.arena import ArenaError, build_contestants, discover_tasks, run_arena

    task_root = Path(args.task_root)
    try:
        tasks = discover_tasks(task_root)
    except ArenaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print(
            f"no tasks under {task_root}: a task is a subdirectory with kernel.py "
            f"and spec.yaml (see `xe-orbit emit`)"
        )
        return 1

    names = [name.strip() for name in args.contestants.split(",") if name.strip()]
    contestants, skipped = build_contestants(names)
    for name, reason in skipped.items():
        print(f"skipped contestant: {name} ({reason})")
    if not contestants:
        print("no contestant is available; nothing to run")
        return 1

    for task in tasks:
        for note in task.notes:
            print(f"note [{task.task_id}]: {note}")

    arena_dir = Path(args.arena_dir) if args.arena_dir else task_root / ".arena"
    report = run_arena(tasks, contestants, arena_dir, resume=not args.no_resume, skipped=skipped)

    if getattr(args, "json", False):
        _emit(report.summary(), True)
    else:
        print(report.format())

    summary_path = arena_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Show serving profiles, weights and derived shapes (§14.5)."""
    from xe_forge.orbit.models import ServingProfile, WorkloadMatrix

    if args.path:
        matrix = WorkloadMatrix.model_validate_json(Path(args.path).read_text(encoding="utf-8"))
    else:
        # The example matrix from §14.3, so `matrix show` is useful with no file.
        matrix = WorkloadMatrix(
            profiles=[
                ServingProfile(
                    id="qwen3-8b-chat",
                    model="Qwen/Qwen3-8B",
                    isl=1024,
                    osl=256,
                    concurrency=32,
                    weight=0.5,
                ),
                ServingProfile(
                    id="qwen3-8b-longctx",
                    model="Qwen/Qwen3-8B",
                    isl=8192,
                    osl=128,
                    concurrency=8,
                    weight=0.3,
                ),
                ServingProfile(
                    id="qwen3-8b-lowlat",
                    model="Qwen/Qwen3-8B",
                    isl=512,
                    osl=512,
                    concurrency=1,
                    weight=0.2,
                ),
            ]
        )

    weights = matrix.normalized_weights()
    if args.json:
        _emit(matrix.model_dump(mode="json"), True)
        return 0

    print(f"{'PROFILE':<22} {'MODEL':<18} {'ISL':>6} {'OSL':>6} {'CONC':>5} {'WEIGHT':>7}")
    print("-" * 70)
    for profile in matrix.profiles:
        print(
            f"{profile.id:<22} {profile.model[:18]:<18} {profile.isl:>6} "
            f"{profile.osl:>6} {profile.concurrency:>5} {weights[profile.id]:>7.2f}"
        )
    print()
    print("Acceptance requires a weighted win AND no per-profile regression (§14.3):")
    print("a candidate that wins decode and loses prefill is a trade, not an improvement.")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    """Show which Intel SYCL kernel source trees are available (§11.2)."""
    from xe_forge.orbit.languages.sources import candidate_roots, discover

    registry = discover()
    if args.json:
        _emit(
            {
                "trees": [
                    {
                        "name": t.name,
                        "path": str(t.path),
                        "provider": t.provider,
                        "symbols": t.symbol_count,
                    }
                    for t in registry.trees
                ],
                "missing": registry.missing,
            },
            True,
        )
        return 0

    print(registry.format())
    print()
    print("searched:", ", ".join(str(r) for r in candidate_roots()) or "(no roots found)")
    print("override with ORBIT_SYCL_SOURCES=/path/to/checkouts")
    if args.resolve:
        print()
        found = registry.resolve(args.resolve)
        if found:
            path, tree, symbol = found
            print(
                f"{args.resolve[:60]}\n  symbol: {symbol}\n  tree:   {tree.name}\n  file:   {path}"
            )
        else:
            print(f"{args.resolve[:60]}\n  unresolved in the indexed trees")
    return 0


def cmd_support_matrix(args: argparse.Namespace) -> int:
    """Publish the measured support matrix (§5.3, §6)."""
    from xe_forge.orbit.support import build_matrix, format_languages

    matrix = build_matrix()
    if args.json:
        _emit([r.__dict__ for r in matrix.rows], True)
        return 0

    print(matrix.format())
    print()
    print(format_languages())
    print()
    print("Versions are pinned, not chased: a change to any of them invalidates stored")
    print("bundles and measurements and forces re-extraction (§12.9).")
    return 0


def cmd_regions(args: argparse.Namespace) -> int:
    """Fusable multi-kernel regions and their Xe-Fuse handoff (§7.3, §12.11)."""
    from xe_forge.orbit.analysis.regions import detect_regions, format_regions
    from xe_forge.orbit.models import KernelCatalog
    from xe_forge.orbit.profiling.trace import TraceEvents

    store = _resolve_store(args)
    events = store.load(EVENTS, TraceEvents)
    catalog = store.load(KERNEL_CATALOG, KernelCatalog)

    regions = detect_regions(events, catalog.kernels, total_gpu_time_us=catalog.total_gpu_time_us)
    catalog.regions = regions
    store.save(KERNEL_CATALOG, catalog)
    store.save_json(
        "regions/catalog.json",
        {"run_id": store.run_id, "regions": [r.model_dump(mode="json") for r in regions]},
    )
    store.record_stage("regions")

    if args.json:
        _emit([r.model_dump(mode="json") for r in regions], True)
        return 0

    if not regions:
        print("no fusable regions found: the largest wins come from eliminating kernels,")
        print("so a trace with no producer-consumer chains has nothing for Xe-Fuse to do.")
        return 0

    # `format_regions` already states whether Xe-Fuse is reachable, so the routing
    # verdict is not repeated here.
    print(format_regions(regions, limit=args.limit))
    return 0


def _region_gemm_shapes(region, kernels) -> tuple[int, int, int] | None:
    """Derive (m, n, k) from the region's GEMM member, or None when ambiguous.

    Only the clean case is derived — two 2-D inputs sharing exactly one dimension,
    read from the dominant observed shape. Anything else returns None and the caller
    asks for --shapes: a guessed problem size would autotune a kernel the workload
    never runs, which is §12.10's wrong-specialization failure by another road.
    """
    by_id = {kernel.id: kernel for kernel in kernels}
    for kernel_id in region.kernel_ids:
        record = by_id.get(kernel_id)
        if record is None or "gemm" not in record.runtime_name.lower():
            continue
        if not record.shapes:
            return None
        dims = record.shapes[0].dims
        a = [value for key, value in sorted(dims.items()) if key.startswith("a0_")]
        b = [value for key, value in sorted(dims.items()) if key.startswith("a1_")]
        if len(a) != 2 or len(b) != 2:
            return None
        m, k = a
        if k == b[0] and b[1] != k:
            return m, b[1], k
        if k == b[1] and b[0] != k:
            return m, b[0], k
        return None
    return None


def cmd_fuse(args: argparse.Namespace) -> int:
    """Autotune a region's fused kernel through Xe-Fuse (§13.4, §11.7).

    Fully automated and deterministic: pattern → preset, shapes from the trace (or
    --shapes), a tile sweep with every result kept, the winner named. The output is
    a timing table, not an acceptance — correctness stays gated by the differential
    check before any comparison built on these numbers is believed.
    """
    from xe_forge.orbit.models import KernelCatalog
    from xe_forge.orbit.optimize.fusion import task_from_region
    from xe_forge.orbit.optimize.xe_fuse_executor import (
        autotune_region,
        checkout_available,
    )

    store = _resolve_store(args)
    catalog = store.load(KERNEL_CATALOG, KernelCatalog)
    region = next((r for r in catalog.regions if r.id == args.region_id), None)
    if region is None:
        print(f"no region {args.region_id!r} in run {store.run_id}; run `xe-orbit regions` first")
        return 1

    if not checkout_available():
        print(
            "Xe-Fuse / sycl-tla checkouts not found; clone them under a source root "
            "or set ORBIT_XE_FUSE_DIR / SYCL_TLA_DIR"
        )
        return 1

    if args.shapes:
        try:
            m, n, k = (int(part) for part in args.shapes.lower().split("x"))
        except ValueError:
            print(f"--shapes must be MxNxK, got {args.shapes!r}")
            return 2
    else:
        derived = _region_gemm_shapes(region, catalog.kernels)
        if derived is None:
            print(
                "could not derive (m, n, k) unambiguously from the region's recorded "
                "shapes; pass --shapes MxNxK (a guessed size would tune a kernel the "
                "workload never runs)"
            )
            return 1
        m, n, k = derived

    task = task_from_region(region, catalog.kernels)
    print(
        f"autotuning {task.pattern} at {m}x{n}x{k} "
        f"(region {region.id}, {region.gpu_time_share * 100:.1f}% of GPU time)"
    )
    out_dir = store.subdir("experiments", region.id) / "xe_fuse"
    sweep = autotune_region(
        task,
        (m, n, k),
        out_dir,
        tiles=args.tiles.split(",") if args.tiles else None,
        iterations=args.iterations,
    )
    store.save_json(
        f"experiments/{region.id}/xe_fuse_sweep.json",
        {
            "region": region.id,
            "pattern": task.pattern,
            "preset": sweep.preset,
            "shapes": [m, n, k],
            "results": [
                {
                    "tile": r.tile,
                    "us": r.per_iteration_us,
                    "tflops": r.tflops,
                    "error": r.error,
                }
                for r in sweep.results
            ],
            "best_tile": sweep.best.tile if sweep.best else None,
        },
    )
    store.record_stage("fuse")

    print()
    print(sweep.format())
    if sweep.best is not None:
        print()
        print(
            "timing only: gate correctness (the differential check) before comparing "
            "this against the unfused chain — the Xe-Fuse binary's own --verify is "
            "inert upstream."
        )
    return 0 if sweep.best is not None else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the full loop over a run's artifacts (§24, PR 13)."""
    from xe_forge.orbit.pipeline import run_pipeline

    store = _resolve_store(args)
    result = run_pipeline(
        store,
        kernel_id=args.kernel_id,
        extraction_level=args.level,
        stop_before_optimize=not args.optimize,
    )
    store.record_stage("pipeline")

    if args.json:
        _emit(
            {
                "run_id": result.run_id,
                "kernel_id": result.kernel_id,
                "completed": result.completed,
                "stop_reason": result.stop_reason,
                "stages": [s.__dict__ for s in result.stages],
            },
            True,
        )
    else:
        print(result.format())
    # A deliberate stop is not a failure: NO_ACTION is a first-class result.
    return 0


def cmd_conformance(args: argparse.Namespace) -> int:
    """Run the adapter conformance suite (§10.7)."""
    from xe_forge.orbit.adapters import get_adapter
    from xe_forge.orbit.adapters.conformance import run_conformance

    adapter = get_adapter(args.adapter)
    report = run_conformance(adapter, repetitions=args.repetitions, quick=args.quick)
    if args.json:
        _emit(
            {
                "adapter": report.adapter,
                "tier": report.tier,
                "passed": report.passed,
                "checks": [c.__dict__ for c in report.checks],
            },
            True,
        )
    else:
        print(report.format())
    return 0 if report.passed else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    """Full-loop invariants on stub data, no LLM and no GPU (§15)."""
    from xe_forge.orbit.selftest import run_selftest

    report = run_selftest(chaos=args.chaos, quick=args.quick)
    if args.json:
        _emit(
            {"passed": report.passed, "results": [r.__dict__ for r in report.results]},
            True,
        )
    else:
        print(report.format())
    return 0 if report.passed else 1


def cmd_schemas(args: argparse.Namespace) -> int:
    """Export the versioned artifact schemas (§16.2)."""
    from xe_forge.orbit import schemas

    if args.export:
        written = schemas.export_schemas(Path(args.export) if args.export != "-" else None)
        for path in written:
            print(path)
        return 0

    if args.name:
        _emit(schemas.schema_for(args.name), True)
        return 0

    for name, model in sorted(schemas.ARTIFACT_MODELS.items()):
        print(f"{name:<14} {model.__name__}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """List recorded runs, so `--replay` has something to name."""
    base = _orbit_base(args)
    runs = RunStore.list_runs(base=base)
    if not runs:
        print(f"no runs recorded under {base}")
        return 0
    for run_id in runs:
        store = RunStore(base, run_id)
        stages = ""
        if store.exists(MANIFEST):
            try:
                manifest = store.load(MANIFEST, RunManifest)
                stages = ", ".join(manifest.stages_completed)
            except ArtifactError:
                stages = "(unreadable manifest)"
        print(f"{run_id}  {stages}")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Verify a stored capture round-trips with strides preserved (§7.5, §PR 6)."""
    from xe_forge.orbit.capture import verify_roundtrip
    from xe_forge.orbit.models import CapturedInvocation

    store = _resolve_store(args)
    name = f"captures/{args.kernel_id}/invocation.json"
    invocation = store.load(name, CapturedInvocation)

    problems = verify_roundtrip(invocation)
    if problems:
        print(f"capture for {args.kernel_id} does NOT round-trip:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"capture for {args.kernel_id}: {len(invocation.tensors)} tensors round-trip faithfully")
    non_contiguous = [n for n, c in invocation.contiguous_map.items() if not c]
    if non_contiguous:
        print(f"  non-contiguous inputs preserved: {', '.join(non_contiguous)}")
    if invocation.data_deps:
        print(f"  data dependencies: {len(invocation.data_deps)}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd", help="working directory for the workload")
    parser.add_argument("--env", action="append", help="KEY=VALUE, repeatable")
    parser.add_argument(
        "--framework",
        default="auto",
        help="adapter to use: auto, generic_torch, or a registered name",
    )
    parser.add_argument("--device", choices=["xpu", "cuda", "cpu"], help="preferred device")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations (discarded)")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="measured repetitions; fewer than 5 cannot support an accept/reject decision",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="workload command after --")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xe-orbit",
        description="Xe-Orbit — workload-level performance optimization for Xe-Forge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xe-orbit frameworks                       # adapters, tiers, capabilities
  xe-orbit run -- python bench.py           # baseline with repetitions and a CI
  xe-orbit trace --from-trace trace.json    # ingest an existing Chrome trace
  xe-orbit kernels --run <id>               # catalog, gating verdict, ranking
  xe-orbit inspect k0 --run <id>            # provenance and headroom for one kernel
  xe-orbit arena tasks/ --contestants dspy  # engine A/B outside the loop (§5.4)
  xe-orbit selftest                         # full-loop invariants, no GPU, no LLM
  xe-orbit conformance generic_torch        # adapter conformance suite
""",
    )
    parser.add_argument("--orbit-dir", help="artifact root (default: ./.orbit)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--debug", action="store_true", help="verbose logging")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("frameworks", help="list adapters, tiers and capabilities")
    p.set_defaults(func=cmd_frameworks)

    p = sub.add_parser("run", help="baseline: environment, versions, device, timing")
    _add_workload_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("trace", help="profile a workload and normalize its trace")
    _add_workload_args(p)
    p.add_argument("--from-trace", help="ingest an existing Chrome trace file instead of running")

    p.add_argument(
        "--wrap",
        action="store_true",
        help="run the workload under the shipped torch.profiler wrapper so it needs "
        "no profiler code of its own (single-process torch workloads; framework "
        "engines use their own hook)",
    )
    p.add_argument("--replay", help="re-read a stored trace from this run id")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("kernels", help="kernel catalog with gating and ranking")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run; re-run this stage from artifacts")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_kernels)

    p = sub.add_parser("inspect", help="provenance, shapes and headroom for one kernel")
    p.add_argument("kernel_id")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("capture", help="verify a captured invocation round-trips")
    p.add_argument("kernel_id")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("extract", help="build a KernelBundle for one kernel, or --all")
    p.add_argument("kernel_id", nargs="?", default=None)
    p.add_argument(
        "--all",
        action="store_true",
        help="extract every kernel in the run and report coverage by GPU time",
    )
    p.add_argument(
        "--no-agent",
        action="store_true",
        help="skip repo-agent resolution for symbols the deterministic path cannot parse",
    )
    p.add_argument(
        "--level",
        default="auto",
        choices=["auto", "E1", "E2", "E3", "E4"],
        help="auto attempts the best level the provenance supports and downgrades on failure",
    )
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("bundle", help="verify or inspect an extracted bundle")
    p.add_argument("action", choices=["test", "verify", "show"])
    p.add_argument("kernel_id")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("emit", help="write the Model + spec contract for a verified bundle")
    p.add_argument("kernel_id")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.add_argument("--rtol", type=float, default=1e-3, help="tightened relative tolerance")
    p.add_argument("--atol", type=float, default=1e-6, help="tightened absolute tolerance")
    p.add_argument("--force", action="store_true", help="emit even if the bundle is unverified")
    p.set_defaults(func=cmd_emit)

    p = sub.add_parser("sources", help="Intel SYCL kernel source trees available for extraction")
    p.add_argument("--resolve", help="resolve a demangled kernel name to its source file")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser(
        "support-matrix", help="measured support matrix: versions, devices, languages"
    )
    p.set_defaults(func=cmd_support_matrix)

    p = sub.add_parser("regions", help="fusable multi-kernel regions and their handoff")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_regions)

    p = sub.add_parser("fuse", help="autotune a region's fused kernel via Xe-Fuse (§13.4, §11.7)")
    p.add_argument("region_id")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.add_argument("--shapes", help="problem size MxNxK; default: derived from the trace")
    p.add_argument("--tiles", help="comma-separated tiles (default: candidates tracking M)")
    p.add_argument("--iterations", type=int, default=300, help="benchmark iterations per tile")
    p.set_defaults(func=cmd_fuse)

    p = sub.add_parser("pipeline", help="run the full loop over a run's artifacts")
    p.add_argument("kernel_id", nargs="?", default=None, help="target a specific kernel")
    p.add_argument("--level", default="auto", choices=["auto", "E1", "E2", "E3", "E4"])
    p.add_argument(
        "--optimize",
        action="store_true",
        help="also invoke the optimizer (costs tokens and GPU time; off by default)",
    )
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("apply", help="patch a candidate back into the workload")
    p.add_argument("kernel_id")
    p.add_argument("--module", help="python module providing optimized_kernel")
    p.add_argument("--framework", default="auto")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("optimize", help="agentic in-place optimization loop for one kernel (§13.5)")
    p.add_argument("kernel_id")
    p.add_argument("--trials", type=int, default=3, help="how many candidates to try")
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="with --apply, run up to this many plan -> implement -> trial rounds, "
        "feeding each round's measured verdicts and reasons back to the proposer; "
        "stops early on a KEPT candidate or when the proposer has nothing to propose",
    )
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument(
        "--no-agent",
        action="store_true",
        help="plan and implement steps are skipped; use for testing the gates alone",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="run the full §13.5 loop: implement each proposal in a workspace, then "
        "apply, verify, measure and keep or revert through the journalled patcher. "
        "Without it, planning stops at ranked proposals.",
    )
    p.add_argument(
        "--harness",
        help="correctness script run in a fresh process per trial; exits 0 (correct), "
        "1 (wrong), 2 (could not check). Required with --apply (§19).",
    )
    p.add_argument(
        "--measure",
        help="shell command printing a duration in microseconds (last token of the "
        "last stdout line); run once per sample. Without it, candidates are checked "
        "but none can be accepted.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help=">=5 collects this many measurements per arm and decides with the §17 "
        "statistics (INCONCLUSIVE is a real outcome); otherwise one measurement "
        "against the --min-improvement floor",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=1.0,
        help="fallback accept floor in percent, used only when --samples is not (§17.5.2)",
    )
    p.add_argument(
        "--sandbox",
        action="append",
        help="additional root the patcher may write under (repeatable); the target's "
        "own directory is always allowed",
    )
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("patch", help="outstanding in-place edits, and how to undo them")
    p.add_argument("action", choices=["status", "recover"], nargs="?", default="status")
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument(
        "--force",
        action="store_true",
        help="restore even where the file changed since we patched it, discarding that change",
    )
    p.set_defaults(func=cmd_patch)

    p = sub.add_parser("compare", help="run the correctness ladder and acceptance decision")
    p.add_argument("kernel_id", nargs="?", default=None)
    p.add_argument("--matrix", help="serving matrix JSON; enables per-profile acceptance")
    p.add_argument("--samples", help="JSON: profile id -> [baseline, candidate] samples")
    p.add_argument(
        "--regression",
        type=float,
        default=2.0,
        help="per-profile regression threshold in percent (§14.3)",
    )
    p.add_argument("--run", help="run id (default: most recent)")
    p.add_argument("--replay", help="alias for --run")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser(
        "arena", help="A/B engines on one task set, outside the optimization loop (§5.4)"
    )
    p.add_argument(
        "task_root",
        help="directory whose candidate subdirectories (kernel.py + spec.yaml) are the tasks",
    )
    p.add_argument(
        "--contestants",
        default="dspy,claude",
        help="comma-separated engine names (default: dspy,claude)",
    )
    p.add_argument(
        "--arena-dir",
        help="workspace and result root (default: <task_root>/.arena)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run pairs that already have a persisted result.json",
    )
    p.set_defaults(func=cmd_arena)

    p = sub.add_parser("matrix", help="show serving profiles, weights and derived shapes")
    p.add_argument("path", nargs="?", default=None, help="matrix JSON (default: the §14 example)")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("conformance", help="run the adapter conformance suite")
    p.add_argument("adapter", nargs="?", default="generic_torch")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--quick", action="store_true", help="skip the null and control tests")
    p.set_defaults(func=cmd_conformance)

    p = sub.add_parser("selftest", help="run the full-loop invariants on stub data")
    p.add_argument("--chaos", action="store_true", help="add failure-injection checks")
    p.add_argument("--quick", action="store_true", help="skip subprocess-based checks")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("schemas", help="list or export versioned artifact schemas")
    p.add_argument("--name", help="print the schema for one artifact")
    p.add_argument("--export", help="write all schemas to this directory ('-' for default)")
    p.set_defaults(func=cmd_schemas)

    p = sub.add_parser("runs", help="list recorded runs")
    p.set_defaults(func=cmd_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # argparse.REMAINDER keeps the `--` separator; drop it so the command is clean.
    command = getattr(args, "command", None)
    if command and command[0] == "--":
        args.command = command[1:]

    try:
        return int(args.func(args) or 0)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
