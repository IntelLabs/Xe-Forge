"""Orchestration of one vISA lowering run.

:class:`LoweringSession` prepares a kernel for lowering and evaluates attempts:
capture the launch, build the ABI stub, derive the contract, plan the
verification cases. :class:`LoweringPipeline` adds the agent, the trial tree
and the run record on top. Nothing here touches the device directly; device
work happens in :mod:`xe_forge.lowering.visa.runner` children.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xe_forge.lowering.context import ArtifactGuard, LoweringContext
from xe_forge.lowering.triton_analyzer import LaunchRecord, build_context
from xe_forge.lowering.visa.compiler import AbiBuild, VisaCompiler
from xe_forge.lowering.visa.launch import LaunchConfig
from xe_forge.lowering.visa.diagnostics import Evaluation
from xe_forge.lowering.visa.executor import VisaExecutor
from xe_forge.lowering.visa.target import get_target
from xe_forge.lowering.visa.verifier import (
    VerificationCase,
    block_size,
    tolerance_for,
    verification_cases,
)


def spec_lowering_block(spec_path: str | None) -> dict[str, Any]:
    """The optional ``lowering:`` section of a spec (family, shapes, sweep, max_dims)."""
    if not spec_path:
        return {}
    return (yaml.safe_load(Path(spec_path).read_text()) or {}).get("lowering") or {}


@dataclass
class Prepared:
    ctx: LoweringContext
    captured: dict
    abi_build: AbiBuild
    cases: list[VerificationCase]
    tolerance: dict[str, float]
    guard: ArtifactGuard
    record: LaunchRecord
    default_config: LaunchConfig = None
    target_visa: list[str] = field(default_factory=list)  # sealed: guard and screen only
    prepare_s: float = 0.0
    notes: list[str] = field(default_factory=list)


class LoweringSession:
    """Prepare one kernel and evaluate ``.visaasm`` attempts for it."""

    def __init__(
        self,
        module_path: str,
        spec_path: str | None,
        *,
        variant: str | None = None,
        kernel: str | None = None,
        target: str = "xe2",
        workdir: str | Path,
        finalizer_igc: str | None = None,
        timeout_s: float = 180,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        self.module_path = str(Path(module_path).resolve())
        self.spec_path = str(Path(spec_path).resolve()) if spec_path else None
        self.variant = variant
        self.kernel = kernel
        self.target = get_target(target)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.compiler = VisaCompiler(device=self.target.ocloc_device, finalizer_igc=finalizer_igc)
        self.executor = VisaExecutor(self.compiler, self.workdir / "jobs", timeout_s=timeout_s)
        self.rtol = rtol
        self.atol = atol
        self.prepared: Prepared | None = None

    def prepare(self) -> Prepared:
        t0 = time.perf_counter()
        captured = self.executor.capture(self.module_path, self.spec_path, self.variant, self.kernel)
        record = LaunchRecord.from_json(captured["launch"])
        guard = ArtifactGuard()
        # The reference kernel's own compiled forms exist on disk from here on;
        # none may reach a prompt.
        guard.add_tree(captured["triton_cache"])
        sealed = Path(captured["sealed_igc_dump"])
        guard.add_tree(sealed)
        target_visa = [p.read_text(errors="replace") for p in sealed.rglob("*.visaasm")]
        shutil.rmtree(sealed, ignore_errors=True)
        abi_build = self.compiler.discover_abi(Path(captured["stub_spv"]), record.kernel, guard)
        default_config = LaunchConfig(record.num_warps, record.threads_per_warp)

        lowering = spec_lowering_block(self.spec_path)
        semantics_ctx = build_context(record, captured["kernel_source"], self.target.info(), family=lowering.get("family"))
        tolerance = self._tolerance(captured)
        transcendental = semantics_ctx.semantics.has_transcendental
        precision = {}
        for a in record.args:
            if a.kind == "pointer":
                import torch

                from xe_forge.lowering.visa.runner import _TL_TO_TORCH

                dt = getattr(torch, _TL_TO_TORCH.get(a.dtype, "float32"))
                rtol, atol = tolerance_for(dt, transcendental=transcendental, **tolerance)
                precision[a.dtype] = [rtol, atol]
        semantics_ctx.precision = precision
        cases = self._cases(captured, record, lowering)
        self.prepared = Prepared(
            ctx=semantics_ctx,
            captured=captured,
            abi_build=abi_build,
            cases=cases,
            tolerance=tolerance,
            guard=guard,
            record=record,
            default_config=default_config,
            target_visa=target_visa,
            prepare_s=time.perf_counter() - t0,
        )
        if len(captured.get("all_kernels", [])) > 1:
            self.prepared.notes.append(f"module launches {captured['all_kernels']}; lowering {record.kernel}")
        (self.workdir / "contract.yaml").write_text(semantics_ctx.to_prompt())
        return self.prepared

    def _tolerance(self, captured: dict) -> dict[str, float]:
        tol: dict[str, float] = {}
        if self.spec_path:
            from xe_forge.core.spec_loader import load_spec

            spec = load_spec(self.spec_path)
            variant = captured.get("variant")
            if spec.get_rtol(variant) is not None:
                tol["rtol"] = spec.get_rtol(variant)
            if spec.get_atol(variant) is not None:
                tol["atol"] = spec.get_atol(variant)
        if self.rtol is not None:
            tol["rtol"] = self.rtol
        if self.atol is not None:
            tol["atol"] = self.atol
        return tol

    def _cases(self, captured: dict, record: LaunchRecord, lowering: dict) -> list[VerificationCase]:
        dims = captured.get("dims") or {}
        shapes = lowering.get("shapes") or []
        cases = verification_cases(
            dims, block_size(record.constexprs), extra=shapes, sweep=lowering.get("sweep", True)
        )
        limits = lowering.get("max_dims") or {}
        return [c for c in cases if all(c.dims.get(k, 0) <= v for k, v in limits.items())]

    def evaluate(self, visa_text: str, *, measure_perf: bool = False, keep_dir: Path | None = None) -> Evaluation:
        p = self.prepared
        if p is None:
            raise RuntimeError("prepare() first")
        return self.executor.evaluate(
            visa_text,
            p.abi_build,
            p.captured,
            p.cases,
            module_path=self.module_path,
            spec_path=self.spec_path,
            tolerance=p.tolerance,
            transcendental=p.ctx.semantics.has_transcendental,
            measure_perf=measure_perf,
            keep_dir=keep_dir,
        )

    def cleanup(self) -> None:
        if self.prepared is not None:
            shutil.rmtree(self.prepared.abi_build.workdir, ignore_errors=True)

    def provenance(self) -> dict:
        """What the prompts were built from, for the run record."""
        p = self.prepared
        return {
            "module": self.module_path,
            "spec": self.spec_path,
            "kernel": p.record.kernel if p else None,
            "contract_fields": list(LoweringContext.PROMPT_FIELDS),
            "abi_source": "ze_info of an inert ABI stub with the same signature",
            "guarded_artifact_lines": len(p.guard) if p else 0,
            "guarded_sources": len(p.guard.sources) if p else 0,
        }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


@dataclass
class LoweringResult:
    run_dir: Path
    record: Any
    visa_path: Path | None = None

    @property
    def success(self) -> bool:
        return bool(self.record.correct)


class LoweringPipeline:
    """One lowering run, with a Claude Code session as the lowering agent.

    prepare -> retrieve knowledge -> guard-check the prompt material -> write the
    sealed workspace -> run the session (it calls ``visa-verify``, which records every
    attempt) -> guard-check the transcript -> collect the record and the best kernel.
    """

    def __init__(
        self,
        module_path: str,
        spec_path: str | None,
        *,
        variant: str | None = None,
        kernel: str | None = None,
        target: str = "xe2",
        knowledge: str = "docs+examples",
        max_attempts: int = 12,
        optimize: bool = False,
        char_budget: int = 60000,
        finalizer_igc: str | None = None,
        timeout_s: float = 180,
        session_timeout_s: float = 4 * 3600,
        max_turns: int = 80,
        out_dir: str | Path | None = None,
        llm: Any = None,
        knowledge_root: str | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        stem = Path(module_path).stem
        self.run_id = f"{stem}-{knowledge.replace('+', '_')}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.run_dir = (Path(out_dir or "lowering_runs") / self.run_id).resolve()
        self.session = LoweringSession(
            module_path, spec_path, variant=variant, kernel=kernel, target=target,
            workdir=self.run_dir / "work", finalizer_igc=finalizer_igc, timeout_s=timeout_s, rtol=rtol, atol=atol,
        )
        self.target = target
        self.knowledge = knowledge
        self.max_attempts = max_attempts
        self.optimize = optimize
        self.char_budget = char_budget
        self.session_timeout_s = session_timeout_s
        self.max_turns = max_turns
        self.llm = llm
        self.knowledge_root = knowledge_root

    def run(self) -> LoweringResult:
        from xe_forge.core.trial_manager import TrialManager
        from xe_forge.lowering.context import LeakDetected
        from xe_forge.lowering.metrics import LoweringRunRecord, append_jsonl
        from xe_forge.lowering.visa import agent
        from xe_forge.lowering.visa.knowledge import load_knowledge
        from xe_forge.lowering.visa.retrieval import MODE_LETTER, VisaRetriever
        from xe_forge.lowering.visa.verify_cli import summarize_attempts

        self.run_dir.mkdir(parents=True, exist_ok=True)
        prepared = self.session.prepare()
        ctx = prepared.ctx
        model = getattr(self.llm, "model", "") or ""
        record = LoweringRunRecord(
            run_id=self.run_id, kernel=ctx.kernel_name, family=ctx.family, target=self.target,
            mode=self.knowledge, mode_letter=MODE_LETTER[self.knowledge], model=model,
        )
        result = LoweringResult(self.run_dir, record)
        try:
            kb = load_knowledge(self.knowledge_root)
            retrieved = VisaRetriever(kb, char_budget=self.char_budget).retrieve(
                ctx.semantics, self.knowledge, family=ctx.family, target_visa=prepared.target_visa
            )
            # Knowledge-base text has controlled provenance (corpus entries are screened
            # against the target above); the guard looks for target-specific text.
            for item in [*kb.examples, *kb.corpus]:
                prepared.guard.allow(item.visa)
            for doc in kb.docs:
                prepared.guard.allow(doc.body)
            contract = ctx.to_prompt()
            prepared.guard.check(contract)
            prepared.guard.check(retrieved.text)
            record.contract_chars, record.knowledge_chars = len(contract), retrieved.chars
            record.doc_ids, record.example_ids, record.corpus_ids = retrieved.doc_ids, retrieved.example_ids, retrieved.corpus_ids
            record.held_out, record.screened_out = retrieved.held_out, retrieved.screened_out

            tree = f"{ctx.kernel_name}__visa"
            TrialManager(self.run_dir / "trials").init(tree, self.session.module_path, triton_baseline=True)
            state_path = self.run_dir / "work" / "state.json"
            save_state(self.session, state_path, run_dir=str(self.run_dir), run_id=self.run_id,
                       max_attempts=self.max_attempts, optimize=self.optimize, trial_tree=tree)
            workspace = self.run_dir / "workspace"
            agent.write_workspace(workspace, contract=contract, knowledge=retrieved.text, state_path=state_path,
                                  budget=self.max_attempts, optimize=self.optimize)
            write_json(self.run_dir / "provenance.json", {
                **self.session.provenance(),
                "agent": "claude-code (headless, --restricted; tools: workspace files + ./visa-verify)",
                "knowledge_mode": self.knowledge,
                "doc_ids": retrieved.doc_ids, "example_ids": retrieved.example_ids, "corpus_ids": retrieved.corpus_ids,
                "held_out": retrieved.held_out, "screened_out": retrieved.screened_out,
            })

            env = agent.claude_env(getattr(self.llm, "api_base", None), getattr(self.llm, "api_key", None), model)
            t0 = time.time()
            session = agent.launch(workspace, env, max_turns=self.max_turns, timeout_s=self.session_timeout_s)
            record.tokens = session.usage
            record.cost_usd, record.num_turns = session.cost_usd, session.num_turns
            if session.is_error:
                record.error = f"session: {session.error}"

            # The transcript holds everything the model saw; it must be free of artefact text too.
            transcript = session.log_path.read_text(errors="replace") if session.log_path.exists() else ""
            try:
                prepared.guard.check(transcript.replace("\\n", "\n"))
            except LeakDetected as e:
                record.error = f"leak_detected in transcript: {e}"

            attempts = summarize_attempts(self.run_dir)
            record.attempts = len(attempts)
            best = None
            for i, ev in enumerate(attempts):
                cat = ev["category"]
                record.categories[cat] = record.categories.get(cat, 0) + 1
                if cat.startswith("correct"):
                    if record.attempts_to_valid is None:
                        record.attempts_to_valid = i + 1
                        record.time_to_valid_s = round(Path(ev["dir"]).stat().st_mtime - t0, 1)
                    perf = ev.get("performance") or {}
                    if best is None or (perf.get("candidate_us") or 1e30) < (
                            (best.get("performance") or {}).get("candidate_us") or 1e30):
                        best = ev
            record.stopped = "correct" if best else ("budget" if record.attempts >= self.max_attempts else "session_end")
            if best is not None:
                record.correct = True
                c = best["compile"]
                record.grf, record.spills, record.binary_size = c.get("grf_count"), c.get("spill_size"), c.get("binary_size")
                if best.get("performance"):
                    record.best_speedup = best["performance"].get("speedup")
                result.visa_path = self.run_dir / f"{ctx.kernel_name}.visaasm"
                shutil.copy(Path(best["dir"]) / "kernel.visaasm", result.visa_path)
            trials = TrialManager(self.run_dir / "trials")
            (self.run_dir / "trials_status.txt").write_text(trials.get_status(tree))
        except LeakDetected as e:
            record.error = f"leak_detected: {e}"
        except Exception as e:
            record.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            record.finished = time.time()
            append_jsonl(self.run_dir / "run.jsonl", record)
            self.session.cleanup()
        return result


# -- persisted session state (read by the visa-verify command) ------------------------


def save_state(session: LoweringSession, path: Path, **extra: Any) -> None:
    """Everything an out-of-process verifier needs to evaluate attempts for this kernel."""
    from dataclasses import asdict

    p = session.prepared
    write_json(path, {
        "module_path": session.module_path,
        "spec_path": session.spec_path,
        "target": session.target.name,
        "finalizer_igc": session.compiler.finalizer_igc,
        "timeout_s": session.executor.timeout_s,
        "captured": p.captured,
        "abi_build": abi_build_to_json(p.abi_build),
        "default_config": p.default_config.to_json(),
        "cases": [asdict(c) for c in p.cases],
        "tolerance": p.tolerance,
        "transcendental": p.ctx.semantics.has_transcendental,
        "guard_lines": sorted(p.guard._lines),
        "guard_allowed": sorted(p.guard._allowed),
        **extra,
    })


def abi_build_to_json(b: AbiBuild) -> dict:
    return {"spv_path": str(b.spv_path), "override_key": b.override_key, "zeinfo": b.zeinfo,
            "workdir": str(b.workdir), "stub_header": b.stub_header}


def abi_build_from_json(d: dict) -> AbiBuild:
    return AbiBuild(Path(d["spv_path"]), d["override_key"], d["zeinfo"], Path(d["workdir"]), d.get("stub_header", []))


def load_state(path: Path) -> tuple[dict, AbiBuild, list[VerificationCase], ArtifactGuard, VisaExecutor]:
    st = json.loads(Path(path).read_text())
    abi_build = abi_build_from_json(st["abi_build"])
    cases = [VerificationCase(**c) for c in st["cases"]]
    guard = ArtifactGuard()
    guard._lines = set(st["guard_lines"])
    guard._allowed = set(st["guard_allowed"])
    target = get_target(st["target"])
    compiler = VisaCompiler(device=target.ocloc_device, finalizer_igc=st["finalizer_igc"])
    executor = VisaExecutor(compiler, Path(st["run_dir"]) / "work" / "verify_jobs", timeout_s=st["timeout_s"])
    return st, abi_build, cases, guard, executor
