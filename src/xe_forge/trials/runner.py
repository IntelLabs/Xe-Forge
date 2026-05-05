"""Trial-tree orchestrator.

Drives the loop: build t0 (from Triton input or GeneratorAgent), then
repeatedly propose -> read parent -> write child -> validate -> save ->
execute via KernelBenchExecutor -> record. On plateau (if VTune enabled)
runs xpu_profiler.profile and caches the report for the next writer call.

All GPU workloads go through the executor in-process so the single-XPU
constraint is enforced naturally.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from xe_forge.core.validate_triton import validate_triton_kernel
from xe_forge.trials import store
from xe_forge.trials.context import TrialContext

logger = logging.getLogger(__name__)


@dataclass
class TrialRunResult:
    """Outcome of a trial-tree run."""

    success: bool
    kernel_name: str
    best_trial_id: str | None
    best_speedup: float | None
    best_code: str | None
    baseline_us: float | None
    output_path: str | None
    trials_count: int


class TrialRunner:
    """Tree-search loop over a single kernel.

    Dependencies are injected at construction so the runner doesn't need to
    know about dspy or any specific agent class.
    """

    def __init__(
        self,
        ctx: TrialContext,
        search,                      # TrialSearch instance
        writer,                      # TrialWriter instance
        executor,                    # KernelBenchExecutor
        generator=None,              # GeneratorAgent | None — required for PyTorch input
    ):
        self.ctx = ctx
        self.search = search
        self.writer = writer
        self.executor = executor
        self.generator = generator

    # ------------------------------------------------------------------ public

    def run(
        self,
        *,
        triton_code: str | None = None,
        pytorch_code: str | None = None,
        pytorch_path: str | None = None,
        output_path: str | None = None,
    ) -> TrialRunResult:
        """Run the full trial loop. Exactly one of triton_code/pytorch_code is required.

        Returns a ``TrialRunResult`` summarizing the outcome.
        """
        kernel_name = self.ctx.kernel_name
        if triton_code and pytorch_code:
            raise ValueError("Pass either triton_code or pytorch_code, not both.")
        if not triton_code and not pytorch_code:
            raise ValueError("Pass triton_code or pytorch_code.")

        # 1. Build t0.
        t0_code, t0_strategy, baseline_type = self._build_t0(
            triton_code=triton_code,
            pytorch_code=pytorch_code,
            pytorch_path=pytorch_path,
        )
        store.init_state(
            kernel_name,
            baseline_file=pytorch_path or "<in-memory>",
            triton_baseline=(baseline_type == "triton"),
        )
        t0_path = self._write_to_disk(kernel_name, "t0.py", t0_code)
        store.save_trial(kernel_name, t0_path, parent=None, strategy=t0_strategy)

        # 2. Benchmark t0 — becomes the cached baseline for later trials.
        baseline_us, t0_speedup, t0_correct = self._benchmark_against_self(t0_code)
        store.record_result(
            kernel_name,
            "t0",
            validation="pass",
            correctness="pass" if t0_correct else "fail",
            speedup=t0_speedup,
            baseline_us=baseline_us,
            triton_us=baseline_us,
        )

        # 3. Trial loop.
        while True:
            state = store.get_state(kernel_name)
            if self.search.should_stop(state, self.ctx):
                break

            proposal = self.search.propose(state, self.ctx)
            logger.info(
                "[runner] proposing: parent=%s hint=%s strategy=%s",
                proposal.parent_id, proposal.hint_tag, proposal.strategy,
            )

            # Optional profiling on plateau (only triggers if enabled).
            vtune_report = self._maybe_profile(state)

            parent_code = store.read_trial(kernel_name, proposal.parent_id)

            # Writer + validation retry loop (doesn't count as a trial).
            new_code = self._write_and_validate(parent_code, proposal, vtune_report)
            if new_code is None:
                logger.warning(
                    "[runner] writer produced no valid code after retries; "
                    "skipping this slot"
                )
                # Record a failed placeholder so search can observe the failure
                # and move on. Using a save+fail avoids infinite looping.
                fail_path = self._write_to_disk(
                    kernel_name,
                    f"t{state['next_id']}.py",
                    parent_code,
                )
                trial_id = store.save_trial(
                    kernel_name, fail_path,
                    parent=proposal.parent_id,
                    strategy=f"[failed] {proposal.strategy}",
                )
                store.record_result(
                    kernel_name, trial_id,
                    validation="fail", correctness=None,
                )
                continue

            # Save + benchmark + record.
            next_state = store.get_state(kernel_name)
            trial_filename = f"t{next_state['next_id']}.py"
            trial_path = self._write_to_disk(kernel_name, trial_filename, new_code)
            trial_id = store.save_trial(
                kernel_name, trial_path,
                parent=proposal.parent_id,
                strategy=proposal.strategy,
            )

            speedup, correct, triton_us = self._benchmark_vs_parent(
                parent_code=parent_code,
                new_code=new_code,
            )
            store.record_result(
                kernel_name, trial_id,
                validation="pass",
                correctness="pass" if correct else "fail",
                speedup=(speedup if correct else None),
                baseline_us=baseline_us,
                triton_us=triton_us if triton_us is not None else None,
            )

        # 4. Finalize + one last definitive benchmark.
        final_state = store.get_state(kernel_name)
        best = store.best_trial(kernel_name)
        if best is None:
            logger.warning("[runner] no correct trial produced; nothing to finalize")
            return TrialRunResult(
                success=False,
                kernel_name=kernel_name,
                best_trial_id=None,
                best_speedup=None,
                best_code=None,
                baseline_us=baseline_us,
                output_path=None,
                trials_count=len(final_state.get("trials", {})),
            )

        out_path = output_path or f"{kernel_name}_triton.py"
        final_path = store.finalize(kernel_name, out_path)

        return TrialRunResult(
            success=True,
            kernel_name=kernel_name,
            best_trial_id=best["trial_id"],
            best_speedup=best.get("speedup"),
            best_code=store.read_trial(kernel_name, best["trial_id"]),
            baseline_us=baseline_us,
            output_path=final_path,
            trials_count=len(final_state.get("trials", {})),
        )

    # ----------------------------------------------------------------- helpers

    def _build_t0(
        self,
        *,
        triton_code: str | None,
        pytorch_code: str | None,
        pytorch_path: str | None,
    ) -> tuple[str, str, str]:
        """Return (code, strategy, baseline_type)."""
        if triton_code is not None:
            errors = self._validate(triton_code)
            critical = [e for e in errors if e.level == "ERROR"]
            if critical:
                raise RuntimeError(
                    f"Input Triton kernel failed validation:\n"
                    + "\n".join(str(e) for e in critical)
                )
            return triton_code, "seed from input Triton kernel", "triton"

        if self.generator is None:
            raise RuntimeError(
                "PyTorch input requires a GeneratorAgent. Pass generator=... "
                "when constructing TrialRunner."
            )
        spec_summary = self._spec_summary()
        code, strategy, static = self.generator.generate(
            pytorch_code=pytorch_code or "",
            pytorch_path=pytorch_path,
            spec_summary=spec_summary,
        )
        self.ctx.static_analysis = static or {}
        return code, strategy or "initial kernel from PyTorch", "pytorch"

    def _write_and_validate(self, parent_code, proposal, vtune_report):
        """Call writer + validate; retry up to max_iterations on validation error."""
        max_iters = max(1, self.ctx.config.agent.max_iterations)
        for attempt in range(1, max_iters + 1):
            code = self.writer.write(
                parent_code, proposal, self.ctx, vtune_report=vtune_report
            )
            if not code or code == parent_code:
                logger.info("[runner] writer returned unchanged/empty on attempt %d", attempt)
                return None
            errors = self._validate(code)
            critical = [e for e in errors if e.level == "ERROR"]
            if not critical:
                return code
            logger.info(
                "[runner] %d validation error(s) on writer attempt %d",
                len(critical), attempt,
            )
            # On the next iteration we'd ideally feed errors back into the
            # writer, but each writer's error-feedback interface varies.
            # Keeping the retry loop simple for now — the writer's own
            # internal retry (CoVeR / generator) handles most cases.
        return None

    def _maybe_profile(self, state) -> str:
        """Run VTune on the current best trial if enabled and plateau detected."""
        if not self.ctx.config.profiler.vtune_enabled:
            return ""
        if not self.search.is_plateau(state, self.ctx):
            return ""
        best_id = state.get("best_trial")
        if best_id is None:
            return ""
        if best_id in self.ctx.profiler_cache:
            return self.ctx.profiler_cache[best_id]
        best_path = store.trial_path(self.ctx.kernel_name, best_id)
        logger.info("[runner] plateau detected — profiling %s", best_id)
        try:
            from xe_forge.core.xpu_profiler import profile

            report = profile(Path(best_path))
        except Exception as e:
            logger.warning("[runner] profiling failed: %s", e)
            report = ""
        self.ctx.profiler_cache[best_id] = report
        return report

    def _validate(self, code: str):
        """Write code to a temp file and run validate_triton_kernel."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            path = Path(f.name)
        try:
            return validate_triton_kernel(path)
        finally:
            try:
                path.unlink()
            except Exception:
                pass

    def _write_to_disk(self, kernel_name: str, filename: str, code: str) -> str:
        trial_dir = store.trial_dir(kernel_name)
        os.makedirs(trial_dir, exist_ok=True)
        path = os.path.join(trial_dir, filename)
        with open(path, "w") as f:
            f.write(code)
        return path

    def _benchmark_against_self(
        self, code: str
    ) -> tuple[float | None, float | None, bool]:
        """Run t0 once to get baseline_us + self-correctness.

        Speedup of t0 vs. itself is 1.0. ``baseline_us`` is used as the
        cached value for later trials.
        """
        if self.executor is None or not self.ctx.input_shapes:
            return None, None, True
        result = self.executor.execute(
            code,
            kernel_name=self.ctx.kernel_name,
            input_shapes=self.ctx.input_shapes,
            flop=self.ctx.flop,
            dtype=self.ctx.dtype,
            init_args=self.ctx.init_args,
        )
        if not result.success:
            logger.warning("[runner] t0 benchmark failed: %s", result.error_message)
            return None, None, False
        us = result.execution_time_ms * 1000 if result.execution_time_ms else None
        return us, 1.0, True

    def _benchmark_vs_parent(
        self, *, parent_code: str, new_code: str
    ) -> tuple[float | None, bool, float | None]:
        """Speedup of new_code over parent_code, plus correctness.

        Returns ``(speedup, correct, triton_us_for_new_code)``.
        """
        if self.executor is None or not self.ctx.input_shapes:
            return None, True, None

        comp = self.executor.compare_kernels(
            original_code=parent_code,
            optimized_code=new_code,
            kernel_name=self.ctx.kernel_name,
            input_shapes=self.ctx.input_shapes,
            flop=self.ctx.flop,
            dtype=self.ctx.dtype,
            init_args=self.ctx.init_args,
        )
        correct = (
            (comp.original_correct is not False)
            and (comp.optimized_correct is not False)
            and comp.speedup > 0
        )
        triton_us = comp.optimized_time_us if comp.optimized_time_us != float("inf") else None

        # Convert parent-relative speedup to baseline-relative using the
        # cached baseline_us + parent's recorded triton_us. If we don't have
        # those (e.g. parent is t0), speedup vs. parent == speedup vs. baseline.
        state = store.get_state(self.ctx.kernel_name)
        baseline_us = None
        if state.get("baseline_us"):
            baseline_us = state["baseline_us"][0]
        speedup = None
        if correct and baseline_us and triton_us and triton_us > 0:
            speedup = baseline_us / triton_us
        elif correct:
            speedup = comp.speedup
        return speedup, correct, triton_us

    def _spec_summary(self) -> str:
        parts = []
        if self.ctx.input_shapes:
            parts.append(f"Input shapes: {self.ctx.input_shapes}")
        if self.ctx.dtype is not None:
            parts.append(f"dtype: {self.ctx.dtype}")
        if self.ctx.flop is not None:
            parts.append(f"FLOP: {self.ctx.flop:,.0f}")
        if self.ctx.init_args:
            parts.append(f"Model init args: {self.ctx.init_args}")
        return "\n".join(parts) or "(no spec summary)"
