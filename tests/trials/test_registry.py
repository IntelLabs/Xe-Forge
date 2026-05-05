"""Sanity tests for the trial-tree registries.

Kept GPU-free and LLM-free so it runs in plain CI. Confirms:
- search strategies resolve by name,
- writer classes resolve by name,
- zero-dep writers instantiate,
- TrialContext constructs from a mocked config without hitting dspy.
"""

from __future__ import annotations

import pytest

from xe_forge.config import Config, ProfilerConfig, TrialConfig
from xe_forge.trials import TrialContext
from xe_forge.trials.search import get_search, list_searches
from xe_forge.trials.search.base import TrialProposal, TrialSearch
from xe_forge.trials.writers import get_writer_cls, list_writers


# ---------------------------------------------------------------- search


def test_all_builtin_searches_registered():
    names = set(list_searches())
    assert {"tree_walk", "best_first", "beam", "mcts"}.issubset(names)


@pytest.mark.parametrize("name", ["tree_walk", "best_first"])
def test_search_instantiates(name):
    instance = get_search(name)
    assert isinstance(instance, TrialSearch)


def test_unknown_search_raises():
    with pytest.raises(KeyError):
        get_search("does_not_exist")


def test_tree_walk_first_proposal_on_t0_only_state():
    search = get_search("tree_walk")
    state = {
        "trials": {
            "t0": {
                "parent": None,
                "file": "t0.py",
                "strategy": "initial",
                "validation": "pass",
                "correctness": "pass",
                "speedup": 1.0,
                "baseline_us": 100.0,
                "triton_us": 100.0,
                "status": "completed",
            }
        },
        "best_trial": "t0",
        "next_id": 1,
        "baseline_us": [100.0],
    }
    ctx = _fake_ctx()
    proposal = search.propose(state, ctx)
    assert isinstance(proposal, TrialProposal)
    assert proposal.parent_id == "t0"
    assert proposal.hint_tag == "first_try"


def test_tree_walk_backtracks_on_regression():
    search = get_search("tree_walk")
    state = {
        "trials": {
            "t0": {"parent": None, "speedup": 1.0, "correctness": "pass",
                   "validation": "pass", "status": "completed",
                   "file": "t0.py", "baseline_us": 100.0, "triton_us": 100.0,
                   "strategy": ""},
            "t1": {"parent": "t0", "speedup": 2.5, "correctness": "pass",
                   "validation": "pass", "status": "completed",
                   "file": "t1.py", "baseline_us": 100.0, "triton_us": 40.0,
                   "strategy": "first pass"},
            "t2": {"parent": "t1", "speedup": 1.5, "correctness": "pass",
                   "validation": "pass", "status": "completed",
                   "file": "t2.py", "baseline_us": 100.0, "triton_us": 66.0,
                   "strategy": "second pass (regressed)"},
        },
        "best_trial": "t1",
        "next_id": 3,
        "baseline_us": [100.0],
    }
    proposal = search.propose(state, _fake_ctx())
    assert proposal.hint_tag == "branch_back"
    assert proposal.parent_id == "t1"  # backtrack to the best


def test_tree_walk_should_stop_on_max_trials():
    search = get_search("tree_walk")
    ctx = _fake_ctx(max_trials=3)
    state = {"trials": {f"t{i}": {} for i in range(3)}, "best_trial": None}
    assert search.should_stop(state, ctx)


# ---------------------------------------------------------------- writers


def test_all_builtin_writers_registered():
    names = set(list_writers())
    assert "stage_sequence" in names


def test_writer_cls_resolves():
    from xe_forge.trials.writers.stage_sequence import StageSequenceWriter

    assert get_writer_cls("stage_sequence") is StageSequenceWriter


def test_unknown_writer_raises():
    with pytest.raises(KeyError):
        get_writer_cls("does_not_exist")


# ---------------------------------------------------------------- context


def test_trial_context_constructs():
    cfg = Config(trial=TrialConfig(max_trials=5), profiler=ProfilerConfig())
    ctx = TrialContext(
        kernel_name="demo",
        config=cfg,
        spec=None,
        variant_type="bench-gpu",
        input_shapes={"X": (128, 128)},
        flop=2 * 128 * 128,
        dtype=None,
        init_args=None,
        xpu_config={"device": "xpu"},
    )
    assert ctx.kernel_name == "demo"
    assert ctx.config.trial.max_trials == 5
    assert ctx.profiler_cache == {}


# ---------------------------------------------------------------- helpers


def _fake_ctx(max_trials: int = 10) -> TrialContext:
    cfg = Config(
        trial=TrialConfig(max_trials=max_trials, plateau_window=2),
        profiler=ProfilerConfig(),
    )
    return TrialContext(
        kernel_name="demo",
        config=cfg,
        spec=None,
        variant_type="bench-gpu",
        input_shapes=None,
        flop=None,
        dtype=None,
        init_args=None,
        xpu_config=None,
    )
