"""
The proposal layer (plan §13.7, §6).

Parsing is tested hard because it is the boundary with a model's free-form output: a
parser that quietly returns an empty list on a formatting variation turns "the agent had
three ideas" into "the agent had none", and the run reports no candidates for a reason
that has nothing to do with the kernel.
"""

from __future__ import annotations

from xe_forge.orbit.optimize.loop import Proposal
from xe_forge.orbit.optimize.proposer import (
    ClaudeProposer,
    ProposerConfig,
    _parse_proposals,
    _plan_prompt,
    _task_markdown,
)

VALID = '[{"title": "Raise BLOCK_SIZE to 2048", "rationale": "fewer blocks", "parameters": {"BLOCK_SIZE": 2048}}]'


class TestParsingProposals:
    def test_a_bare_json_array_parses(self):
        proposals = _parse_proposals(VALID)
        assert len(proposals) == 1
        assert proposals[0].title == "Raise BLOCK_SIZE to 2048"
        assert proposals[0].parameters == {"BLOCK_SIZE": 2048}

    def test_a_fenced_block_parses(self):
        assert len(_parse_proposals(f"Here you go:\n```json\n{VALID}\n```\nHope that helps.")) == 1

    def test_prose_around_the_array_is_tolerated(self):
        assert (
            len(_parse_proposals(f"I considered several options.\n\n{VALID}\n\nRanked best first."))
            == 1
        )

    def test_several_proposals_keep_their_order(self):
        text = (
            '[{"title": "first", "rationale": "a"},'
            ' {"title": "second", "rationale": "b"},'
            ' {"title": "third", "rationale": "c"}]'
        )
        assert [p.title for p in _parse_proposals(text)] == ["first", "second", "third"]

    def test_an_entry_without_a_title_is_dropped(self):
        """A proposal with no title cannot be trialled or reported."""
        text = '[{"rationale": "no title"}, {"title": "real", "rationale": "b"}]'
        assert [p.title for p in _parse_proposals(text)] == ["real"]

    def test_malformed_json_yields_nothing_rather_than_raising(self):
        assert _parse_proposals("[{title: unquoted}]") == []

    def test_an_empty_response_yields_nothing(self):
        assert _parse_proposals("") == []
        assert _parse_proposals("   \n  ") == []

    def test_a_non_list_payload_is_rejected(self):
        assert _parse_proposals('{"title": "not a list"}') == []

    def test_non_dict_entries_are_skipped(self):
        assert [p.title for p in _parse_proposals('["a string", {"title": "ok"}]')] == ["ok"]

    def test_bad_parameters_do_not_discard_the_proposal(self):
        """A malformed parameter block should cost the parameters, not the idea."""
        proposals = _parse_proposals('[{"title": "keep me", "parameters": "not a dict"}]')
        assert len(proposals) == 1
        assert proposals[0].parameters == {}


class TestPlanPrompt:
    def test_the_measured_context_reaches_the_model(self):
        """Without it the model optimizes a kernel in the abstract (§9.5)."""
        prompt = _plan_prompt("SOURCE", "ceiling is 0.22%", 3, "gumbel_sample")
        assert "ceiling is 0.22%" in prompt
        assert "gumbel_sample" in prompt

    def test_missing_context_is_stated_rather_than_left_blank(self):
        assert "(none available)" in _plan_prompt("SOURCE", "", 3, "k")

    def test_the_requested_count_is_explicit(self):
        assert "exactly 5" in _plan_prompt("SOURCE", "", 5, "k")

    def test_numerical_changes_are_ruled_out_in_the_prompt(self):
        """Correctness is gated separately; a different-numerics kernel is a new kernel."""
        assert "changes numerical results" in _plan_prompt("S", "", 3, "k")

    def test_a_huge_source_is_truncated_rather_than_sent_whole(self):
        prompt = _plan_prompt("x" * 50000, "", 3, "k")
        assert len(prompt) < 30000


class TestWorkspaceTask:
    def test_the_task_says_the_copy_is_safe_to_edit(self):
        task = _task_markdown(Proposal("t", "r"), "kernel.py", None, "python check.py")
        assert "copy" in task
        assert "python check.py" in task

    def test_the_exit_codes_are_explained(self):
        """The agent must know that 2 is 'could not check', not 'wrong'."""
        task = _task_markdown(Proposal("t", "r"), "k.py", None, "cmd")
        assert "Exit 0" in task and "2 means" in task

    def test_the_agent_is_told_its_own_run_is_advisory(self):
        """The rule the loop depends on: the agent does not grade its own homework."""
        task = _task_markdown(Proposal("t", "r"), "k.py", None, "cmd")
        assert "advisory" in task


class TestProviderFailures:
    def test_an_unreachable_binary_yields_no_proposals_rather_than_raising(self):
        proposer = ClaudeProposer(ProposerConfig(binary="/nonexistent/claude"))
        assert proposer.plan("source", "knowledge") == []

    def test_availability_is_reported_honestly(self):
        assert not ClaudeProposer(ProposerConfig(binary="/nonexistent/claude")).available()

    def test_implement_returns_none_when_nothing_changed(self, tmp_path):
        """An agent that edited nothing produced no candidate, which is not an error."""
        target = tmp_path / "kernel.py"
        target.write_bytes(b"X = 1\n")
        proposer = ClaudeProposer(ProposerConfig(binary="/nonexistent/claude"))
        assert proposer.implement(Proposal("t", "r"), target, tmp_path / "ws") is None

    def test_implement_never_edits_the_live_target(self, tmp_path):
        """The agent works on a copy; the installed tree stays untouched until Orbit acts."""
        target = tmp_path / "kernel.py"
        target.write_bytes(b"ORIGINAL\n")
        proposer = ClaudeProposer(ProposerConfig(binary="/nonexistent/claude"))
        proposer.implement(Proposal("t", "r"), target, tmp_path / "ws")
        assert target.read_bytes() == b"ORIGINAL\n"

    def test_the_workspace_carries_the_harness_when_given_one(self, tmp_path):
        target = tmp_path / "kernel.py"
        target.write_bytes(b"X = 1\n")
        harness = tmp_path / "check.py"
        harness.write_text("print('ok')")
        ws = tmp_path / "ws"
        ClaudeProposer(ProposerConfig(binary="/nonexistent/claude")).implement(
            Proposal("t", "r"), target, ws, harness=harness
        )
        assert (ws / "check.py").is_file()
        assert (ws / "TASK.md").is_file()


class TestImplementationPermissions:
    """A print-mode session with no permission grant silently refuses every edit.

    Found live: two good proposals, two workspaces created correctly, and "no edit
    produced" for both — because each Edit call hit a prompt with nobody to answer it.
    """

    def test_implementation_sessions_are_allowed_to_edit(self):
        from xe_forge.orbit.optimize.proposer import IMPLEMENT_PERMISSION_MODE, ProposerConfig

        assert ProposerConfig().permission_mode == IMPLEMENT_PERMISSION_MODE

    def test_the_grant_can_be_withheld(self):
        """Read-only sessions are how the 'proposal carries no source' path is tested."""
        from xe_forge.orbit.optimize.proposer import ProposerConfig

        assert ProposerConfig(permission_mode="").permission_mode == ""

    def test_the_planning_call_gets_no_edit_permission(self):
        """PLAN only reads and answers; it has no business editing anything."""
        import inspect

        from xe_forge.orbit.optimize.proposer import ClaudeProposer

        source = inspect.getsource(ClaudeProposer.plan)
        assert "permission_mode" not in source
