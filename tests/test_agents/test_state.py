"""Tests for L2LAgentState."""

import pytest

from l2l_bench.agents.state import L2LAgentState


class TestL2LAgentState:
    """Tests for the L2LAgentState dataclass."""

    def test_default_initialization(self):
        """Test that state initializes with correct defaults."""
        state = L2LAgentState()

        assert state.messages == []
        assert state.tools == []
        assert state.current_step == 0
        assert state.experiments_completed == 0
        assert state.budget == 10
        assert state.learned_knowledge_draft == ""
        assert state.pending_hypothesis is None
        assert state.hypothesis_history == []
        assert state.interpretation_history == []

    def test_custom_initialization(self):
        """Test state with custom values."""
        state = L2LAgentState(
            budget=5,
            current_step=2,
            experiments_completed=1,
            learned_knowledge_draft="some knowledge",
        )

        assert state.budget == 5
        assert state.current_step == 2
        assert state.experiments_completed == 1
        assert state.learned_knowledge_draft == "some knowledge"

    def test_budget_remaining(self):
        """Test budget_remaining property."""
        state = L2LAgentState(budget=10, experiments_completed=3)
        assert state.budget_remaining == 7

        state = L2LAgentState(budget=5, experiments_completed=5)
        assert state.budget_remaining == 0

    def test_is_budget_exhausted(self):
        """Test is_budget_exhausted property."""
        state = L2LAgentState(budget=10, experiments_completed=9)
        assert not state.is_budget_exhausted

        state = L2LAgentState(budget=10, experiments_completed=10)
        assert state.is_budget_exhausted

        state = L2LAgentState(budget=10, experiments_completed=11)
        assert state.is_budget_exhausted

    def test_copy_basic(self):
        """Test copy() creates independent copy."""
        state = L2LAgentState(
            current_step=1,
            experiments_completed=1,
            learned_knowledge_draft="original",
        )

        copied = state.copy()

        # values should match
        assert copied.current_step == 1
        assert copied.experiments_completed == 1
        assert copied.learned_knowledge_draft == "original"

        # modifying copy shouldn't affect original
        copied.current_step = 5
        assert state.current_step == 1

    def test_copy_with_updates(self):
        """Test copy() with field updates."""
        state = L2LAgentState(
            current_step=1,
            budget=10,
        )

        copied = state.copy(current_step=5, budget=20)

        assert copied.current_step == 5
        assert copied.budget == 20
        # original unchanged
        assert state.current_step == 1
        assert state.budget == 10

    def test_copy_lists_are_independent(self):
        """Test that copied lists are independent."""
        state = L2LAgentState()
        state.hypothesis_history.append({"step": 1})

        copied = state.copy()
        copied.hypothesis_history.append({"step": 2})

        assert len(state.hypothesis_history) == 1
        assert len(copied.hypothesis_history) == 2
