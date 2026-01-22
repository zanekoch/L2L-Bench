"""Tests for L2LAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from l2l_bench.agents.agent import L2LAgent
from l2l_bench.agents.config import GPT4O_MINI, LLMConfig
from l2l_bench.agents.state import L2LAgentState


@pytest.fixture
def mock_tool():
    """Create a mock Tool object."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_tools(mock_tool):
    """Create a list of mock tools."""
    return [mock_tool]


@pytest.fixture
def mock_message():
    """Create a mock Message object."""
    msg = MagicMock()
    msg.role = "user"
    msg.content = "test message"
    return msg


class TestL2LAgentInit:
    """Tests for L2LAgent initialization."""

    def test_default_config(self):
        """Test agent initializes with default config."""
        agent = L2LAgent()

        assert agent.llm_config.model == "anthropic/claude-3-5-sonnet-20241022"
        assert agent.budget == 10
        assert "{budget}" not in agent.system_prompt  # should be formatted

    def test_custom_config(self):
        """Test agent with custom LLM config."""
        agent = L2LAgent(llm_config=GPT4O_MINI, budget=5)

        assert agent.llm_config.model == "openai/gpt-4o-mini"
        assert agent.budget == 5

    def test_custom_system_prompt(self):
        """Test agent with custom system prompt."""
        custom_prompt = "Custom prompt with {budget} experiments"
        agent = L2LAgent(system_prompt=custom_prompt, budget=7)

        assert "7" in agent.system_prompt
        assert "{budget}" not in agent.system_prompt


class TestL2LAgentInitState:
    """Tests for L2LAgent.init_state()."""

    @pytest.mark.asyncio
    async def test_init_state_empty_tools(self):
        """Test init_state with empty tools list."""
        agent = L2LAgent()
        state = await agent.init_state([])

        assert isinstance(state, L2LAgentState)
        assert state.messages == []
        assert state.tools == []
        assert state.budget == 10

    @pytest.mark.asyncio
    async def test_init_state_with_tools(self, mock_tools):
        """Test init_state stores tools correctly."""
        agent = L2LAgent(budget=5)
        state = await agent.init_state(mock_tools)

        assert state.tools == mock_tools
        assert state.budget == 5


class TestL2LAgentGetAsv:
    """Tests for L2LAgent.get_asv()."""

    @pytest.mark.asyncio
    async def test_get_asv_appends_messages(self, mock_tools, mock_message):
        """Test that get_asv appends observations to message history."""
        agent = L2LAgent()

        # mock the LLMCallOp
        mock_action = MagicMock()
        mock_action.tool_calls = None
        agent.llm_call_op = AsyncMock(return_value=mock_action)

        state = await agent.init_state(mock_tools)
        action, new_state, value = await agent.get_asv(state, [mock_message])

        # action is returned
        assert action == mock_action

        # message history is updated
        assert len(new_state.messages) == 2  # observation + action
        assert new_state.messages[0] == mock_message

        # value is 0.0 (no value model)
        assert value == 0.0

    @pytest.mark.asyncio
    async def test_get_asv_includes_system_prompt(self, mock_tools, mock_message):
        """Test that system prompt is prepended to messages."""
        agent = L2LAgent()

        mock_action = MagicMock()
        mock_action.tool_calls = None
        agent.llm_call_op = AsyncMock(return_value=mock_action)

        state = await agent.init_state(mock_tools)
        await agent.get_asv(state, [mock_message])

        # verify LLMCallOp was called with system prompt first
        call_args = agent.llm_call_op.call_args
        msgs = call_args.kwargs.get("msgs") or call_args[1].get("msgs")

        assert msgs[0].role == "system"
        assert "scientific" in msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_get_asv_tracks_hypothesis_submission(self, mock_tools, mock_message):
        """Test that hypothesis submissions are tracked in state."""
        agent = L2LAgent()

        # create mock action with submit_hypothesis tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "submit_hypothesis"
        mock_tool_call.id = "call_123"

        mock_action = MagicMock()
        mock_action.tool_calls = [mock_tool_call]
        agent.llm_call_op = AsyncMock(return_value=mock_action)

        state = await agent.init_state(mock_tools)
        _, new_state, _ = await agent.get_asv(state, [mock_message])

        # hypothesis should be tracked
        assert len(new_state.hypothesis_history) == 1
        assert new_state.hypothesis_history[0]["tool_call_id"] == "call_123"
        assert new_state.pending_hypothesis == "call_123"

    @pytest.mark.asyncio
    async def test_get_asv_preserves_state_immutability(self, mock_tools, mock_message):
        """Test that original state is not modified."""
        agent = L2LAgent()

        mock_action = MagicMock()
        mock_action.tool_calls = None
        agent.llm_call_op = AsyncMock(return_value=mock_action)

        state = await agent.init_state(mock_tools)
        original_messages = list(state.messages)

        _, new_state, _ = await agent.get_asv(state, [mock_message])

        # original state unchanged
        assert state.messages == original_messages
        # new state has updates
        assert len(new_state.messages) > len(original_messages)
