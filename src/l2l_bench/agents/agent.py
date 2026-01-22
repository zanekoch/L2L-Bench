"""LDP-based scientific reasoning agent for L2L Bench.

This agent inherits from ldp.agent.Agent and uses LDP's LLMCallOp for
LLM calls, enabling integration with LDP's stochastic computation graph
for gradient-based training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aviary.core import Message, Tool, ToolRequestMessage
from ldp.agent import Agent
from ldp.graph import LLMCallOp

from l2l_bench.agents.config import CLAUDE_SONNET, LLMConfig
from l2l_bench.agents.prompts import SCIENTIFIC_REASONING_SYSTEM
from l2l_bench.agents.state import L2LAgentState

if TYPE_CHECKING:
    pass


class L2LAgent(Agent[L2LAgentState]):
    """
    LDP Agent for L2L-bench scientific reasoning.

    Inherits from ldp.agent.Agent with generic type L2LAgentState.
    Uses LLMCallOp from ldp.graph for LLM calls (enables SCG training).

    The agent follows a ReAct-style architecture where it:
    1. Receives observations from the environment
    2. Maintains conversation history in state
    3. Uses an LLM to decide on tool calls
    4. Returns actions (tool requests) to the environment

    Example:
        >>> from l2l_bench.agents import L2LAgent, GPT4O_MINI
        >>> from l2l_bench.aviary import L2LEnvironment
        >>>
        >>> agent = L2LAgent(llm_config=GPT4O_MINI)
        >>> env = L2LEnvironment(question=question, data=data)
        >>>
        >>> obs, tools = await env.reset()
        >>> state = await agent.init_state(tools)
        >>>
        >>> while not done:
        ...     action, state, _ = await agent.get_asv(state, obs)
        ...     obs, reward, done, _ = await env.step(action)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        system_prompt: str | None = None,
        budget: int = 10,
    ):
        """
        Initialize L2L agent.

        Args:
            llm_config: LLM configuration (model, temperature, etc.)
            system_prompt: Custom system prompt (uses SCIENTIFIC_REASONING_SYSTEM if None)
            budget: Experiment budget (used in system prompt formatting)
        """
        super().__init__()
        self.llm_config = llm_config or CLAUDE_SONNET
        self.budget = budget

        # format system prompt with budget
        base_prompt = system_prompt or SCIENTIFIC_REASONING_SYSTEM
        self.system_prompt = base_prompt.format(budget=budget)

        # LDP's LLMCallOp for compute graph integration
        self.llm_call_op = LLMCallOp()

    async def init_state(self, tools: list[Tool]) -> L2LAgentState:
        """
        Initialize agent state with tools from env.reset().

        Args:
            tools: List of Tool objects from the environment

        Returns:
            Initial L2LAgentState with empty message history and provided tools
        """
        return L2LAgentState(
            messages=[],
            tools=tools,
            budget=self.budget,
        )

    async def get_asv(
        self,
        agent_state: L2LAgentState,
        obs: list[Message],
    ) -> tuple[ToolRequestMessage, L2LAgentState, float]:
        """
        Get action, state, value using LDP's LLMCallOp.

        This method:
        1. Appends new observations to message history
        2. Prepends system prompt to messages
        3. Calls LLM via LDP's LLMCallOp
        4. Updates state with new messages
        5. Returns (action, new_state, value_estimate)

        Args:
            agent_state: Current agent state
            obs: New observations (messages) from environment

        Returns:
            Tuple of (action, new_state, value_estimate):
                - action: ToolRequestMessage with tool calls
                - new_state: Updated L2LAgentState
                - value_estimate: Always 0.0 (no value model)
        """
        # 1. append observations to message history
        new_messages = agent_state.messages + obs

        # 2. prepend system prompt as first message
        msgs_with_system = [
            Message(role="system", content=self.system_prompt)
        ] + new_messages

        # 3. call LLM via LDP's LLMCallOp (integrates with SCG)
        action = await self.llm_call_op(
            config=self.llm_config.to_litellm_kwargs(),
            msgs=msgs_with_system,
            tools=agent_state.tools,
        )

        # 4. update state
        # append observation messages and the action to history
        updated_messages = new_messages + [action]

        # track hypothesis submissions
        hypothesis_history = agent_state.hypothesis_history
        pending_hypothesis = agent_state.pending_hypothesis

        if hasattr(action, "tool_calls") and action.tool_calls:
            for tc in action.tool_calls:
                if tc.function.name == "submit_hypothesis":
                    # record hypothesis submission
                    hypothesis_history = hypothesis_history + [
                        {
                            "step": agent_state.current_step + 1,
                            "tool_call_id": tc.id,
                        }
                    ]
                    pending_hypothesis = tc.id

        new_state = agent_state.copy(
            messages=updated_messages,
            hypothesis_history=hypothesis_history,
            pending_hypothesis=pending_hypothesis,
        )

        # 5. return (action, new_state, value_estimate)
        return action, new_state, 0.0
