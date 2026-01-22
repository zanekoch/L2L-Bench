"""Agent state model for LDP-based L2L agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aviary.core import Message, Tool


@dataclass
class L2LAgentState:
    """
    State for LDP-based L2L scientific reasoning agent.

    This state is passed through the agent's get_asv() method and tracks
    the agent's message history, tools, and scientific reasoning progress.

    Attributes:
        messages: Conversation history (user/assistant/tool messages)
        tools: Available tools from environment
        current_step: Current experimental step number
        experiments_completed: Number of experiments run
        budget: Maximum number of experiments allowed
        learned_knowledge_draft: Working draft of learned knowledge document
        pending_hypothesis: Current hypothesis awaiting experiment
        hypothesis_history: List of submitted hypotheses with metadata
        interpretation_history: List of interpretations with metadata
    """

    # core LDP state
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)

    # scientific reasoning state
    current_step: int = 0
    experiments_completed: int = 0
    budget: int = 10

    # working memory
    learned_knowledge_draft: str = ""
    pending_hypothesis: str | None = None

    # tracking
    hypothesis_history: list[dict] = field(default_factory=list)
    interpretation_history: list[dict] = field(default_factory=list)

    def copy(self, **updates) -> L2LAgentState:
        """Create a copy of this state with optional field updates."""
        import copy as copy_module

        new_state = L2LAgentState(
            messages=copy_module.copy(self.messages),
            tools=self.tools,  # tools don't change, shallow copy is fine
            current_step=self.current_step,
            experiments_completed=self.experiments_completed,
            budget=self.budget,
            learned_knowledge_draft=self.learned_knowledge_draft,
            pending_hypothesis=self.pending_hypothesis,
            hypothesis_history=copy_module.copy(self.hypothesis_history),
            interpretation_history=copy_module.copy(self.interpretation_history),
        )
        for key, value in updates.items():
            setattr(new_state, key, value)
        return new_state

    @property
    def budget_remaining(self) -> int:
        """Number of experiments remaining in budget."""
        return self.budget - self.experiments_completed

    @property
    def is_budget_exhausted(self) -> bool:
        """True if no experiments remaining."""
        return self.experiments_completed >= self.budget
