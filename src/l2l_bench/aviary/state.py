"""State model for L2L Environment."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from l2l_bench.l2l_data import L2LData


class StepRecord(BaseModel):
    """Record of a single experimental step."""

    step_number: int
    hypothesis: str | None = None
    hypothesis_score: float | None = None
    experiment: dict | None = None  # {drug, concentration, cell_line}
    observation: dict | None = None  # pathway activities, metadata
    prediction: str | None = None
    interpretation: str | None = None
    interpretation_score: float | None = None


class L2LState(BaseModel):
    """
    State for L2L Environment.

    Tracks episode progress, accumulated observations, and scoring.

    Follows aviary convention: tools can set `state.done = True` and
    `state.reward = X` directly to control episode termination and rewards.
    """

    # episode identity
    episode_id: str
    episode_dir: Path

    # question info (for reference, not mutable)
    question_text: str
    target_pathway: str
    budget: int

    # data access (injected so tools can access it via state)
    data: Any  # L2LData - use Any to avoid Pydantic validation issues

    # progress tracking
    current_step: int = 0
    experiments_run: int = 0

    # current step state (reset each step)
    current_hypothesis: str | None = None
    current_prediction: str | None = None
    awaiting_interpretation: bool = False

    # accumulated data
    step_history: list[StepRecord] = Field(default_factory=list)
    learned_knowledge: str = ""

    # restricted treatments (held-out test set)
    restricted_treatments: set[tuple[str, float, str]] = Field(default_factory=set)

    # scoring
    process_scores: list[float] = Field(default_factory=list)

    # aviary convention: episode status controlled by tools
    # tools set these directly (e.g., update_learned_knowledge sets done=True when done=True arg passed)
    reward: float = 0.0
    done: bool = False

    model_config = {"arbitrary_types_allowed": True}
