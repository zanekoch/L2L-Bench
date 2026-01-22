"""Evaluation harness for running L2L agents through episodes.

The harness orchestrates:
1. Environment setup with a Question
2. Agent initialization
3. Episode execution (agent-environment loop)
4. Result collection and scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from l2l_bench.aviary import DummyScorer, L2LEnvironment, Scorer
from l2l_bench.agents.agent import L2LAgent

if TYPE_CHECKING:
    from l2l_bench.l2l_data import L2LData
    from l2l_bench.question import Question


@dataclass
class EpisodeResult:
    """Results from a single episode run.

    Attributes:
        question_id: Identifier for the question (episode_id from env)
        question_text: The research question text
        s_process: Average score across hypothesis and interpretation quality
        s_progress: Held-out test accuracy improvement (not yet implemented)
        s_episode: Combined episode score (s_process for now)
        steps_completed: Number of experimental steps completed
        experiments_run: Number of experiments executed
        trajectory: List of step records with hypothesis/experiment/interpretation
        learned_knowledge: Final learned knowledge document
        total_reward: Cumulative reward from environment
    """

    question_id: str
    question_text: str
    s_process: float
    s_progress: float
    s_episode: float
    steps_completed: int
    experiments_run: int
    trajectory: list[dict]
    learned_knowledge: str
    total_reward: float


@dataclass
class EvaluationResult:
    """Aggregated results from running multiple episodes.

    Attributes:
        episodes: List of individual episode results
        mean_s_process: Mean S_process across all episodes
        mean_s_progress: Mean S_progress across all episodes (placeholder)
        mean_s_episode: Mean combined score
        total_episodes: Number of episodes run
        total_experiments: Total experiments across all episodes
    """

    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def mean_s_process(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.s_process for e in self.episodes) / len(self.episodes)

    @property
    def mean_s_progress(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.s_progress for e in self.episodes) / len(self.episodes)

    @property
    def mean_s_episode(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.s_episode for e in self.episodes) / len(self.episodes)

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_experiments(self) -> int:
        return sum(e.experiments_run for e in self.episodes)


class EvaluationHarness:
    """
    Harness for running L2L agents through evaluation episodes.

    Handles the agent-environment interaction loop and collects results
    for analysis and scoring.

    Example:
        >>> from l2l_bench import L2LData, Question
        >>> from l2l_bench.agents import L2LAgent, EvaluationHarness, GPT4O_MINI
        >>> from l2l_bench.aviary import DummyScorer
        >>>
        >>> data = L2LData()
        >>> question = Question.from_drug_response(...)
        >>> agent = L2LAgent(llm_config=GPT4O_MINI)
        >>> harness = EvaluationHarness(data=data, agent=agent, scorer=DummyScorer())
        >>>
        >>> result = await harness.run_episode(question)
        >>> print(f"S_episode: {result.s_episode}")
    """

    def __init__(
        self,
        data: L2LData,
        agent: L2LAgent,
        scorer: Scorer | None = None,
        budget: int = 10,
        episodes_dir: Path | None = None,
    ):
        """
        Initialize evaluation harness.

        Args:
            data: L2LData instance for data access
            agent: L2LAgent to evaluate
            scorer: Scorer for S_process evaluation (defaults to DummyScorer)
            budget: Maximum experiments per episode
            episodes_dir: Directory for episode artifacts
        """
        self.data = data
        self.agent = agent
        self.scorer = scorer or DummyScorer()
        self.budget = budget
        self.episodes_dir = episodes_dir or Path("data/episodes")

    async def run_episode(
        self,
        question: Question,
        max_turns: int | None = None,
    ) -> EpisodeResult:
        """
        Run agent through a single episode.

        Args:
            question: The Question defining this episode
            max_turns: Maximum number of agent turns (safety limit)

        Returns:
            EpisodeResult with scores and trajectory
        """
        max_turns = max_turns or (self.budget * 5)  # generous limit

        # create environment
        env = L2LEnvironment(
            question=question,
            data=self.data,
            budget=self.budget,
            scorer=self.scorer,
            episodes_dir=self.episodes_dir,
        )

        # reset environment and agent
        obs, tools = await env.reset()
        agent_state = await self.agent.init_state(tools)

        # run episode loop
        done = False
        total_reward = 0.0
        turn = 0

        while not done and turn < max_turns:
            # get agent action
            action, agent_state, _ = await self.agent.get_asv(agent_state, obs)

            # execute in environment
            obs, reward, done, _ = await env.step(action)
            total_reward = reward  # env accumulates reward
            turn += 1

        # close environment
        await env.close()

        # extract results from environment state
        state = env.state
        trajectory = [
            {
                "step": step.step_number,
                "hypothesis": step.hypothesis,
                "hypothesis_score": step.hypothesis_score,
                "experiment": step.experiment,
                "observation": step.observation,
                "interpretation": step.interpretation,
                "interpretation_score": step.interpretation_score,
            }
            for step in state.step_history
        ]

        # compute S_process (average of hypothesis and interpretation scores)
        if state.process_scores:
            s_process = sum(state.process_scores) / len(state.process_scores)
        else:
            s_process = 0.0

        # S_progress would require held-out test evaluation (not yet implemented)
        s_progress = 0.0

        # combined episode score (for now just S_process)
        s_episode = s_process

        return EpisodeResult(
            question_id=state.episode_id,
            question_text=state.question_text,
            s_process=s_process,
            s_progress=s_progress,
            s_episode=s_episode,
            steps_completed=state.current_step,
            experiments_run=state.experiments_run,
            trajectory=trajectory,
            learned_knowledge=state.learned_knowledge,
            total_reward=total_reward,
        )

    async def run_evaluation(
        self,
        questions: list[Question],
        max_turns_per_episode: int | None = None,
    ) -> EvaluationResult:
        """
        Run agent on multiple questions and aggregate results.

        Args:
            questions: List of Questions to evaluate on
            max_turns_per_episode: Safety limit per episode

        Returns:
            EvaluationResult with aggregated statistics
        """
        result = EvaluationResult()

        for question in questions:
            episode_result = await self.run_episode(
                question=question,
                max_turns=max_turns_per_episode,
            )
            result.episodes.append(episode_result)

        return result
