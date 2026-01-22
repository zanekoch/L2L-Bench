"""Tests for EvaluationHarness."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from l2l_bench.agents.harness import EpisodeResult, EvaluationHarness, EvaluationResult


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_episode_result_creation(self):
        """Test EpisodeResult can be created with all fields."""
        result = EpisodeResult(
            question_id="test_123",
            question_text="What determines pathway X?",
            s_process=0.75,
            s_progress=0.0,
            s_episode=0.75,
            steps_completed=3,
            experiments_run=3,
            trajectory=[{"step": 1}, {"step": 2}],
            learned_knowledge="Some insights",
            total_reward=1.5,
        )

        assert result.question_id == "test_123"
        assert result.s_process == 0.75
        assert result.experiments_run == 3


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_empty_evaluation_result(self):
        """Test EvaluationResult with no episodes."""
        result = EvaluationResult()

        assert result.total_episodes == 0
        assert result.mean_s_process == 0.0
        assert result.mean_s_episode == 0.0
        assert result.total_experiments == 0

    def test_evaluation_result_aggregation(self):
        """Test EvaluationResult aggregates correctly."""
        result = EvaluationResult()

        result.episodes.append(
            EpisodeResult(
                question_id="1",
                question_text="Q1",
                s_process=0.6,
                s_progress=0.0,
                s_episode=0.6,
                steps_completed=2,
                experiments_run=2,
                trajectory=[],
                learned_knowledge="",
                total_reward=0.6,
            )
        )
        result.episodes.append(
            EpisodeResult(
                question_id="2",
                question_text="Q2",
                s_process=0.8,
                s_progress=0.0,
                s_episode=0.8,
                steps_completed=3,
                experiments_run=4,
                trajectory=[],
                learned_knowledge="",
                total_reward=0.8,
            )
        )

        assert result.total_episodes == 2
        assert result.mean_s_process == 0.7  # (0.6 + 0.8) / 2
        assert result.mean_s_episode == 0.7
        assert result.total_experiments == 6  # 2 + 4


class TestEvaluationHarness:
    """Tests for EvaluationHarness."""

    @pytest.fixture
    def mock_data(self):
        """Create mock L2LData."""
        return MagicMock()

    @pytest.fixture
    def mock_agent(self):
        """Create mock L2LAgent."""
        agent = MagicMock()
        agent.init_state = AsyncMock(return_value=MagicMock())
        agent.get_asv = AsyncMock(return_value=(MagicMock(), MagicMock(), 0.0))
        return agent

    @pytest.fixture
    def mock_question(self):
        """Create mock Question."""
        question = MagicMock()
        question.question = "Test question"
        question.target_pathway = "Test pathway"
        question.test_set = []
        return question

    def test_harness_initialization(self, mock_data, mock_agent):
        """Test harness initializes with correct defaults."""
        harness = EvaluationHarness(data=mock_data, agent=mock_agent)

        assert harness.data == mock_data
        assert harness.agent == mock_agent
        assert harness.budget == 10

    def test_harness_custom_budget(self, mock_data, mock_agent):
        """Test harness with custom budget."""
        harness = EvaluationHarness(data=mock_data, agent=mock_agent, budget=5)

        assert harness.budget == 5

    @pytest.mark.asyncio
    async def test_run_episode_basic(self, mock_data, mock_agent, mock_question):
        """Test run_episode creates environment and runs agent loop."""
        harness = EvaluationHarness(data=mock_data, agent=mock_agent, budget=2)

        # mock environment
        with patch("l2l_bench.agents.harness.L2LEnvironment") as MockEnv:
            mock_env = MagicMock()
            mock_env.reset = AsyncMock(return_value=([MagicMock()], []))
            mock_env.step = AsyncMock(return_value=([MagicMock()], 0.5, True, False))
            mock_env.close = AsyncMock()
            mock_env.state = MagicMock()
            mock_env.state.episode_id = "test_ep"
            mock_env.state.question_text = "Test Q"
            mock_env.state.current_step = 1
            mock_env.state.experiments_run = 1
            mock_env.state.step_history = []
            mock_env.state.process_scores = [0.5]
            mock_env.state.learned_knowledge = "learned"
            MockEnv.return_value = mock_env

            result = await harness.run_episode(mock_question)

            # verify environment was created
            MockEnv.assert_called_once()

            # verify result structure
            assert isinstance(result, EpisodeResult)
            assert result.question_id == "test_ep"
            assert result.s_process == 0.5

    @pytest.mark.asyncio
    async def test_run_evaluation_multiple_questions(
        self, mock_data, mock_agent, mock_question
    ):
        """Test run_evaluation handles multiple questions."""
        harness = EvaluationHarness(data=mock_data, agent=mock_agent)

        with patch.object(harness, "run_episode") as mock_run:
            mock_run.return_value = EpisodeResult(
                question_id="test",
                question_text="Q",
                s_process=0.5,
                s_progress=0.0,
                s_episode=0.5,
                steps_completed=1,
                experiments_run=1,
                trajectory=[],
                learned_knowledge="",
                total_reward=0.5,
            )

            questions = [mock_question, mock_question, mock_question]
            result = await harness.run_evaluation(questions)

            assert mock_run.call_count == 3
            assert result.total_episodes == 3
            assert result.mean_s_process == 0.5
