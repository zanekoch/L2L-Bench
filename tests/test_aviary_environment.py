"""Tests for L2L aviary environment."""

import pytest
from pathlib import Path

from l2l_bench.aviary import L2LEnvironment, L2LState, DummyScorer


@pytest.fixture
def mock_data(mocker):
    """Create mock L2LData."""
    mock = mocker.MagicMock()
    mock.list_drugs.return_value = ["DrugA", "DrugB", "DrugC"]
    mock.list_cell_lines.return_value = ["CellLine1", "CellLine2"]
    return mock


@pytest.fixture
def mock_question(mocker):
    """Create mock Question."""
    mock = mocker.MagicMock()
    mock.question = "What pathway is affected by DrugA?"
    mock.target_pathway = "MAPK signaling"
    mock.test_set = []
    mock.id = "test_q_001"
    return mock


@pytest.mark.asyncio
async def test_environment_reset(mock_data, mock_question, tmp_path):
    """Test environment reset returns correct initial state."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        scorer=DummyScorer(),
        episodes_dir=tmp_path,
    )

    obs, tools = await env.reset()

    assert len(obs) == 1
    assert "L2L Benchmark Episode" in obs[0].content
    assert len(tools) > 0
    assert env.state is not None
    assert env.state.budget == 5
    # verify data is accessible via state
    assert env.state.data is mock_data


@pytest.mark.asyncio
async def test_state_done_reward_convention(mock_data, mock_question, tmp_path):
    """Test that update_learned_knowledge sets state.done and state.reward (aviary convention)."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        scorer=DummyScorer(),
        episodes_dir=tmp_path,
    )

    await env.reset()

    # initially, done should be False and reward 0
    assert env.state.done is False
    assert env.state.reward == 0.0

    # calling update_learned_knowledge with done=True should set done=True and reward=1.0
    result = env.update_learned_knowledge("Some insights", done=True, state=env.state)
    assert env.state.done is True
    assert env.state.reward == 1.0
    assert "complete" in result.lower()


@pytest.mark.asyncio
async def test_tools_are_methods(mock_data, mock_question, tmp_path):
    """Test that tools are defined as environment methods."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    # tools should be bound methods
    tool_names = [t.info.name for t in env.tools]
    assert "list_available_drugs" in tool_names
    assert "run_experiment" in tool_names
    assert "submit_hypothesis" in tool_names
    assert "update_learned_knowledge" in tool_names
    # these tools were removed:
    assert "signal_done" not in tool_names
    assert "get_current_knowledge" not in tool_names
    assert "submit_prediction" not in tool_names


@pytest.mark.asyncio
async def test_restricted_treatments(mock_data, mock_question, tmp_path):
    """Test that held-out treatments are restricted."""
    # add a restricted treatment
    mock_item = type(
        "MockItem",
        (),
        {
            "treatment": type(
                "MockTreatment",
                (),
                {
                    "drug": type("MockDrug", (), {"name": "RestrictedDrug"})(),
                    "concentration": 1.0,
                    "cell_line": type("MockCellLine", (), {"name": "RestrictedCell"})(),
                },
            )()
        },
    )()
    mock_question.test_set = [mock_item]

    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    # verify restricted treatment is tracked
    assert ("RestrictedDrug", 1.0, "RestrictedCell") in env.state.restricted_treatments


@pytest.mark.asyncio
async def test_budget_exhaustion_sets_done(mock_data, mock_question, tmp_path, mocker):
    """Test that exhausting budget sets state.done (aviary convention)."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=1,  # only 1 experiment allowed
        episodes_dir=tmp_path,
    )

    await env.reset()

    # mock treatment data
    mock_treatment = mocker.MagicMock()
    mock_treatment.cell_line.organ = "lung"
    mock_treatment.cell_line.driver_mutations = []
    mock_treatment.drug.moa_fine = "kinase inhibitor"
    mock_treatment.drug.targets = ["EGFR"]
    mock_treatment.get_significant_pathways.return_value = []
    mock_treatment.get_top_activated.return_value = mocker.MagicMock(empty=True)
    mock_treatment.get_top_repressed.return_value = mocker.MagicMock(empty=True)
    mock_data.get_treatment.return_value = mock_treatment

    # submit hypothesis with prediction (combined tool)
    env.submit_hypothesis(
        hypothesis="H1 test",
        alternatives="H2 test",
        prediction="Expect pathway X activation",
        confidence="medium",
        state=env.state,
    )

    # run experiment (this should exhaust budget and set done=True)
    env.run_experiment("DrugA", 1.0, "CellLine1", env.state)

    assert env.state.experiments_run == 1
    assert env.state.done is True  # aviary convention: tool sets done


@pytest.mark.asyncio
async def test_context_includes_learned_knowledge(mock_data, mock_question, tmp_path):
    """Test that context includes learned_knowledge section."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    obs, tools = await env.reset()

    # context should include learned knowledge section (empty at start)
    assert "## Learned Knowledge" in obs[0].content
    assert "(No learned knowledge yet" in obs[0].content

    # update learned knowledge
    env.state.learned_knowledge = "MEK inhibitors affect MAPK pathway"

    # rebuild context should include the knowledge
    context = env._build_context()
    assert "MEK inhibitors affect MAPK pathway" in context


@pytest.mark.asyncio
async def test_hypothesis_flow(mock_data, mock_question, tmp_path):
    """Test hypothesis submission flow."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    # initially no hypothesis
    assert env.state.current_hypothesis is None

    # submit hypothesis
    result = env.submit_hypothesis(
        hypothesis="MEK inhibitors suppress MAPK pathway",
        alternatives="MEK inhibitors have no effect on MAPK pathway",
        prediction="MAPK pathway will show negative NES",
        confidence="high - based on known mechanism",
        state=env.state,
    )

    assert "recorded" in result.lower()
    assert env.state.current_hypothesis is not None
    assert "H1" in env.state.current_hypothesis
    assert "H2" in env.state.current_hypothesis

    # cannot submit another hypothesis before experiment
    error_result = env.submit_hypothesis(
        hypothesis="Another hypothesis",
        alternatives="Alt",
        prediction="Pred",
        confidence="low",
        state=env.state,
    )
    assert "error" in error_result.lower()


@pytest.mark.asyncio
async def test_list_drugs_tool(mock_data, mock_question, tmp_path):
    """Test list_available_drugs tool."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    result = env.list_available_drugs(env.state)
    assert "DrugA" in result
    assert "DrugB" in result
    assert "3 total" in result


@pytest.mark.asyncio
async def test_list_cell_lines_tool(mock_data, mock_question, tmp_path):
    """Test list_available_cell_lines tool."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    result = env.list_available_cell_lines(env.state)
    assert "CellLine1" in result
    assert "CellLine2" in result
    assert "2 total" in result


@pytest.mark.asyncio
async def test_knowledge_file_created(mock_data, mock_question, tmp_path, mocker):
    """Test that knowledge files are created in episode directory."""
    env = L2LEnvironment(
        question=mock_question,
        data=mock_data,
        budget=5,
        episodes_dir=tmp_path,
    )

    await env.reset()

    # mock an experiment cycle to get to step 1
    mock_treatment = mocker.MagicMock()
    mock_treatment.cell_line.organ = "lung"
    mock_treatment.cell_line.driver_mutations = []
    mock_treatment.drug.moa_fine = "kinase inhibitor"
    mock_treatment.drug.targets = ["EGFR"]
    mock_treatment.get_significant_pathways.return_value = []
    mock_treatment.get_top_activated.return_value = mocker.MagicMock(empty=True)
    mock_treatment.get_top_repressed.return_value = mocker.MagicMock(empty=True)
    mock_data.get_treatment.return_value = mock_treatment

    # submit hypothesis
    env.submit_hypothesis(
        hypothesis="Test hypothesis",
        alternatives="Test alternative",
        prediction="Test prediction",
        confidence="medium",
        state=env.state,
    )

    # run experiment
    env.run_experiment("DrugA", 1.0, "CellLine1", env.state)

    # submit interpretation
    env.submit_interpretation(
        interpretation="The results show X",
        hypothesis_supported="H1 - because of Y",
        state=env.state,
    )

    # update knowledge
    knowledge_content = "# What I learned\n\nDrugA affects CellLine1..."
    env.update_learned_knowledge(knowledge_content, done=False, state=env.state)

    # verify file exists
    knowledge_file = env.state.episode_dir / "knowledge_v1.md"
    assert knowledge_file.exists()
    assert knowledge_file.read_text() == knowledge_content
