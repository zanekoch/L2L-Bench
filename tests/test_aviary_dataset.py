"""Tests for L2L TaskDataset."""

import pytest

from l2l_bench.aviary import L2LTaskDataset


@pytest.fixture
def mock_questions(mocker):
    """Create mock Question list."""
    questions = []
    for i in range(10):
        q = mocker.MagicMock()
        q.id = f"q_{i:03d}"
        q.question = f"Question {i}"
        q.target_pathway = f"Pathway {i}"
        q.test_set = []
        questions.append(q)
    return questions


def test_dataset_length(mock_questions):
    """Test dataset reports correct length."""
    dataset = L2LTaskDataset(questions=mock_questions)
    assert len(dataset) == 10


def test_dataset_iteration(mock_questions):
    """Test dataset iteration yields task dicts."""
    dataset = L2LTaskDataset(questions=mock_questions)

    tasks = list(dataset)
    assert len(tasks) == 10

    # verify task structure
    task = tasks[0]
    assert "question_id" in task
    assert "question" in task
    assert "question_text" in task
    assert "target_pathway" in task
    assert "split" in task


def test_dataset_indexing(mock_questions):
    """Test dataset indexing."""
    dataset = L2LTaskDataset(questions=mock_questions)

    task = dataset[5]
    assert task["question_id"] == "q_005"


def test_dataset_split(mock_questions):
    """Test train/val/test split."""
    dataset = L2LTaskDataset(questions=mock_questions)

    train, val, test = dataset.get_train_val_test_split(
        train_frac=0.6,
        val_frac=0.2,
        seed=42,
    )

    assert train.split == "train"
    assert val.split == "val"
    assert test.split == "test"
    assert len(train) + len(val) + len(test) == len(mock_questions)


def test_dataset_from_questions(mock_questions):
    """Test factory method."""
    dataset = L2LTaskDataset.from_questions(mock_questions, split="eval")
    assert len(dataset) == 10
    assert dataset.split == "eval"


def test_dataset_task_structure(mock_questions):
    """Test that task dict contains expected fields."""
    dataset = L2LTaskDataset(questions=mock_questions, split="test")

    task = dataset[0]

    # check all required fields
    assert task["question_id"] == "q_000"
    assert task["question"] is mock_questions[0]  # full Question object
    assert task["question_text"] == "Question 0"
    assert task["target_pathway"] == "Pathway 0"
    assert task["test_set_size"] == 0
    assert task["split"] == "test"


def test_dataset_name(mock_questions):
    """Test dataset name attribute."""
    dataset = L2LTaskDataset(questions=mock_questions, name="my_dataset")
    assert dataset.name == "my_dataset"


def test_dataset_empty():
    """Test empty dataset."""
    dataset = L2LTaskDataset(questions=[])
    assert len(dataset) == 0
    assert list(dataset) == []


def test_dataset_split_reproducible(mock_questions):
    """Test that split is reproducible with same seed."""
    dataset = L2LTaskDataset(questions=mock_questions)

    train1, val1, test1 = dataset.get_train_val_test_split(seed=42)
    train2, val2, test2 = dataset.get_train_val_test_split(seed=42)

    # same seed should produce same splits
    assert [t["question_id"] for t in train1] == [t["question_id"] for t in train2]
    assert [t["question_id"] for t in val1] == [t["question_id"] for t in val2]
    assert [t["question_id"] for t in test1] == [t["question_id"] for t in test2]


def test_dataset_split_different_seeds(mock_questions):
    """Test that different seeds produce different splits."""
    dataset = L2LTaskDataset(questions=mock_questions)

    train1, _, _ = dataset.get_train_val_test_split(seed=42)
    train2, _, _ = dataset.get_train_val_test_split(seed=123)

    # different seeds should produce different order
    train1_ids = [t["question_id"] for t in train1]
    train2_ids = [t["question_id"] for t in train2]
    assert train1_ids != train2_ids
