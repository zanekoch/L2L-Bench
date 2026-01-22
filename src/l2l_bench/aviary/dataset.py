"""TaskDataset implementation for L2L Bench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from aviary.core import TaskDataset

if TYPE_CHECKING:
    from l2l_bench.question import Question


class L2LTaskDataset(TaskDataset):
    """
    Aviary TaskDataset wrapping L2L Bench questions.

    Enables standard aviary evaluation workflows:
        dataset = L2LTaskDataset.from_name("pathway_prediction")
        for task in dataset:
            env = L2LEnvironment.from_task(task)
            # run agent...

    Example:
        >>> from l2l_bench.aviary import L2LTaskDataset
        >>> dataset = L2LTaskDataset(questions=my_questions, split="test")
        >>> len(dataset)
        50
        >>> task = dataset[0]
        >>> task["question_id"]
        'q_001'
    """

    def __init__(
        self,
        questions: list[Question],
        split: str = "test",
        name: str = "l2l_bench",
    ):
        """
        Initialize L2L TaskDataset.

        Args:
            questions: List of Question objects to include
            split: Dataset split name (train/val/test)
            name: Dataset name for identification
        """
        self.questions = questions
        self.split = split
        self.name = name

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[dict]:
        for q in self.questions:
            yield self._question_to_task(q)

    def __getitem__(self, idx: int) -> dict:
        return self._question_to_task(self.questions[idx])

    def _question_to_task(self, question: Question) -> dict:
        """Convert a Question to an aviary task dict."""
        return {
            "question_id": question.id if hasattr(question, "id") else str(hash(question.question)),
            "question": question,  # pass full Question object
            "question_text": question.question,
            "target_pathway": question.target_pathway,
            "test_set_size": len(question.test_set),
            "split": self.split,
        }

    @classmethod
    def from_questions(
        cls,
        questions: list[Question],
        split: str = "test",
    ) -> "L2LTaskDataset":
        """Create dataset from a list of Questions."""
        return cls(questions=questions, split=split)

    def get_train_val_test_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> tuple["L2LTaskDataset", "L2LTaskDataset", "L2LTaskDataset"]:
        """Split dataset into train/val/test sets."""
        import random

        random.seed(seed)
        shuffled = list(self.questions)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        return (
            L2LTaskDataset(shuffled[:train_end], split="train", name=self.name),
            L2LTaskDataset(shuffled[train_end:val_end], split="val", name=self.name),
            L2LTaskDataset(shuffled[val_end:], split="test", name=self.name),
        )
