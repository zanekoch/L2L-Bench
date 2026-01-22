"""LLM-as-judge scoring for S_process."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProcessScore:
    """Score for a single step's scientific process."""

    hypothesis_score: float  # 0-1
    interpretation_score: float  # 0-1
    hypothesis_feedback: str
    interpretation_feedback: str

    @property
    def total(self) -> float:
        """Combined score for this step."""
        return (self.hypothesis_score + self.interpretation_score) / 2


class Scorer(ABC):
    """Abstract base class for S_process scoring."""

    @abstractmethod
    async def score_hypothesis(
        self,
        hypothesis: str,
        question_context: str,
        available_experiments: str,
    ) -> tuple[float, str]:
        """Score a hypothesis for quality.

        Returns:
            (score, feedback) where score is 0-1
        """
        pass

    @abstractmethod
    async def score_interpretation(
        self,
        interpretation: str,
        hypothesis: str,
        observation: str,
    ) -> tuple[float, str]:
        """Score an interpretation for validity.

        Returns:
            (score, feedback) where score is 0-1
        """
        pass


class LLMScorer(Scorer):
    """LLM-based scorer for S_process."""

    HYPOTHESIS_PROMPT = '''You are evaluating the quality of a scientific hypothesis.

Context: {question_context}

Available experiments: {available_experiments}

Hypothesis submitted:
{hypothesis}

Rate this hypothesis on a scale of 0.0 to 1.0 based on:
1. Testable: Can it be evaluated with available experiments?
2. Falsifiable: Is it clear what would disprove it?
3. Discriminating: Does it distinguish between competing explanations?

Respond with JSON:
{{"score": <float 0-1>, "feedback": "<brief explanation>"}}
'''

    INTERPRETATION_PROMPT = '''You are evaluating the validity of a scientific interpretation.

Original hypothesis:
{hypothesis}

Observed results:
{observation}

Interpretation submitted:
{interpretation}

Rate this interpretation on a scale of 0.0 to 1.0 based on:
1. Logical: Does it follow from the observed evidence?
2. Calibrated: Does it avoid overreaching beyond what data shows?
3. Appropriate: Does it update beliefs correctly based on results?

Respond with JSON:
{{"score": <float 0-1>, "feedback": "<brief explanation>"}}
'''

    def __init__(self, model: str = "gpt-4o-mini", client: Any = None):
        """
        Initialize LLM scorer.

        Args:
            model: Model name to use for scoring
            client: Optional pre-configured client (OpenAI, Anthropic, etc.)
        """
        self.model = model
        self._client = client

    @property
    def client(self):
        """Lazy-load client."""
        if self._client is None:
            # default to OpenAI
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client

    async def _query_llm(self, prompt: str) -> dict:
        """Query LLM and parse JSON response."""
        import json

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    async def score_hypothesis(
        self,
        hypothesis: str,
        question_context: str,
        available_experiments: str,
    ) -> tuple[float, str]:
        prompt = self.HYPOTHESIS_PROMPT.format(
            hypothesis=hypothesis,
            question_context=question_context,
            available_experiments=available_experiments,
        )
        result = await self._query_llm(prompt)
        return result["score"], result["feedback"]

    async def score_interpretation(
        self,
        interpretation: str,
        hypothesis: str,
        observation: str,
    ) -> tuple[float, str]:
        prompt = self.INTERPRETATION_PROMPT.format(
            interpretation=interpretation,
            hypothesis=hypothesis,
            observation=observation,
        )
        result = await self._query_llm(prompt)
        return result["score"], result["feedback"]


class DummyScorer(Scorer):
    """Dummy scorer for testing (returns constant scores)."""

    def __init__(self, default_score: float = 0.5):
        self.default_score = default_score

    async def score_hypothesis(
        self, hypothesis: str, question_context: str, available_experiments: str
    ) -> tuple[float, str]:
        return self.default_score, "Dummy score"

    async def score_interpretation(
        self, interpretation: str, hypothesis: str, observation: str
    ) -> tuple[float, str]:
        return self.default_score, "Dummy score"
