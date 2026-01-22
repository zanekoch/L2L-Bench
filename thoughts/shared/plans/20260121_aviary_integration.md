# Aviary Integration Implementation Plan

## Overview

Integrate L2L Bench with the Future-House aviary framework to create a standard gymnasium-style environment for scientific reasoning agents. This enables evaluation with any LDP-compatible agent and supports RL training via aviary's infrastructure.

## Current State Analysis

### Existing Components
- `L2LData` (`src/l2l_bench/l2l_data.py`): DuckDB-powered data access with queryable objects
- `Question` / `TestItem` (`src/l2l_bench/question.py`): Episode definition with held-out test sets
- `TreatmentCondition`, `Drug`, `CellLine`, `DriverMutation`: Rich queryable data classes
- Pathway enrichment via cached GSEA results

### What's Missing
- No formal Environment class with `reset()` / `step()` API
- No tool definitions for agent interaction
- No S_process scoring implementation (LLM-as-judge)
- No learned_knowledge tracking system

## Desired End State

A fully functional aviary Environment that:
1. Accepts a `Question` and exposes it as a gymnasium-style environment
2. Provides tools for hypothesis submission, experiment execution, interpretation, and metadata lookup
3. Computes S_process scores via configurable LLM judge
4. Tracks learned_knowledge versions as filesystem snapshots
5. Auto-restricts held-out treatments from the action space
6. Terminates when budget exhausted OR agent achieves perfect H prediction

### Verification
- Unit tests for environment lifecycle (reset → step → done)
- Integration test running a mock agent through a full episode
- S_process scoring produces reasonable grades for good/bad hypotheses

## What We're NOT Doing

- LDP agent implementation (separate task)
- RL training loop (aviary/LDP handles this)
- UI/visualization for episodes
- Multi-question batching (one Question per environment instance)

## Implementation Approach

Create a new `src/l2l_bench/aviary/` subpackage with:
1. `state.py` - Pydantic state model with `done`/`reward` following aviary convention
2. `dataset.py` - TaskDataset implementation wrapping Question
3. `environment.py` - Main Environment subclass with tools as methods
4. `scoring.py` - LLM-as-judge S_process implementation

**Key Design Decisions (aligned with aviary patterns):**
- Tools are defined as methods on `L2LEnvironment`, not standalone functions
- State uses aviary's `state.done` and `state.reward` convention (tools set these directly)
- `L2LTaskDataset` implements aviary's `TaskDataset` interface for evaluation workflows

---

## Phase 1: Project Setup & State Model

### Overview
Add fhaviary dependency and create the state model that tracks episode progress.

### Changes Required:

#### 1. Update pyproject.toml
**File**: `pyproject.toml`
**Changes**: Add fhaviary as required dependency

```toml
dependencies = [
    # ... existing deps ...
    "fhaviary>=0.5.0",
]
```

#### 2. Create aviary subpackage
**File**: `src/l2l_bench/aviary/__init__.py`

```python
"""Aviary environment integration for L2L Bench."""

from l2l_bench.aviary.environment import L2LEnvironment
from l2l_bench.aviary.state import L2LState

__all__ = ["L2LEnvironment", "L2LState"]
```

#### 3. Create state model
**File**: `src/l2l_bench/aviary/state.py`

```python
"""State model for L2L Environment."""

from pathlib import Path
from typing import TYPE_CHECKING

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
    data: "L2LData"

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

    class Config:
        arbitrary_types_allowed = True
```

### Success Criteria:

#### Automated Verification:
- [x] `pip install -e .` succeeds with fhaviary
- [x] `python -c "from l2l_bench.aviary import L2LState"` works
- [x] `pytest tests/test_aviary_state.py` passes (create basic test)

#### Manual Verification:
- [x] L2LState can be instantiated with test data

---

## Phase 2: TaskDataset Implementation

### Overview
Implement aviary's `TaskDataset` interface to enable standard evaluation workflows.
This wraps our existing `Question` class to integrate with aviary's data loading patterns.

### Changes Required:

#### 1. Create dataset module
**File**: `src/l2l_bench/aviary/dataset.py`

```python
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
            "question_id": question.id,
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
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from l2l_bench.aviary import L2LTaskDataset"` works
- [x] `pytest tests/test_aviary_dataset.py` passes
- [x] Dataset iteration produces valid task dicts

#### Manual Verification:
- [x] Can create dataset from Question list and iterate

---

## Phase 3: LLM-as-Judge Scoring

### Overview
Implement S_process scoring that evaluates hypothesis and interpretation quality.

### Changes Required:

#### 1. Create scoring module
**File**: `src/l2l_bench/aviary/scoring.py`

```python
"""LLM-as-judge scoring for S_process."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def __init__(self, model: str = "gpt-4o-mini", client: any = None):
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

    async def score_hypothesis(self, hypothesis: str, question_context: str, available_experiments: str) -> tuple[float, str]:
        return self.default_score, "Dummy score"

    async def score_interpretation(self, interpretation: str, hypothesis: str, observation: str) -> tuple[float, str]:
        return self.default_score, "Dummy score"
```

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/test_aviary_scoring.py` passes with DummyScorer
- [ ] LLMScorer integration test with mocked client

#### Manual Verification:
- [ ] LLMScorer produces reasonable scores for good vs bad hypotheses

---

## Phase 4: Environment Class

### Overview
Create the main `L2LEnvironment` class that ties everything together.

**Key design choices aligned with aviary patterns:**
1. Tools are defined as methods on the environment class (not standalone functions)
2. Tools access data via `state.data` (injected during reset)
3. Tools set `state.done` and `state.reward` directly (aviary convention)
4. `step()` reads done/reward from state rather than computing separately

### Changes Required:

#### 1. Create environment module
**File**: `src/l2l_bench/aviary/environment.py`

```python
"""L2L Environment for aviary."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from aviary.core import Environment, Message, Tool, ToolRequestMessage

from l2l_bench.aviary.state import L2LState, StepRecord
from l2l_bench.aviary.scoring import Scorer, DummyScorer

if TYPE_CHECKING:
    from l2l_bench.l2l_data import L2LData
    from l2l_bench.question import Question


class L2LEnvironment(Environment[L2LState]):
    """
    Aviary environment for L2L Bench scientific reasoning episodes.

    Wraps a Question and L2LData to provide a gymnasium-style interface
    for agents to conduct experiments and learn from observations.

    Tools are defined as methods on this class, following aviary's recommended
    pattern. They access data via state.data and set state.done/state.reward
    directly.

    Example:
        >>> from l2l_bench import L2LData, Question
        >>> from l2l_bench.aviary import L2LEnvironment
        >>>
        >>> data = L2LData()
        >>> question = Question.from_drug_response(...)
        >>> env = L2LEnvironment(question=question, data=data, budget=10)
        >>>
        >>> obs, tools = await env.reset()
        >>> # agent loop...
    """

    def __init__(
        self,
        question: Question,
        data: L2LData,
        budget: int = 10,
        scorer: Scorer | None = None,
        episodes_dir: Path | None = None,
    ):
        """
        Initialize L2L Environment.

        Args:
            question: The Question defining this episode
            data: L2LData instance for data access
            budget: Maximum number of experiments allowed
            scorer: Scorer for S_process evaluation (defaults to DummyScorer)
            episodes_dir: Directory for episode artifacts (defaults to data/episodes/)
        """
        self.question = question
        self.data = data
        self.budget = budget
        self.scorer = scorer or DummyScorer()
        self.episodes_dir = episodes_dir or Path("data/episodes")

        # will be set in reset()
        self.state: L2LState | None = None
        self.tools: list[Tool] = []

    # -------------------------------------------------------------------------
    # Tool methods (aviary pattern: tools as methods, state as last arg)
    # -------------------------------------------------------------------------

    def list_available_drugs(self, state: L2LState) -> str:
        """List all drugs available for experiments.

        Args:
            state: Environment state (hidden from agent).
        """
        drugs = state.data.list_drugs()
        return f"Available drugs ({len(drugs)} total):\n" + "\n".join(f"- {d}" for d in drugs)

    def list_available_cell_lines(self, state: L2LState) -> str:
        """List all cell lines available for experiments.

        Args:
            state: Environment state (hidden from agent).
        """
        cell_lines = state.data.list_cell_lines()
        return f"Available cell lines ({len(cell_lines)} total):\n" + "\n".join(f"- {c}" for c in cell_lines)

    def get_drug_info(self, drug_name: str, state: L2LState) -> str:
        """Get detailed information about a drug including mechanism of action and targets.

        Args:
            drug_name: Name of the drug to look up (e.g., "Trametinib")
            state: Environment state (hidden from agent).
        """
        try:
            drug = state.data.get_drug(drug_name)
            lines = [
                f"# {drug.name}",
                f"- Mechanism (broad): {drug.moa_broad}",
                f"- Mechanism (fine): {drug.moa_fine}",
                f"- Targets: {', '.join(drug.targets) if drug.targets else 'Unknown'}",
                f"- Human approved: {'Yes' if drug.human_approved else 'No'}",
                f"- Clinical trials: {'Yes' if drug.clinical_trials else 'No'}",
            ]
            if drug.pubchem_cid:
                lines.append(f"- PubChem CID: {drug.pubchem_cid}")
            return "\n".join(lines)
        except ValueError as e:
            return f"Error: {e}"

    def get_cell_line_info(self, cell_line_name: str, state: L2LState) -> str:
        """Get detailed information about a cell line including driver mutations.

        Args:
            cell_line_name: Name of the cell line to look up (e.g., "A549")
            state: Environment state (hidden from agent).
        """
        try:
            cell_line = state.data.get_cell_line(cell_line_name)
            lines = [
                f"# {cell_line.name}",
                f"- Organ/tissue: {cell_line.organ}",
                f"- Driver mutations ({len(cell_line.driver_mutations)}):",
            ]
            for mut in cell_line.driver_mutations:
                effect = f" ({mut.protein_effect})" if mut.protein_effect else ""
                mech = f" [{mut.mechanism}]" if mut.mechanism else ""
                lines.append(f"  - {mut.gene_symbol}{effect}: {mut.var_type}{mech} ({mut.gene_type})")

            if not cell_line.driver_mutations:
                lines.append("  (no annotated driver mutations)")

            return "\n".join(lines)
        except ValueError as e:
            return f"Error: {e}"

    def get_drugs_by_mechanism(self, mechanism: str, state: L2LState) -> str:
        """Find drugs with a specific mechanism of action.

        Args:
            mechanism: Mechanism to search for (partial match, e.g., "MEK inhibitor")
            state: Environment state (hidden from agent).
        """
        drugs = state.data.get_drugs_by_moa(mechanism, moa_type="fine")
        if not drugs:
            drugs = state.data.get_drugs_by_moa(mechanism, moa_type="broad")

        if drugs:
            return f"Drugs matching '{mechanism}':\n" + "\n".join(f"- {d}" for d in drugs)
        else:
            return f"No drugs found matching '{mechanism}'"

    def get_cell_lines_with_mutation(self, gene: str, state: L2LState) -> str:
        """Find cell lines with mutations in a specific gene.

        Args:
            gene: Gene symbol to search for (e.g., "KRAS", "BRAF")
            state: Environment state (hidden from agent).
        """
        df = state.data.get_cell_lines_by_mutation(gene)
        if df.empty:
            return f"No cell lines found with {gene} mutations"

        lines = [f"Cell lines with {gene} mutations:"]
        for _, row in df.iterrows():
            effect = row.get('Driver_ProtEffect_or_CdnaEffect', '')
            lines.append(f"- {row['cell_name']}: {effect}")
        return "\n".join(lines)

    def run_experiment(
        self,
        drug: str,
        concentration: float,
        cell_line: str,
        state: L2LState,
    ) -> str:
        """Run an experiment to observe pathway activities for a drug treatment.

        This tool consumes 1 unit of your experimental budget.
        You must submit a hypothesis and prediction before running an experiment.
        After observing results, you must submit an interpretation.

        Args:
            drug: Drug name (e.g., "Trametinib")
            concentration: Drug concentration in uM (e.g., 0.05, 5.0)
            cell_line: Cell line name (e.g., "A549")
            state: Environment state (hidden from agent).
        """
        # check hypothesis/prediction submitted
        if state.current_hypothesis is None:
            return "Error: You must submit a hypothesis and prediction before running an experiment. Use submit_hypothesis() first."

        # check budget
        if state.experiments_run >= state.budget:
            return f"Error: Budget exhausted. You have run {state.experiments_run}/{state.budget} experiments."

        # check restricted treatments
        treatment_key = (drug, concentration, cell_line)
        if treatment_key in state.restricted_treatments:
            return f"Error: This treatment ({drug} @ {concentration}uM in {cell_line}) is in the held-out test set and cannot be queried."

        # run experiment
        try:
            treatment = state.data.get_treatment(drug, concentration, cell_line)
        except ValueError as e:
            return f"Error: {e}"

        # get pathway activities
        sig_pathways = treatment.get_significant_pathways(fdr_threshold=0.05)
        top_activated = treatment.get_top_activated(n=5)
        top_repressed = treatment.get_top_repressed(n=5)

        # update state
        state.experiments_run += 1
        state.awaiting_interpretation = True
        state.step_history[-1].experiment = {
            "drug": drug,
            "concentration": concentration,
            "cell_line": cell_line,
        }

        # check if budget exhausted (aviary convention: set done in tool)
        if state.experiments_run >= state.budget:
            state.done = True

        # format response
        lines = [
            f"# Experiment Results: {drug} @ {concentration}uM in {cell_line}",
            f"Budget: {state.experiments_run}/{state.budget} experiments used",
            "",
            f"## Cell Line Context",
            f"- Organ: {treatment.cell_line.organ}",
            f"- Driver mutations: {', '.join(m.gene_symbol for m in treatment.cell_line.driver_mutations) or 'None annotated'}",
            "",
            f"## Drug Context",
            f"- Mechanism: {treatment.drug.moa_fine}",
            f"- Targets: {', '.join(treatment.drug.targets) if treatment.drug.targets else 'Unknown'}",
            "",
            f"## Pathway Activities",
            f"Significant pathways (FDR < 0.05): {len(sig_pathways)}",
            "",
            "### Top Activated Pathways (positive NES):",
        ]

        if not top_activated.empty:
            for _, row in top_activated.iterrows():
                lines.append(f"- {row['pathway']}: NES={row['nes']:.2f}, FDR={row['fdr']:.3f}")
        else:
            lines.append("(none)")

        lines.append("")
        lines.append("### Top Repressed Pathways (negative NES):")

        if not top_repressed.empty:
            for _, row in top_repressed.iterrows():
                lines.append(f"- {row['pathway']}: NES={row['nes']:.2f}, FDR={row['fdr']:.3f}")
        else:
            lines.append("(none)")

        # store observation in state
        state.step_history[-1].observation = {
            "significant_count": len(sig_pathways),
            "top_activated": top_activated.to_dict('records') if not top_activated.empty else [],
            "top_repressed": top_repressed.to_dict('records') if not top_repressed.empty else [],
        }

        lines.append("")
        lines.append("---")
        lines.append("Now submit your interpretation using submit_interpretation().")

        return "\n".join(lines)

    def submit_hypothesis(
        self,
        hypothesis: str,
        alternatives: str,
        prediction: str,
        confidence: str,
        state: L2LState,
    ) -> str:
        """Submit your hypothesis and prediction before running an experiment.

        A good hypothesis should be:
        - Testable: Can be evaluated with available experiments
        - Falsifiable: Clear what would disprove it
        - Discriminating: Distinguishes between competing explanations

        Args:
            hypothesis: Your primary hypothesis (H1)
            alternatives: Competing alternative hypothesis (H2) that your experiment could support instead
            prediction: What you expect to observe if H1 is correct
            confidence: Your confidence level (low/medium/high) and reasoning
            state: Environment state (hidden from agent).
        """
        if state.current_hypothesis is not None:
            return "Error: You already submitted a hypothesis for this step. Run an experiment or start a new step."

        full_hypothesis = f"H1: {hypothesis}\nH2: {alternatives}"
        full_prediction = f"Prediction: {prediction}\nConfidence: {confidence}"

        state.current_hypothesis = full_hypothesis
        state.current_prediction = full_prediction

        # record in step history
        if not state.step_history or state.step_history[-1].experiment is not None:
            # start new step record
            state.step_history.append(StepRecord(step_number=state.current_step + 1))
            state.current_step += 1

        state.step_history[-1].hypothesis = full_hypothesis
        state.step_history[-1].prediction = full_prediction

        return f"Hypothesis and prediction recorded for step {state.current_step}.\n\n{full_hypothesis}\n\n{full_prediction}\n\nNow run an experiment using run_experiment()."

    def submit_interpretation(
        self,
        interpretation: str,
        hypothesis_supported: str,
        state: L2LState,
    ) -> str:
        """Submit your interpretation of the experiment results.

        A good interpretation should:
        - Logically follow from the observed evidence
        - Not overreach beyond what the data shows
        - Update beliefs appropriately based on results

        Args:
            interpretation: Your interpretation of what the results mean
            hypothesis_supported: Which hypothesis (H1 or H2) the results support and why
            state: Environment state (hidden from agent).
        """
        if not state.awaiting_interpretation:
            return "Error: Run an experiment first before submitting an interpretation."

        full_interpretation = f"Interpretation: {interpretation}\nSupported: {hypothesis_supported}"
        state.step_history[-1].interpretation = full_interpretation
        state.awaiting_interpretation = False
        state.current_hypothesis = None
        state.current_prediction = None

        return f"Interpretation recorded.\n\n{full_interpretation}\n\nNow update your learned knowledge using update_learned_knowledge(). Set done=True if you have enough evidence, or done=False to continue experimenting."

    def update_learned_knowledge(
        self,
        new_knowledge: str,
        done: bool,
        state: L2LState,
    ) -> str:
        """Update your accumulated learned knowledge document and optionally end the episode.

        This document captures what you've learned from experiments so far.
        Write in markdown format. Be concise but capture key insights.
        This document will be used to evaluate your learning progress.

        Call this after each interpretation to record your updated understanding.
        Set done=True when you have gathered enough evidence to answer the question.

        Args:
            new_knowledge: Your updated learned knowledge document (replaces previous version)
            done: Set to True to end experimentation and proceed to held-out test evaluation
            state: Environment state (hidden from agent).
        """
        state.learned_knowledge = new_knowledge

        # save to filesystem
        knowledge_path = state.episode_dir / f"knowledge_v{state.current_step}.md"
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_path.write_text(new_knowledge)

        if done:
            # aviary convention: tool sets state.done directly
            state.done = True
            state.reward = 1.0  # base reward for completing episode
            return f"Learned knowledge updated and saved to {knowledge_path.name}.\n\nEpisode marked as complete. Proceeding to held-out test set evaluation."
        else:
            return f"Learned knowledge updated and saved to {knowledge_path.name}.\n\nYou can now start a new hypothesis-experiment cycle."

    # -------------------------------------------------------------------------
    # Environment lifecycle methods
    # -------------------------------------------------------------------------

    def _build_restricted_treatments(self) -> set[tuple[str, float, str]]:
        """Build set of treatments to restrict (from held-out test set)."""
        restricted = set()
        for item in self.question.test_set:
            key = (
                item.treatment.drug.name,
                item.treatment.concentration,
                item.treatment.cell_line.name,
            )
            restricted.add(key)
        return restricted

    def _build_tools(self) -> list[Tool]:
        """Build tool list from environment methods."""
        # aviary pattern: Tool.from_function on bound methods
        # state arg is automatically injected by aviary
        return [
            Tool.from_function(self.list_available_drugs),
            Tool.from_function(self.list_available_cell_lines),
            Tool.from_function(self.get_drug_info),
            Tool.from_function(self.get_cell_line_info),
            Tool.from_function(self.get_drugs_by_mechanism),
            Tool.from_function(self.get_cell_lines_with_mutation),
            Tool.from_function(self.run_experiment),
            Tool.from_function(self.submit_hypothesis),
            Tool.from_function(self.submit_interpretation),
            Tool.from_function(self.update_learned_knowledge),
        ]

    def _build_context(self) -> str:
        """Build the context message for the agent.

        Includes task description, resources, and current learned knowledge.
        This is called on reset() and can be called during step() to refresh context.
        """
        drugs_sample = self.data.list_drugs()[:20]
        cell_lines_sample = self.data.list_cell_lines()[:20]

        # format learned knowledge section
        if self.state.learned_knowledge:
            knowledge_section = f"## Learned Knowledge\n\n{self.state.learned_knowledge}"
        else:
            knowledge_section = "## Learned Knowledge\n\n(No learned knowledge yet. Update after your first experiment.)"

        context = f"""# L2L Benchmark Episode

## Your Task
{self.question.question}

## Target Pathway
{self.question.target_pathway}

## Experimental Budget
You have {self.budget} experiments available. Used: {self.state.experiments_run}/{self.budget}

## Available Resources
- **Drugs**: {len(self.data.list_drugs())} total (sample: {', '.join(drugs_sample[:10])}...)
- **Cell lines**: {len(self.data.list_cell_lines())} total (sample: {', '.join(cell_lines_sample[:10])}...)

{knowledge_section}

## Tools Available
Use the metadata tools to explore available drugs and cell lines.
Use submit_hypothesis() to submit your hypothesis and prediction before experiments.
Use run_experiment() to observe pathway activities (consumes budget).
Use submit_interpretation() after observing results.
Use update_learned_knowledge() to record insights and optionally end the episode.

## Scientific Process
For each experiment:
1. Submit a testable hypothesis with alternatives (H1 vs H2) and your prediction
2. Run the experiment
3. Submit your interpretation of results
4. Update your learned knowledge document (set done=True when ready to end)
"""
        return context

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        """Reset the environment for a new episode."""
        # create episode directory
        episode_id = str(uuid.uuid4())[:8]
        episode_dir = self.episodes_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # initialize state with data reference (aviary pattern)
        self.state = L2LState(
            episode_id=episode_id,
            episode_dir=episode_dir,
            question_text=self.question.question,
            target_pathway=self.question.target_pathway,
            budget=self.budget,
            data=self.data,  # inject data so tools can access via state.data
            restricted_treatments=self._build_restricted_treatments(),
        )

        # build tools from methods
        self.tools = self._build_tools()

        # build context (includes learned_knowledge, empty at start)
        context = self._build_context()

        return [Message(role="user", content=context)], self.tools

    async def step(self, action: ToolRequestMessage) -> tuple[list[Message], float, bool, bool]:
        """Execute a step in the environment."""
        # execute tool calls (tools may set state.done and state.reward)
        observations = await self.exec_tool_calls(action, state=self.state)

        # check if we need to score a completed step (adds to state.reward)
        if self.state.step_history:
            last_step = self.state.step_history[-1]
            if (last_step.interpretation is not None and
                last_step.interpretation_score is None):
                # score this step
                hyp_score, hyp_feedback = await self.scorer.score_hypothesis(
                    hypothesis=last_step.hypothesis or "",
                    question_context=self.state.question_text,
                    available_experiments="Any drug/cell line combination",
                )

                obs_str = str(last_step.observation) if last_step.observation else ""
                interp_score, interp_feedback = await self.scorer.score_interpretation(
                    interpretation=last_step.interpretation,
                    hypothesis=last_step.hypothesis or "",
                    observation=obs_str,
                )

                last_step.hypothesis_score = hyp_score
                last_step.interpretation_score = interp_score

                step_reward = (hyp_score + interp_score) / 2
                self.state.process_scores.append(step_reward)
                # aviary convention: accumulate reward on state
                self.state.reward += step_reward

        # aviary convention: read done/reward from state (tools set these)
        return observations, self.state.reward, self.state.done, False

    async def close(self) -> None:
        """Clean up resources."""
        pass
```

#### 2. Update package exports
**File**: `src/l2l_bench/aviary/__init__.py`

```python
"""Aviary environment integration for L2L Bench."""

from l2l_bench.aviary.environment import L2LEnvironment
from l2l_bench.aviary.state import L2LState
from l2l_bench.aviary.dataset import L2LTaskDataset
from l2l_bench.aviary.scoring import Scorer, LLMScorer, DummyScorer

__all__ = [
    "L2LEnvironment",
    "L2LState",
    "L2LTaskDataset",
    "Scorer",
    "LLMScorer",
    "DummyScorer",
]
```

### Success Criteria:

#### Automated Verification:
- [x] `python -c "from l2l_bench.aviary import L2LEnvironment, L2LTaskDataset"` works
- [x] `pytest tests/test_aviary_environment.py` passes
- [x] Environment can complete a full episode with mock agent
- [x] Tools correctly set state.done and state.reward

#### Manual Verification:
- [x] Episode artifacts saved to data/episodes/{id}/
- [x] knowledge_v{step}.md files created correctly

---

## Phase 5: Integration Tests

### Overview
Create comprehensive tests for the aviary integration, including TaskDataset and state.done/reward patterns.

### Changes Required:

#### 1. Create environment test file
**File**: `tests/test_aviary_environment.py`

```python
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
    mock_item = type('MockItem', (), {
        'treatment': type('MockTreatment', (), {
            'drug': type('MockDrug', (), {'name': 'RestrictedDrug'})(),
            'concentration': 1.0,
            'cell_line': type('MockCellLine', (), {'name': 'RestrictedCell'})(),
        })()
    })()
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
```

#### 2. Create TaskDataset test file
**File**: `tests/test_aviary_dataset.py`

```python
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
```

### Success Criteria:

#### Automated Verification:
- [x] `pytest tests/test_aviary*.py -v` passes
- [ ] Coverage > 80% for aviary subpackage
- [x] TaskDataset tests verify aviary interface compliance
- [x] State done/reward tests verify aviary convention

#### Manual Verification:
- [ ] Run example notebook demonstrating full episode

---

## Testing Strategy

### Unit Tests
- State model serialization/deserialization
- State has `data` reference accessible by tools
- Scorer prompts produce valid JSON
- TaskDataset iteration and indexing

### Integration Tests
- Full episode lifecycle with DummyScorer
- Restricted treatment enforcement
- Knowledge versioning to filesystem
- Budget exhaustion termination sets `state.done` (aviary convention)
- `update_learned_knowledge(done=True)` sets `state.done` and `state.reward` (aviary convention)
- Tools access data via `state.data` (not separate parameter)

### Aviary Pattern Compliance Tests
- Tools are methods on Environment class
- `state` is last parameter and hidden from agent
- `Tool.from_function()` works on bound methods
- `state.done` and `state.reward` control episode flow
- TaskDataset implements expected interface

### Manual Testing Steps
1. Create a Question using existing data
2. Initialize L2LEnvironment
3. Run through hypothesis → experiment → interpretation cycle
4. Verify knowledge files saved correctly
5. Check S_process scores are reasonable with LLMScorer
6. Test with aviary's `RolloutManager` if possible

## Performance Considerations

- Tool methods should remain synchronous where possible (aviary handles async wrapping)
- LLM scoring adds latency - consider batching or caching for repeated evaluations
- Pathway data is lazy-loaded, first access may be slow
- `state.data` reference avoids repeated data lookups

## Dependencies

```
fhaviary>=0.5.0
openai>=1.0.0  # for LLMScorer (or anthropic, etc.)
```

## Aviary Patterns Used

This implementation follows aviary's recommended patterns:

| Pattern | Our Implementation |
|---------|-------------------|
| Tools as methods | All tools defined as methods on `L2LEnvironment` |
| State injection | `state` is last parameter, hidden from agent |
| `state.done` convention | `update_learned_knowledge(done=True)` and `run_experiment()` (on budget exhaustion) set `state.done = True` |
| `state.reward` convention | Tools and step() accumulate rewards on `state.reward` |
| `TaskDataset` interface | `L2LTaskDataset` wraps Question for evaluation workflows |
| `Tool.from_function()` | Used on bound methods in `_build_tools()` |

## References

- Aviary GitHub: https://github.com/Future-House/aviary
- Aviary README patterns: Environment, Tool.from_function, state.done/reward, TaskDataset
- LDP GitHub: https://github.com/Future-House/ldp
- Current implementation: `src/l2l_bench/l2l_data.py`, `src/l2l_bench/question.py`
