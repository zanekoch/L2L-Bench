# LDP Framework Integration Plan for L2L Benchmark

## Overview

Integrate the [Future-House LDP](https://github.com/Future-House/ldp) (Language Decision Processes) framework to create agents that can answer `Question` episodes in multi-turn scientific reasoning tasks.

## Design Decisions

- **Step workflow**: Strict (hypothesis → prediction → experiment → interpretation each step)
- **Agent model**: Claude claude-sonnet-4-20250514 (via Anthropic API)
- **Judge model**: GPT-4o (via OpenAI API)
- **Approach**: MVP first - get basic environment + agent loop working, then add judge/harness

## Key Resources

- [LDP GitHub](https://github.com/Future-House/ldp) - Agent framework
- [Aviary GitHub](https://github.com/Future-House/aviary) - Environment library
- [FutureHouse Cookbook](https://edisonscientific.gitbook.io/edison-cookbook/ldp-language-decision-processes) - Documentation

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   L2LAgent      │────▶│ L2LEnvironment  │
│  (ldp.Agent)    │◀────│ (aviary.Env)    │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │    L2LData      │
                        │  (existing)     │
                        └─────────────────┘
```

---

## File Structure

```
src/l2l_bench/
├── ldp/                      # NEW MODULE
│   ├── __init__.py
│   ├── state.py              # L2LEnvState, ExperimentRecord, PredictionRecord
│   ├── tools.py              # Tool definitions (run_experiment, get_drug_info, etc.)
│   ├── environment.py        # L2LEnvironment (Aviary Environment subclass)
│   ├── agent.py              # L2LAgent (LDP Agent subclass)
│   ├── judge.py              # LLM judge for hypothesis/interpretation scoring
│   ├── scoring.py            # S_process and S_progress computation
│   └── harness.py            # EvaluationHarness for running episodes
```

---

## Implementation Details

### 1. Dependencies (`pyproject.toml`)

Add:
```toml
"ldp>=0.1.0",
"fhaviary>=0.1.0",
"litellm>=1.0.0",
"anthropic>=0.40.0",
```

### 2. Environment State (`state.py`)

```python
@dataclass
class L2LEnvState:
    question: Question
    current_step: int = 0
    budget: int = 10
    experiments: list[ExperimentRecord]      # history
    predictions: list[PredictionRecord]      # held-out predictions at each step
    current_hypothesis: str | None           # per-step submissions
    current_interpretation: str | None
    process_scores: list[float]
    baseline_accuracy: float | None
```

### 3. Tools (`tools.py`)

| Tool | Purpose |
|------|---------|
| `run_experiment(drug, conc, cell_line)` | Observe pathway activities for a treatment |
| `get_drug_info(drug)` | Get MOA, targets, approval status |
| `get_cell_line_info(cell_line)` | Get mutations, organ |
| `submit_hypothesis(hypothesis)` | Record testable hypothesis |
| `submit_prediction(prediction)` | Record expected outcome |
| `submit_interpretation(interpretation)` | Record result interpretation |
| `make_prediction_on_test_set(predictions)` | Predict held-out items |
| `list_available_drugs()` | List queryable drugs |
| `list_available_cell_lines()` | List queryable cell lines |

Tools wrap `L2LData` methods and update `L2LEnvState`.

### 4. Environment (`environment.py`)

```python
class L2LEnvironment(Environment[L2LEnvState]):
    def __init__(self, question: Question, data: L2LData, budget: int = 10): ...

    async def reset(self) -> tuple[Messages, list[Tool]]:
        # Return episode description + tools

    async def step(self, action: ToolRequestMessage) -> tuple[Messages, float, bool, bool]:
        # Execute tools, compute process score when step complete
        # Return (observations, reward, done, truncated)
```

### 5. Agent (`agent.py`)

```python
class L2LAgent:
    async def init_state(self, tools: list[Tool]) -> L2LAgentState:
        # Initialize with tools, empty message history

    async def get_asv(self, state, obs) -> tuple[Action, State, Value]:
        # Call Claude claude-sonnet-4-20250514 with history + obs, return tool calls
```

### 6. LLM Judge (`judge.py`)

Score hypotheses and interpretations on 0-1 scale using rubrics:
- **Hypothesis**: Testable? Falsifiable? Discriminating? Biologically grounded?
- **Interpretation**: References data? Valid inference? Acknowledges uncertainty?

### 7. Scoring (`scoring.py`)

- `S_process(t)` = average of hypothesis + interpretation scores
- `S_progress` = φ(M_final, H) - φ(M_0, H) (accuracy improvement)
- `S_episode` = Σ S_process + λ * S_progress

### 8. Harness (`harness.py`)

```python
class EvaluationHarness:
    async def run_episode(self, question: Question) -> EpisodeResult
    async def run_evaluation(self, questions: list[Question]) -> list[EpisodeResult]
```

---

## Episode Flow

1. **Reset**: Agent receives Question, target pathway, test set IDs, available tools
2. **Baseline**: Agent makes initial predictions on held-out set (before experiments)
3. **Loop** (budget steps):
   - Agent submits hypothesis
   - Agent submits prediction
   - Agent runs experiment → observes pathway data
   - Agent submits interpretation
   - Judge scores process quality
4. **Final**: Agent makes final predictions, compute progress score

---

## Example Usage

```python
import asyncio
from l2l_bench import L2LData, Question
from l2l_bench.ldp import L2LAgent, EvaluationHarness

async def main():
    data = L2LData()
    question = Question.from_drug_response(
        data=data,
        question="What determines MAPK pathway inhibition by Trametinib?",
        drug="Trametinib",
        concentration=0.05,
        target_pathway="MAPK6/MAPK4 Signaling R-HSA-5687128",
        positive_class_direction="repressed",
    )

    harness = EvaluationHarness(data=data, budget=10)
    result = await harness.run_episode(question)
    print(f"S_episode: {result.s_episode:.2f}")

asyncio.run(main())
```

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add ldp, fhaviary, litellm, anthropic dependencies |
| `src/l2l_bench/__init__.py` | Export new LDP classes |

## New Files to Create

| File | Purpose |
|------|---------|
| `src/l2l_bench/ldp/__init__.py` | Module exports |
| `src/l2l_bench/ldp/state.py` | State dataclasses |
| `src/l2l_bench/ldp/tools.py` | Tool definitions |
| `src/l2l_bench/ldp/environment.py` | Aviary Environment |
| `src/l2l_bench/ldp/agent.py` | LDP Agent |
| `src/l2l_bench/ldp/judge.py` | LLM scoring |
| `src/l2l_bench/ldp/scoring.py` | Episode scoring |
| `src/l2l_bench/ldp/harness.py` | Evaluation runner |

---

## Implementation Phases

### Phase 1: MVP (Start Here)
Build minimal working agent-environment loop:

1. **Dependencies**: Add `ldp`, `fhaviary`, `litellm`, `anthropic` to `pyproject.toml`
2. **State** (`state.py`): Basic `L2LEnvState` dataclass
3. **Tools** (`tools.py`): `run_experiment`, `get_drug_info`, `get_cell_line_info`, `submit_hypothesis`, `submit_interpretation`
4. **Environment** (`environment.py`): `reset()` and `step()` with strict workflow enforcement
5. **Agent** (`agent.py`): Simple Claude-based agent with tool calling

**MVP Verification**: Run a single episode manually in a notebook, observe agent making tool calls and receiving observations.

### Phase 2: Scoring (After MVP works)
1. **Judge** (`judge.py`): GPT-4o rubric scoring for hypothesis/interpretation
2. **Scoring** (`scoring.py`): `S_process` and `S_progress` computation
3. **Held-out predictions**: Add `make_prediction_on_test_set` tool

### Phase 3: Evaluation Harness
1. **Harness** (`harness.py`): Run multiple episodes, aggregate scores
2. **Notebook demo**: Full evaluation with visualizations

---

## Verification

**MVP Test** (Phase 1):
```python
# in notebooks/07_ldp_episode.ipynb
from l2l_bench import L2LData, Question
from l2l_bench.ldp import L2LEnvironment, L2LAgent

data = L2LData()
question = Question.from_drug_response(...)
env = L2LEnvironment(question, data, budget=3)
agent = L2LAgent()

obs, tools = await env.reset()
state = await agent.init_state(tools)

# manual loop - verify tool calls work
action, state, _ = await agent.get_asv(state, obs)
obs, reward, done, _ = await env.step(action)
```

**Full Test** (Phase 3):
- Run harness on 3+ questions
- Verify S_process and S_progress computed correctly
- Check episode scores match expected formula
