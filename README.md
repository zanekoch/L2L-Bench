# L2L Bench: Learning to Learn Scientific Reasoning Benchmark

## Motivation

Many existing benchmarks only reward prediction accuracy. These sort of benchmarks can incentivize learning domain facts. In this benchmark, we seek to reward the successful application of scientific reasoning. This is enforced by a two-layered reward loop: an inner loop rewards adherence to the scientific process, while an outer loop rewards progressive gains in accuracy arising from information gained by the inner loop. Crucially, the outer loop reward is proportional to the degree of improvement performance above baseline performance (i.e., above the agent's first attempt) – meaning that we are explicitly rewarding learning, not knowledge.

## Episode Structure

An episode E is defined by the tuple:

```
E = (Q, O, A, H, B)
```

Where:
- **Q** = Episode question (mechanistic "why" question about observed phenomenon)
- **O** = Initial observations (facts given to agent that motivate Q)
- **A** = Action space (available experiments agent can run)
- **H** = Held-out test set (curated such that progressive, successful investigation of Q enables improved prediction of H)
- **B** = Experimental budget (maximum queries allowed)

At each step t ∈ {1, ..., B}, the agent:

1. **Hypothesizes**: States testable hypothesis h_t with competing alternatives
2. **Designs**: Selects experiment a_t ∈ A
3. **Predicts**: States expected outcome ŷ_t with confidence c_t before execution
4. **Observes**: Receives result y_t from environment

## Evaluation

The inner loop of the benchmark rewards correctly conducting the scientific process (e.g., stating a falsifiable hypothesis) while the outer loop rewards progress towards understanding (performance on unseen test data).

### Per-step Process Scores S_process(t): The Inner Loop Reward

```
S_process(t) = Σ_i w_i · s_i(t)
```

| Component | Measures | Grading Method |
|-----------|----------|----------------|
| Hypothesis quality | Is h_t testable, falsifiable, and discriminating? | LLM as a judge |
| Interpretation validity | Does stated interpretation of the result logically follow from evidence? | LLM as a judge |

### Epistemic Progress Score S_progress: The Outer Loop Reward

At each step {0, 1, ..., B}, the agent predicts the answer to a set of held-out questions H without seeing results. Let φ(M_t, H) denote the prediction accuracy of the model at time step t (M_t) on H. The progress score is then the difference between the initial performance on H and the performance after t steps.

```
S_progress = φ(M_B, H) − φ(M_0, H)
```

**Note**: Even if the agent already has some baked in knowledge of H, because we measure improvement from before the first time step (i.e., "0-shot" baseline), we can still detect learning.

### Total Episode Score

The total score is a combination of the process score (how well the agent adhered to the scientific method) and the progress score (how well it gained information towards Q as measured by performance on H):

```
S_episode = Σ_{t=1}^B S_process(t) + λ · S_progress
```

**Note**: Within an episode, learning is purely in-context – the agent reasons over accumulating experimental results with no weight updates. RL training could be done on this benchmark, but would occur across episodes: after an episode, we compute S_process and S_progress and update weights via the preferred RL method.

## Key Challenges

1. **Defining (Q, H) pairs** such that the investigation of Q leads to increased performance on H without being game-able is crucial and likely challenging.

2. **LLM judging regime** that accurately rewards discerning hypotheses and interpretations in the inner loop will be challenging to construct.

3. **Programmatic generation of episodes** is highly desirable and may be achievable using large-scale datasets.

## Potential Data Sources

### Primary: Tahoe-100M
Single-cell perturbation atlas with 100M transcriptomic profiles across ~1,100 drug perturbations in 47 cancer cell lines. Annotated driver mutations and drug mechanisms enable mechanistic questions about drug response with clear ground-truth for held-out predictions.

### Additional Sources
- **Replogle et al. Perturb-seq (2022)**: Genome-wide CRISPRi screen with ~2.5M single-cell profiles in K562/RPE1 cells. Genetic perturbations with rich phenotypic readouts support episodes probing gene function and pathway relationships.
- **DepMap/CCLE**: Genetic dependency maps across 1,000+ cancer cell lines paired with comprehensive genomic annotations. Well-suited for synthetic lethality questions where molecular context predicts dependency patterns.
- **GDSC/PRISM**: Large-scale drug sensitivity screening across hundreds of cell lines with compound target annotations.
- **L1000/CMAP**: Transcriptional signatures from >1M chemical and genetic perturbations.
- **Open Targets Genetics**: Integrated GWAS and functional genomics linking genetic variants to disease traits across tissues.
- **sci-Plex**: Single-cell chemical transcriptomics with dose-response information.

---

## Example Episode: Tahoe-100M

Tahoe-100M is a single-cell perturbation atlas containing 100M transcriptomic profiles across ~1,100 drug perturbations in 47 cancer cell lines. Each cell line has annotated driver mutations (KRAS, BRAF, TP53, etc.) and each drug has annotated mechanisms of action and targets. From single-cell expression profiles, we derive quantitative readouts: proliferation signature, apoptosis signature, and pathway activation scores.

### Example Episode Setup

**Q**: "What mechanism leads to KRAS-mutant cells showing >30% proliferation reduction to Trametinib while KRAS-wildtype cells show <15% reduction?"

**O**: SW620 (KRAS G12V): Trametinib → Proliferation ↓42%; HT29 (KRAS WT): Trametinib → Proliferation ↓8%

**A**: Observe the effect of drug D in cell line C, for any D ∈ {available drugs} and C ∈ {available cell lines}

> Note: some drugs and cell lines would be held out, if they could give the answer to H directly (i.e., via memorization)

**B**: 10 steps

### Held-Out Test Set H

The agent is asked to predict whether each drug-cell pair shows >30% proliferation reduction (the threshold defining "response" in Q):

| Item | Drug | Cell Line | KRAS Status | True Answer |
|------|------|-----------|-------------|-------------|
| H1 | Binimetinib | SW480 | Mutant (G12V) | Yes (↓36%) |
| H2 | Cobimetinib | LoVo | Mutant (G13D) | Yes (↓41%) |
| H3 | Selumetinib | SW620 | Mutant (G12V) | Yes (↓37%) |
| H4 | Binimetinib | HT29 | Wildtype | No (↓11%) |
| H5 | Selumetinib | Caco2 | Wildtype | No (↓9%) |
| H6 | Cobimetinib | RKO | Wildtype | No (↓13%) |
| H7 | Alpelisib | SW620 | Mutant (G12V) | No (↓18%) |
| H8 | Venetoclax | LoVo | Mutant (G13D) | No (↓7%) |

Items H1-H3 test whether the agent generalizes "MEK inhibitors work in KRAS-mutant" across drugs and cell lines. H4-H6 test whether the agent understands KRAS-wildtype cells don't respond. H7-H8 are non-MEK drugs that serve as controls – they should remain unpredictable until the agent understands the mechanism is MEK-specific, not a general property of KRAS-mutant cells.

### Example Agent Outputs

**Baseline performance on H**: 4/8 correct (50%). The agent knows from prior training that KRAS is an oncogene involved in growth signaling, and can infer from drug naming conventions that Trametinib, Cobimetinib, Binimetinib, and Selumetinib are likely MEK inhibitors.

#### Step 1

| Phase | Agent Output | Grade |
|-------|--------------|-------|
| Hypothesis | "KRAS G12V constitutively activates growth signaling. If Trametinib targets something downstream of KRAS, other drugs targeting the same pathway should show similar KRAS-dependence. H1: Trametinib effect is KRAS-pathway-mediated. H2: Trametinib effect is coincidental to KRAS status." | 0.85 |
| Design | Observe effect of Cobimetinib in SW620 (KRAS-mutant) | – |
| Predict | "If H1: >30% reduction. If H2: No predictable pattern." | – |
| Observe | Cobimetinib in SW620 → Proliferation ↓38% | – |
| Interpret | "Result supports H1. Cobimetinib shows similar effect to Trametinib in KRAS-mutant context, suggesting shared mechanism." | 0.90 |

**Performance on H after Step 1**: 5/8 correct (62%). The agent now predicts MEK inhibitors work in KRAS-mutant lines (H1-H3 correct) but hasn't yet tested KRAS-wildtype, so H4-H6 predictions are uncertain. H7-H8 remain incorrect – agent assumes all drugs might show KRAS-dependent response.

#### Step 2

| Phase | Agent Output | Grade |
|-------|--------------|-------|
| Hypothesis | "More drugs in this class probably work in KRAS-mutant cells." | 0.25 |
| Design | Observe effect of Selumetinib in SW620 (KRAS-mutant) | – |
| Predict | "It will probably work, >30% reduction." | – |
| Observe | Selumetinib in SW620 → Proliferation ↓37% | – |
| Interpret | "As expected, this drug class works in KRAS-mutant cells. This proves MEK inhibitors are effective cancer treatments." | 0.30 |

**Performance on H after Step 2**: 5/8 correct (62%). No change – this step was uninformative. Testing a third MEK inhibitor in the same KRAS-mutant context confirmed an already-established pattern but did not discriminate any open questions. The agent still cannot predict H4-H6 (KRAS-wildtype response) or H7-H8 (non-MEK drug response). The hypothesis was vague and unfalsifiable, and the interpretation overreached ("proves") and generalized beyond the evidence ("effective cancer treatments").

#### Step 3

| Phase | Agent Output | Grade |
|-------|--------------|-------|
| Hypothesis | "If Trametinib and Cobimetinib share a KRAS-dependent mechanism, they should also fail in KRAS-wildtype cells. H1: The drug class effect is KRAS-dependent. H2: The drug class works regardless of KRAS status." | 0.90 |
| Design | Observe effect of Cobimetinib in HT29 (KRAS-wildtype) | – |
| Predict | "If H1: <15% reduction, similar to Trametinib in HT29. If H2: >30% reduction." | – |
| Observe | Cobimetinib in HT29 → Proliferation ↓9% | – |
| Interpret | "Confirmed H1. Cobimetinib shows KRAS-dependent response pattern matching Trametinib. Both drugs likely target a shared node downstream of KRAS that is only essential when KRAS is constitutively active." | 0.90 |

**Performance on H after Step 3**: 6/8 correct (75%). Agent now correctly predicts H4-H6 (MEK inhibitors fail in KRAS-wildtype). H7-H8 still incorrect – agent hasn't yet established that the effect is specific to this drug class rather than a general property of KRAS-mutant cells.

#### Step 4

| Phase | Agent Output | Grade |
|-------|--------------|-------|
| Hypothesis | "I've established a drug class with KRAS-dependent effects. To understand if this is specific to this drug class or a general KRAS-mutant property, I should test a different drug class. H1: KRAS-dependence is specific to the Trametinib/Cobimetinib class. H2: KRAS-mutant cells are generally more drug-sensitive." | 0.95 |
| Design | Observe effect of Alpelisib (PI3K inhibitor) in SW620 (KRAS-mutant) | – |
| Predict | "If H1 (class-specific): <30% reduction. If H2 (general sensitivity): >30% reduction." | – |
| Observe | Alpelisib in SW620 → Proliferation ↓18% | – |
| Interpret | "Supports H1. Alpelisib (a different drug class) does not show strong effect in KRAS-mutant cells despite Trametinib/Cobimetinib working. The KRAS-dependent response is specific to the Trametinib/Cobimetinib drug class, not a general property of KRAS-mutant cells." | 0.95 |

**Performance on H after Step 4**: 8/8 correct (100%). Critical insight gained: the agent now understands KRAS-dependence is mechanism-specific. H7-H8 now correctly predicted as non-responders – different drug classes don't show KRAS-dependent effects.

### Epistemic Progress Summary

| Checkpoint | Key Information Gained | Accuracy on H |
|------------|------------------------|---------------|
| t=0 | Prior knowledge only | 50% (4/8) |
| t=1 | Drug class shares KRAS-mutant sensitivity | 62% (5/8) |
| t=2 | (Redundant step, no new information) | 62% (5/8) |
| t=3 | Drug class fails in KRAS-wildtype | 75% (6/8) |
| t=4 | Effect is drug-class-specific, not general KRAS property | 100% (8/8) |

The progress score captures information gained through experimentation:

```
S_progress = φ(M_final, H) − φ(M_0, H) = 1.00 − 0.50 = 0.50
```

Note that Step 2 contributed poorly to both scores: low S_process (poor hypothesis and interpretation grades) and zero contribution to S_progress (no accuracy improvement). An agent that runs redundant experiments with sloppy reasoning will score poorly on both metrics – incentivizing efficient, well-reasoned experimental design.

### Total Episode Score

| Component | Value |
|-----------|-------|
| Σ S_process (across 4 steps shown) | 5.10 |
| λ · S_progress (λ=10) | 5.0 |
| S_episode | 10.1 |

---

## Implementation

### Data Layer

The `L2LData` class (`src/l2l_bench/l2l_data.py`) provides efficient access to Tahoe-100M via DuckDB queries against remote parquet files. A pre-built treatment index enables fast single-file lookups for specific drug/concentration/cell-line combinations.

Queryable data classes encapsulate treatment metadata for agent consumption:
- `Drug`: name, targets, MOA, SMILES, PubChem ID
- `CellLine`: name, organ, driver mutations
- `DriverMutation`: gene, protein effect, variant type, mechanism (GoF/LoF)
- `TreatmentCondition`: combines drug + cell line + concentration with lazy-loaded pathway activities

Factory methods (`get_drug()`, `get_cell_line()`, `get_treatment()`) construct these objects from the underlying metadata.

Pathway enrichment scores are pre-computed using GSEApy prerank with Reactome pathways and cached per treatment condition. The `TreatmentCondition` class lazy-loads these on first access and provides query methods (`get_significant_pathways()`, `get_top_activated()`, etc.).
