"""L2L Environment for aviary."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from aviary.core import Environment, Message, Tool, ToolRequestMessage

from l2l_bench.aviary.scoring import DummyScorer, Scorer
from l2l_bench.aviary.state import L2LState, StepRecord

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
        return f"Available drugs ({len(drugs)} total):\n" + "\n".join(
            f"- {d}" for d in drugs
        )

    def list_available_cell_lines(self, state: L2LState) -> str:
        """List all cell lines available for experiments.

        Args:
            state: Environment state (hidden from agent).
        """
        cell_lines = state.data.list_cell_lines()
        return f"Available cell lines ({len(cell_lines)} total):\n" + "\n".join(
            f"- {c}" for c in cell_lines
        )

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
                lines.append(
                    f"  - {mut.gene_symbol}{effect}: {mut.var_type}{mech} ({mut.gene_type})"
                )

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
            effect = row.get("Driver_ProtEffect_or_CdnaEffect", "")
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
            "## Cell Line Context",
            f"- Organ: {treatment.cell_line.organ}",
            f"- Driver mutations: {', '.join(m.gene_symbol for m in treatment.cell_line.driver_mutations) or 'None annotated'}",
            "",
            "## Drug Context",
            f"- Mechanism: {treatment.drug.moa_fine}",
            f"- Targets: {', '.join(treatment.drug.targets) if treatment.drug.targets else 'Unknown'}",
            "",
            "## Pathway Activities",
            f"Significant pathways (FDR < 0.05): {len(sig_pathways)}",
            "",
            "### Top Activated Pathways (positive NES):",
        ]

        if not top_activated.empty:
            for _, row in top_activated.iterrows():
                lines.append(
                    f"- {row['pathway']}: NES={row['nes']:.2f}, FDR={row['fdr']:.3f}"
                )
        else:
            lines.append("(none)")

        lines.append("")
        lines.append("### Top Repressed Pathways (negative NES):")

        if not top_repressed.empty:
            for _, row in top_repressed.iterrows():
                lines.append(
                    f"- {row['pathway']}: NES={row['nes']:.2f}, FDR={row['fdr']:.3f}"
                )
        else:
            lines.append("(none)")

        # store observation in state
        state.step_history[-1].observation = {
            "significant_count": len(sig_pathways),
            "top_activated": top_activated.to_dict("records")
            if not top_activated.empty
            else [],
            "top_repressed": top_repressed.to_dict("records")
            if not top_repressed.empty
            else [],
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

        full_interpretation = (
            f"Interpretation: {interpretation}\nSupported: {hypothesis_supported}"
        )
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
            knowledge_section = (
                f"## Learned Knowledge\n\n{self.state.learned_knowledge}"
            )
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

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        """Execute a step in the environment."""
        # execute tool calls (tools may set state.done and state.reward)
        observations = await self.exec_tool_calls(action, state=self.state)

        # check if we need to score a completed step (adds to state.reward)
        if self.state.step_history:
            last_step = self.state.step_history[-1]
            if (
                last_step.interpretation is not None
                and last_step.interpretation_score is None
            ):
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
