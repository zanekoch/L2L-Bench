"""
Question data classes for L2L Benchmark episodes.

Defines the structure for benchmark questions (episodes) that test
whether agents can learn to predict pathway activity from experimental evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from l2l_bench.l2l_data import L2LData, TreatmentCondition


@dataclass
class TestItem:
    """
    A single test item in the held-out test set.

    Attributes:
        treatment: The treatment condition being tested
        ground_truth: True if pathway shows significant activity in expected direction
        pathway_nes: Actual NES score for reference
        pathway_fdr: Actual FDR value for reference
    """

    treatment: TreatmentCondition
    ground_truth: bool
    pathway_nes: float
    pathway_fdr: float


@dataclass
class Question:
    """
    Represents a single benchmark question/episode.

    A Question defines a mechanistic "why" question about pathway activity,
    along with a held-out test set where ground truth is computed from
    pathway enrichment scores.

    Attributes:
        question: The mechanistic "why" question
        target_pathway: Exact Reactome pathway name (e.g., "MAPK6/MAPK4 Signaling R-HSA-5687128")
        drug: Drug name used in the question
        concentration: Drug concentration used
        positive_class_direction: What direction counts as "active"
            - 'activated': ground_truth = True if NES > 0 AND FDR < threshold
            - 'repressed': ground_truth = True if NES < 0 AND FDR < threshold
            - 'either': ground_truth = True if FDR < threshold (any direction)
        test_set: Held-out test items with ground truth
        fdr_threshold: Threshold for significance (default 0.05)
    """

    question: str
    target_pathway: str
    drug: str
    concentration: float
    positive_class_direction: Literal['activated', 'repressed', 'either']
    test_set: list[TestItem]
    fdr_threshold: float = 0.05

    @classmethod
    def from_drug_response(
        cls,
        data: L2LData,
        question: str,
        drug: str,
        concentration: float,
        target_pathway: str,
        positive_class_direction: Literal['activated', 'repressed', 'either'],
        fdr_threshold: float = 0.05,
        reactome_cache_dir: Path | None = None,
    ) -> 'Question':
        """
        Create a Question by auto-discovering all eligible treatments.

        Scans the reactome cache directory to find all cell lines that:
        1. Have been treated with the specified drug at the specified concentration
        2. Have pathway activity data for the target pathway

        Args:
            data: L2LData instance for creating TreatmentCondition objects
            question: The mechanistic "why" question
            drug: Drug name to filter treatments
            concentration: Drug concentration to filter treatments
            target_pathway: Exact Reactome pathway name to look up
            positive_class_direction: Direction that counts as "active"
            fdr_threshold: FDR threshold for significance
            reactome_cache_dir: Path to reactome cache directory (defaults to data/processed/reactome)

        Returns:
            Question with computed ground truth for all eligible treatments

        Raises:
            ValueError: If no eligible treatments found
        """
        from l2l_bench.l2l_data import REACTOME_CACHE_DIR

        if reactome_cache_dir is None:
            reactome_cache_dir = REACTOME_CACHE_DIR

        # build expected filename for this drug/concentration
        safe_drug = drug.replace("/", "_").replace(" ", "_")
        rounded_conc = round(concentration, 1)
        expected_filename = f"{safe_drug}_{rounded_conc}.parquet"

        # scan cache directory to find all cell lines with this treatment
        eligible_cell_lines = []
        for cell_dir in reactome_cache_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            pathway_file = cell_dir / expected_filename
            if pathway_file.exists():
                # check if the target pathway exists in this file
                df = pd.read_parquet(pathway_file)
                if target_pathway in df['pathway'].values:
                    eligible_cell_lines.append(cell_dir.name)

        if not eligible_cell_lines:
            raise ValueError(
                f"No eligible treatments found for {drug} at {concentration}uM "
                f"with pathway '{target_pathway}'"
            )

        # create TreatmentCondition objects and compute ground truth
        test_items = []
        for cell_line in eligible_cell_lines:
            treatment = data.get_treatment(drug, concentration, cell_line)
            pathway_data = treatment.get_pathway(target_pathway)

            # pathway_data should exist since we checked above
            nes = float(pathway_data['nes'])
            fdr = float(pathway_data['fdr'])

            if positive_class_direction == 'activated':
                ground_truth = nes > 0 and fdr < fdr_threshold
            elif positive_class_direction == 'repressed':
                ground_truth = nes < 0 and fdr < fdr_threshold
            else:  # either
                ground_truth = fdr < fdr_threshold

            test_items.append(TestItem(
                treatment=treatment,
                ground_truth=ground_truth,
                pathway_nes=nes,
                pathway_fdr=fdr,
            ))

        return cls(
            question=question,
            target_pathway=target_pathway,
            drug=drug,
            concentration=concentration,
            positive_class_direction=positive_class_direction,
            test_set=test_items,
            fdr_threshold=fdr_threshold,
        )
