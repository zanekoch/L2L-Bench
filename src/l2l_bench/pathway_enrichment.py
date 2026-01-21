"""
Pathway enrichment analysis using GSEApy prerank with Reactome pathways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

from l2l_bench.l2l_data import L2LData, _get_root_dir


@dataclass
class PathwayEnrichment:
    """
    Compute and cache Reactome pathway enrichment scores for treatment conditions.

    Uses GSEApy prerank method with sign(logFC) * -log10(padj) as ranking metric.
    This combines effect size direction with statistical significance.
    Results are cached to data/processed/reactome/ for efficient retrieval.
    """

    data: L2LData
    cache_dir: Path = field(default_factory=lambda: _get_root_dir() / "data/processed/reactome")
    gene_set_library: str = "Reactome_2022"
    min_size: int = 15
    max_size: int = 500
    permutation_num: int = 1000

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _round_concentration(self, concentration: float) -> float:
        """Round concentration to 1 decimal place to avoid float precision issues."""
        return round(concentration, 1)

    def _get_cache_path(self, drug: str, concentration: float, cell_line: str) -> Path:
        """Get path to cached enrichment results."""
        # sanitize drug name for filesystem
        safe_drug = drug.replace("/", "_").replace(" ", "_")
        rounded_conc = self._round_concentration(concentration)
        cell_dir = self.cache_dir / cell_line
        cell_dir.mkdir(parents=True, exist_ok=True)
        return cell_dir / f"{safe_drug}_{rounded_conc}.parquet"

    def _is_cached(self, drug: str, concentration: float, cell_line: str) -> bool:
        """Check if enrichment results are already cached."""
        return self._get_cache_path(drug, concentration, cell_line).exists()

    def _load_cached(self, drug: str, concentration: float, cell_line: str) -> pd.DataFrame:
        """Load cached enrichment results."""
        return pd.read_parquet(self._get_cache_path(drug, concentration, cell_line))

    def _save_cache(self, df: pd.DataFrame, drug: str, concentration: float, cell_line: str) -> None:
        """Save enrichment results to cache atomically."""
        import os
        import tempfile

        cache_path = self._get_cache_path(drug, concentration, cell_line)

        # round concentration in the saved data for consistency
        df = df.copy()
        if 'concentration' in df.columns:
            df['concentration'] = self._round_concentration(concentration)

        # write to temp file, then atomic rename
        fd, temp_path = tempfile.mkstemp(suffix='.parquet.tmp', dir=cache_path.parent)
        try:
            os.close(fd)
            df.to_parquet(temp_path, index=False)
            os.replace(temp_path, cache_path)  # atomic on POSIX
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def compute_enrichment_from_df(
        self,
        de_data: pd.DataFrame,
        drug: str,
        concentration: float,
        cell_line: str,
        force_recompute: bool = False,
        verbose: bool = True,
        threads: int = 4,
    ) -> pd.DataFrame:
        """
        Compute Reactome pathway enrichment from pre-loaded DataFrame.

        This method avoids re-downloading data when DE data is already loaded,
        useful for single-pass processing during index building.

        Args:
            de_data: DataFrame with columns gene_name, log2FoldChange, padj
            drug: Drug name
            concentration: Drug concentration
            cell_line: Cell line name
            force_recompute: If True, recompute even if cached
            verbose: If True, print progress messages
            threads: Number of threads for GSEApy prerank (default 4, use 1 for parallel workers)

        Returns:
            DataFrame with columns: pathway, nes, pvalue, fdr, leading_edge
        """
        # check cache first
        if not force_recompute and self._is_cached(drug, concentration, cell_line):
            if verbose:
                print(f"Loading cached enrichment for {drug}@{concentration} in {cell_line}")
            return self._load_cached(drug, concentration, cell_line)

        if len(de_data) == 0:
            raise ValueError(f"No expression data found for {drug}@{concentration} in {cell_line}")

        # prepare ranked gene list for prerank
        # use sign(logFC) * -log10(pvalue) as ranking metric
        # this combines effect size direction with statistical significance
        ranked = de_data[['gene_name', 'log2FoldChange', 'padj']].dropna()

        # filter out genes without official symbols (ENSG* entries)
        # ~1/3 of genes lack HGNC symbols and use Ensembl IDs as fallback
        # these won't match any pathway gene sets, so we exclude them
        before_filter = len(ranked)
        ranked = ranked[~ranked['gene_name'].str.startswith('ENSG', na=False)]
        filtered_count = before_filter - len(ranked)

        # compute ranking metric: sign(logFC) * -log10(padj)
        # clamp padj to avoid -log10(0) = inf (use 1e-300 as floor)
        padj_clamped = ranked['padj'].clip(lower=1e-300)
        ranked['rank_metric'] = np.sign(ranked['log2FoldChange']) * -np.log10(padj_clamped)

        # convert gene names to uppercase (required by Enrichr libraries)
        ranked['gene_name'] = ranked['gene_name'].str.upper()
        ranked = ranked.drop_duplicates(subset='gene_name')
        ranked = ranked.set_index('gene_name')['rank_metric']

        if verbose:
            print(f"Running prerank enrichment for {drug}@{concentration} in {cell_line}...")
            print(f"  {len(ranked)} genes in ranked list (filtered {filtered_count} without official symbols)")

        # run GSEApy prerank
        pre_res = gp.prerank(
            rnk=ranked,
            gene_sets=self.gene_set_library,
            min_size=self.min_size,
            max_size=self.max_size,
            permutation_num=self.permutation_num,
            outdir=None,  # don't save figures
            seed=42,
            verbose=False,
            threads=threads,
        )

        # extract results
        results = pre_res.res2d.copy()
        results = results.rename(columns={
            'Term': 'pathway',
            'NES': 'nes',
            'NOM p-val': 'pvalue',
            'FDR q-val': 'fdr',
            'Lead_genes': 'leading_edge',
        })
        results = results[['pathway', 'nes', 'pvalue', 'fdr', 'leading_edge']]

        # convert numeric columns (GSEApy returns them as object dtype)
        for col in ['nes', 'pvalue', 'fdr']:
            results[col] = pd.to_numeric(results[col], errors='coerce')

        # add treatment metadata
        results['drug'] = drug
        results['concentration'] = concentration
        results['cell_line'] = cell_line

        # cache results
        self._save_cache(results, drug, concentration, cell_line)
        if verbose:
            print(f"  Cached {len(results)} pathway scores")

        return results

    def compute_enrichment(
        self,
        drug: str,
        concentration: float,
        cell_line: str,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Compute Reactome pathway enrichment for a treatment condition.

        Uses GSEApy prerank with sign(logFC) * -log10(padj) as ranking metric.
        Results are cached to avoid recomputation.

        Args:
            drug: Drug name
            concentration: Drug concentration
            cell_line: Cell line name
            force_recompute: If True, recompute even if cached

        Returns:
            DataFrame with columns: pathway, nes, pvalue, fdr, leading_edge
        """
        # check cache first (avoids downloading if already cached)
        if not force_recompute and self._is_cached(drug, concentration, cell_line):
            print(f"Loading cached enrichment for {drug}@{concentration} in {cell_line}")
            return self._load_cached(drug, concentration, cell_line)

        # get differential expression data (uses indexed single-file download)
        de_data = self.data.get_expression_data_indexed(
            drug=drug,
            concentration=concentration,
            cell_line=cell_line
        )

        # delegate to compute_enrichment_from_df
        return self.compute_enrichment_from_df(
            de_data=de_data,
            drug=drug,
            concentration=concentration,
            cell_line=cell_line,
            force_recompute=True,  # we already checked cache above
            verbose=True,
        )

    def get_pathway_score(
        self,
        pathway: str,
        drug: str,
        concentration: float,
        cell_line: str,
    ) -> dict:
        """
        Get enrichment score for a specific pathway in a treatment condition.

        Computes enrichment if not already cached.

        Args:
            pathway: Pathway name (partial match supported)
            drug: Drug name
            concentration: Drug concentration
            cell_line: Cell line name

        Returns:
            Dict with keys: pathway, nes, pvalue, fdr, leading_edge

        Raises:
            ValueError: If pathway not found in results
        """
        # ensure we have enrichment results (compute if needed)
        if not self._is_cached(drug, concentration, cell_line):
            self.compute_enrichment(drug, concentration, cell_line)

        results = self._load_cached(drug, concentration, cell_line)

        # find matching pathway (case-insensitive partial match)
        mask = results['pathway'].str.contains(pathway, case=False, na=False)
        matches = results[mask]

        if len(matches) == 0:
            available = results['pathway'].head(10).tolist()
            raise ValueError(
                f"Pathway '{pathway}' not found. "
                f"Example pathways: {available}"
            )

        if len(matches) > 1:
            # return exact match if exists, else first partial match
            exact = matches[matches['pathway'].str.lower() == pathway.lower()]
            if len(exact) == 1:
                matches = exact
            else:
                print(f"Warning: Multiple pathways match '{pathway}', returning first")

        row = matches.iloc[0]
        return {
            'pathway': row['pathway'],
            'nes': row['nes'],
            'pvalue': row['pvalue'],
            'fdr': row['fdr'],
            'leading_edge': row['leading_edge'],
        }

    def list_pathways(self, drug: str, concentration: float, cell_line: str) -> list[str]:
        """List all pathway names for a treatment condition."""
        if not self._is_cached(drug, concentration, cell_line):
            self.compute_enrichment(drug, concentration, cell_line)

        results = self._load_cached(drug, concentration, cell_line)
        return results['pathway'].tolist()

    def compute_enrichment_for_cell_line(
        self,
        cell_line: str,
        drugs: list[str] | None = None,
        skip_cached: bool = True,
    ) -> pd.DataFrame:
        """
        Compute pathway enrichment for all treatments in a cell line.

        Discovers actual drug/concentration combinations from the DE data,
        then computes enrichment for each.

        Args:
            cell_line: Cell line name
            drugs: Optional list of drugs to filter to. If None, processes all available.
            skip_cached: If True, skip treatments that are already cached.

        Returns:
            Combined DataFrame with all pathway scores for the cell line.
        """
        # get available treatments from the index (built in Phase 2)
        treatments = self.data.get_available_treatments_for_cell_line(cell_line)

        if drugs is not None:
            treatments = treatments[treatments['drug'].isin(drugs)]

        print(f"Processing {len(treatments)} treatments for {cell_line}")

        all_results = []

        for _, row in treatments.iterrows():
            drug = row['drug']
            conc = row['concentration']

            if skip_cached and self._is_cached(drug, conc, cell_line):
                print(f"  Skipping {drug}@{conc} (cached)")
                all_results.append(self._load_cached(drug, conc, cell_line))
                continue

            try:
                result = self.compute_enrichment(drug, conc, cell_line)
                all_results.append(result)
            except Exception as e:
                print(f"  Warning: Failed {drug}@{conc}: {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)
