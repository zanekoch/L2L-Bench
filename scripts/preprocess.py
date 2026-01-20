#!/usr/bin/env python3
"""
Pre-process Tahoe-100M pseudobulk differential expression data
to compute response metrics for benchmark episodes.

This script computes proliferation/apoptosis signature scores from the
differential expression data by averaging log2FoldChange across signature genes.
"""

import argparse
import json
from pathlib import Path

try:
    import dask.dataframe as dd
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install dask pandas pyarrow")
    exit(1)


DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# HuggingFace path to pseudobulk differential expression
DE_PATH = "hf://datasets/tahoebio/Tahoe-100M/pseudobulk_differential_expression/train/*.parquet"


def load_gene_signatures() -> dict:
    """Load gene signatures from JSON file."""
    sig_path = PROCESSED_DIR / "gene_signatures.json"
    if not sig_path.exists():
        raise FileNotFoundError(
            f"Gene signatures not found at {sig_path}. "
            "Run download_data.py first."
        )
    with open(sig_path) as f:
        return json.load(f)


def compute_signature_scores(
    signature_name: str,
    genes: list[str],
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Compute signature scores from pseudobulk differential expression.

    Args:
        signature_name: name for the output column
        genes: list of gene symbols in the signature
        sample_size: optional limit on number of rows to process (for testing)

    Returns:
        DataFrame with columns: drug, cell_line, {signature_name}_log2fc,
        {signature_name}_padj, n_genes
    """
    print(f"Loading differential expression data...")
    print(f"  (This may take a while - dataset has 4B+ rows)")

    # load with dask for out-of-core processing
    de = dd.read_parquet(DE_PATH)

    if sample_size:
        print(f"  Sampling {sample_size} rows for testing...")
        de = de.head(sample_size, npartitions=-1, compute=False)

    print(f"Filtering to {len(genes)} {signature_name} genes...")
    de_sig = de[de['gene_name'].isin(genes)]

    print("Computing signature scores per drug-cell-line pair...")
    # aggregate: mean log2FC and min padj (most significant)
    scores = de_sig.groupby(['drug', 'cell_line_id']).agg({
        'log2FoldChange': 'mean',
        'padj': 'min',
        'gene_name': 'count',
    }).compute()

    scores = scores.reset_index()
    scores.columns = [
        'drug', 'cell_line',
        f'{signature_name}_log2fc',
        f'{signature_name}_padj',
        'n_genes'
    ]

    # convert log2FC to percent reduction
    # log2FC of -0.5 ≈ 29% reduction, log2FC of -1 ≈ 50% reduction
    scores[f'{signature_name}_pct_change'] = (
        (2 ** scores[f'{signature_name}_log2fc'] - 1) * 100
    )

    return scores


def merge_with_metadata(scores: pd.DataFrame) -> pd.DataFrame:
    """Merge response scores with drug and cell line metadata."""
    # load metadata
    drugs = pd.read_parquet(PROCESSED_DIR / "drug_metadata.parquet")
    cell_lines = pd.read_parquet(PROCESSED_DIR / "cell_line_metadata.parquet")

    # merge drug info
    scores = scores.merge(
        drugs[['drug', 'moa-fine', 'moa-broad', 'targets']],
        on='drug',
        how='left'
    )

    # merge cell line info
    scores = scores.merge(
        cell_lines[['cell_name', 'Driver_Gene_Symbol', 'Driver_ProtEffect_or_CdnaEffect', 'Organ']],
        left_on='cell_line',
        right_on='cell_name',
        how='left'
    )

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute response metrics from Tahoe-100M differential expression"
    )
    parser.add_argument(
        '--signature',
        choices=['proliferation', 'apoptosis', 'mapk_pathway', 'all'],
        default='all',
        help='Which signature to compute'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Process only N rows (for testing)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROCESSED_DIR / "drug_cell_responses.parquet",
        help='Output file path'
    )
    args = parser.parse_args()

    signatures = load_gene_signatures()

    if args.signature == 'all':
        sig_names = list(signatures.keys())
    else:
        sig_names = [args.signature]

    # compute each signature
    all_scores = None
    for sig_name in sig_names:
        print(f"\n=== Computing {sig_name} signature ===")
        scores = compute_signature_scores(
            sig_name,
            signatures[sig_name],
            sample_size=args.sample
        )

        if all_scores is None:
            all_scores = scores
        else:
            # merge on drug + cell_line
            all_scores = all_scores.merge(
                scores.drop(columns=['n_genes']),
                on=['drug', 'cell_line'],
                how='outer'
            )

    print("\nMerging with metadata...")
    all_scores = merge_with_metadata(all_scores)

    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_scores.to_parquet(args.output, index=False)

    print(f"\nDone! Computed responses for {len(all_scores)} drug-cell pairs")
    print(f"Columns: {list(all_scores.columns)}")


if __name__ == "__main__":
    main()
