#!/usr/bin/env python3
"""
Download and cache Tahoe-100M metadata tables from HuggingFace.

This script downloads the smaller metadata tables needed for episode generation.
The larger expression/differential expression data is loaded on-demand via dask.
"""

import argparse
import json
from pathlib import Path

# check for required packages
try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install datasets pandas pyarrow")
    exit(1)


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def download_drug_metadata(force: bool = False) -> pd.DataFrame:
    """Download drug metadata (targets, MOA, SMILES)."""
    outpath = DATA_DIR / "drug_metadata.parquet"

    if outpath.exists() and not force:
        print(f"Drug metadata already exists at {outpath}")
        return pd.read_parquet(outpath)

    print("Downloading drug metadata...")
    ds = load_dataset("tahoebio/Tahoe-100M", "drug_metadata")
    df = ds['train'].to_pandas()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath, index=False)
    print(f"Saved {len(df)} drugs to {outpath}")

    return df


def download_cell_line_metadata(force: bool = False) -> pd.DataFrame:
    """Download cell line metadata (mutations, organ)."""
    outpath = DATA_DIR / "cell_line_metadata.parquet"

    if outpath.exists() and not force:
        print(f"Cell line metadata already exists at {outpath}")
        return pd.read_parquet(outpath)

    print("Downloading cell line metadata...")
    ds = load_dataset("tahoebio/Tahoe-100M", "cell_line_metadata")
    df = ds['train'].to_pandas()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath, index=False)
    print(f"Saved {len(df)} cell lines to {outpath}")

    return df


def download_gene_metadata(force: bool = False) -> pd.DataFrame:
    """Download gene metadata (symbols, Ensembl IDs)."""
    outpath = DATA_DIR / "gene_metadata.parquet"

    if outpath.exists() and not force:
        print(f"Gene metadata already exists at {outpath}")
        return pd.read_parquet(outpath)

    print("Downloading gene metadata...")
    ds = load_dataset("tahoebio/Tahoe-100M", "gene_metadata")
    df = ds['train'].to_pandas()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath, index=False)
    print(f"Saved {len(df)} genes to {outpath}")

    return df


def save_gene_signatures():
    """Save predefined gene signatures for computing response metrics."""
    outpath = DATA_DIR / "gene_signatures.json"

    # proliferation signature - genes associated with cell cycle and division
    proliferation_genes = [
        'MKI67',   # Ki-67, classic proliferation marker
        'PCNA',    # DNA replication
        'TOP2A',   # DNA topoisomerase
        'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',  # DNA replication licensing
        'CDK1',    # cell cycle kinase
        'CCNB1', 'CCNB2',  # cyclins B
        'CCNA2',   # cyclin A
        'PLK1',    # polo-like kinase
        'AURKA', 'AURKB',  # aurora kinases
        'BUB1', 'BUB1B',   # spindle checkpoint
        'CDC20',   # cell division cycle
        'FOXM1',   # cell cycle transcription factor
        'E2F1',    # cell cycle transcription factor
        'TYMS',    # thymidylate synthase
    ]

    # apoptosis signature - genes involved in programmed cell death
    apoptosis_genes = [
        'BCL2',    # anti-apoptotic
        'BAX',     # pro-apoptotic
        'BAK1',    # pro-apoptotic
        'BID',     # BH3-only
        'PUMA',    # BH3-only (BBC3)
        'NOXA',    # BH3-only (PMAIP1)
        'CASP3',   # executioner caspase
        'CASP7',   # executioner caspase
        'CASP8',   # initiator caspase
        'CASP9',   # initiator caspase
        'CYCS',    # cytochrome c
        'APAF1',   # apoptosome
        'PARP1',   # PARP (cleaved during apoptosis)
        'XIAP',    # IAP family
        'BIRC5',   # survivin
    ]

    # MAPK/ERK pathway - relevant for MEK inhibitor responses
    mapk_pathway_genes = [
        'EGFR',    # receptor
        'KRAS', 'NRAS', 'HRAS',  # RAS family
        'BRAF', 'RAF1', 'ARAF',  # RAF family
        'MAP2K1', 'MAP2K2',      # MEK1/2
        'MAPK1', 'MAPK3',        # ERK1/2
        'DUSP1', 'DUSP4', 'DUSP6',  # MAPK phosphatases
        'ETV1', 'ETV4', 'ETV5',  # ETS transcription factors
        'SPRY1', 'SPRY2', 'SPRY4',  # sprouty (negative regulators)
    ]

    signatures = {
        'proliferation': proliferation_genes,
        'apoptosis': apoptosis_genes,
        'mapk_pathway': mapk_pathway_genes,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(signatures, f, indent=2)

    print(f"Saved gene signatures to {outpath}")
    return signatures


def main():
    parser = argparse.ArgumentParser(description="Download Tahoe-100M metadata")
    parser.add_argument('--force', action='store_true', help='Re-download even if files exist')
    parser.add_argument('--drugs-only', action='store_true', help='Only download drug metadata')
    parser.add_argument('--cells-only', action='store_true', help='Only download cell line metadata')
    parser.add_argument('--genes-only', action='store_true', help='Only download gene metadata')
    args = parser.parse_args()

    if args.drugs_only:
        download_drug_metadata(force=args.force)
    elif args.cells_only:
        download_cell_line_metadata(force=args.force)
    elif args.genes_only:
        download_gene_metadata(force=args.force)
    else:
        # download all
        download_drug_metadata(force=args.force)
        download_cell_line_metadata(force=args.force)
        download_gene_metadata(force=args.force)
        save_gene_signatures()

    print("\nDone! Metadata downloaded to:", DATA_DIR)


if __name__ == "__main__":
    main()
