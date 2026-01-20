# Tahoe-100M Data Acquisition Plan

## Overview

The Tahoe-100M dataset on HuggingFace provides the substrate for our benchmark episodes. This document outlines how to obtain and pre-process the data we need.

## Dataset Structure on HuggingFace

The dataset (`tahoebio/Tahoe-100M`) contains 7 configurations:

| Configuration | Rows | Relevance to Benchmark |
|---------------|------|------------------------|
| `expression_data` | 95.6M | Raw gene expression per cell |
| `pseudobulk_differential_expression` | 4.09B | **Primary**: log2FoldChange, p-values per gene/drug/cell-line |
| `drug_metadata` | 379 | **Essential**: targets, MOA, SMILES |
| `cell_line_metadata` | 1K | **Essential**: driver mutations, organ |
| `gene_metadata` | 62.7K | Gene symbols, Ensembl IDs |
| `obs_metadata` | 101M | Cell cycle phase, QC metrics |
| `sample_metadata` | 1.34K | Aggregated statistics per sample |

## Data We Need for Episodes

### 1. Cell Line Annotations
From `cell_line_metadata`:
- `cell_name`: identifier
- `Driver_Gene_Symbol`: e.g., KRAS, BRAF, TP53
- `Driver_ProtEffect_or_CdnaEffect`: e.g., G12V, V600E
- `Organ`: tissue of origin

### 2. Drug Annotations
From `drug_metadata`:
- `drug`: compound name
- `targets`: protein targets
- `moa-broad` / `moa-fine`: mechanism of action (e.g., MEK inhibitor)

### 3. Response Measurements
**Challenge**: The benchmark example uses "proliferation signature reduction" (e.g., ↓42%), but this isn't directly available. Options:

#### Option A: Derive from Pseudobulk Differential Expression
- Use `pseudobulk_differential_expression` which has log2FoldChange per gene
- Compute proliferation signature by averaging log2FC across proliferation-related genes (e.g., MKI67, PCNA, MCM genes)
- Compute apoptosis signature similarly

#### Option B: Use E-distance as Response Metric
- Paper computes "E-distance" measuring separability of perturbed vs control populations
- Not directly available in HuggingFace, would need to compute from expression data

#### Option C: Compute from Gene Set Scores
- Paper uses Vision scores for pathway activation
- Could compute proliferation pathway scores from gene expression data

**Recommendation**: Start with Option A (pseudobulk differential expression) as it's pre-computed and available.

## Pre-processing Pipeline

### Step 1: Download Metadata Tables
```python
from datasets import load_dataset

# small tables - load fully
drugs = load_dataset("tahoebio/Tahoe-100M", "drug_metadata")['train'].to_pandas()
cell_lines = load_dataset("tahoebio/Tahoe-100M", "cell_line_metadata")['train'].to_pandas()
genes = load_dataset("tahoebio/Tahoe-100M", "gene_metadata")['train'].to_pandas()
```

### Step 2: Define Gene Signatures
```python
# proliferation signature genes (example)
PROLIFERATION_GENES = [
    'MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
    'CDK1', 'CCNB1', 'CCNA2', 'PLK1', 'AURKA', 'AURKB', 'BUB1', 'CDC20'
]

# apoptosis signature genes (example)
APOPTOSIS_GENES = [
    'BCL2', 'BAX', 'CASP3', 'CASP8', 'CASP9', 'CYCS', 'APAF1', 'BID', 'PARP1'
]
```

### Step 3: Compute Signature Scores from Differential Expression
```python
import dask.dataframe as dd

# load differential expression (large - use dask)
de_path = "hf://datasets/tahoebio/Tahoe-100M/pseudobulk_differential_expression/train/*.parquet"
de = dd.read_parquet(de_path)

# filter to signature genes and aggregate
prolif_de = de[de['gene_name'].isin(PROLIFERATION_GENES)]
prolif_scores = prolif_de.groupby(['drug', 'cell_line_id']).agg({
    'log2FoldChange': 'mean',
    'padj': 'min'  # most significant p-value
}).compute()
```

### Step 4: Create Episode-Ready Dataset
```python
# merge with metadata
episode_data = prolif_scores.merge(drugs[['drug', 'moa-fine', 'targets']], on='drug')
episode_data = episode_data.merge(
    cell_lines[['cell_name', 'Driver_Gene_Symbol', 'Driver_ProtEffect_or_CdnaEffect']],
    left_on='cell_line_id', right_on='cell_name'
)

# convert log2FC to percent reduction
# log2FC of -0.5 ≈ 30% reduction, log2FC of -1 ≈ 50% reduction
episode_data['pct_reduction'] = (1 - 2**episode_data['log2FoldChange']) * 100
```

## Output Schema

The pre-processed data should be stored as:

```
data/
├── processed/
│   ├── drug_cell_responses.parquet      # main response matrix
│   ├── drug_metadata.parquet            # drug annotations
│   ├── cell_line_metadata.parquet       # cell line annotations
│   └── gene_signatures.json             # signature gene lists
├── episodes/
│   └── example_kras_mek/                # example episode
│       ├── config.yaml                  # Q, O, A, H, B definition
│       └── ground_truth.parquet         # held-out answers
```

## Response Matrix Schema

`drug_cell_responses.parquet`:
| Column | Type | Description |
|--------|------|-------------|
| drug | string | compound name |
| cell_line | string | cell line name |
| prolif_log2fc | float | mean log2FC of proliferation genes |
| prolif_pct_reduction | float | percent reduction in proliferation |
| apoptosis_log2fc | float | mean log2FC of apoptosis genes |
| n_cells_treated | int | number of cells in treatment |
| significant | bool | padj < 0.05 for signature |

## Next Steps

1. [ ] Write data download script (`scripts/download_data.py`)
2. [ ] Write pre-processing pipeline (`scripts/preprocess.py`)
3. [ ] Validate against example episode in README
4. [ ] Generate additional episodes programmatically

## Storage Estimates

- Raw expression data: ~500GB (don't download unless needed)
- Pseudobulk DE: ~50GB (main source for responses)
- Metadata tables: ~100MB
- Processed episode data: ~1GB
