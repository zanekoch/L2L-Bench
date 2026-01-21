# Reactome Pathway Enrichment Implementation Plan

## Overview

Implement pathway enrichment scoring for drug/concentration/cell_line treatment conditions using GSEApy's prerank method with Reactome pathways. Results will be cached to `data/processed/reactome/` with one file per treatment tuple.

## Current State Analysis

### Existing Infrastructure:
- `L2LData` class in `src/l2l_bench/l2l_data.py` provides:
  - `get_expression_data(drug, concentration, cell_line)` → returns DataFrame with `gene_name`, `log2FoldChange`, `padj`, etc.
  - Metadata access: `drug_metadata`, `cell_line_metadata`, `gene_metadata`
  - Helper methods: `list_drugs()`, `list_cell_lines()`, `get_drugs_by_moa()`

### Data Structure:
- Expression data columns: `gene_name`, `log2FoldChange`, `padj`, `baseMean`, `drug`, `concentration`, `Cell_Name_Vevo`
- Existing processed data in `data/processed/`: metadata parquets, gene_signatures.json

### Key Constraint:
- DuckDB httpfs downloads entire parquet files (~75MB each) before filtering
- However, each parquet file contains ~64 complete treatments (not split across files)
- Data is sorted by: cell_line → drug (alphabetically)
- This means we can potentially download only specific files if we know the index

### Data Partitioning (1026 parquet files):
- ~4 billion rows total → ~4 million rows per file
- Each treatment = ~62,000 rows (one per gene)
- ~64 treatments per file
- Treatments are NOT split across files (important for efficient querying)
- Files are named `train-00000-of-01026.parquet` through `train-01025-of-01026.parquet`

## Desired End State

After implementation:
1. New `PathwayEnrichment` class that composes with `L2LData`
2. Method `compute_enrichment(drug, concentration, cell_line)` → runs GSEApy prerank
3. Method `get_pathway_score(pathway, drug, concentration, cell_line)` → returns cached score or computes if missing
4. Cached results in `data/processed/reactome/{cell_line}/{drug}_{concentration}.parquet`
5. Each cache file contains: pathway_name, NES, pvalue, fdr, leading_edge_genes

### Verification:
```python
from l2l_bench import L2LData, PathwayEnrichment

data = L2LData()
enrichment = PathwayEnrichment(data)

# compute for one treatment
result = enrichment.compute_enrichment(drug="Trametinib", concentration=0.05, cell_line="A549")
# result is DataFrame with pathway scores

# get specific pathway score (uses cache)
score = enrichment.get_pathway_score(
    pathway="Reactome_MAPK_Pathway",
    drug="Trametinib",
    concentration=0.05,
    cell_line="A549"
)
```

## What We're NOT Doing

- Not implementing GSEA (requires expression matrix), only prerank (uses ranked gene list)
- Not implementing ssGSEA (single-sample, requires raw expression)
- Not processing all cell lines at once (too slow)
- Not storing raw GSEApy output figures (only numerical results)

## Implementation Approach

Use composition pattern: `PathwayEnrichment` wraps `L2LData` and adds enrichment capabilities. This keeps concerns separated and allows independent testing.

### Treatment Index Strategy
Since treatments are NOT split across files, we can build an index:
1. On first use, stream through data once to build index: `(cell_line, drug, conc) → file_number`
2. Save index to `data/processed/treatment_index.parquet`
3. For subsequent queries, load index and download only the specific file(s) needed
4. This reduces download from ~50GB (all files) to ~75MB (one file) per query

### GSEApy prerank workflow:
1. Look up file number from treatment index
2. Download only that specific parquet file
3. Filter to target treatment → DataFrame with gene_name, log2FoldChange
4. Rank genes by log2FoldChange (positive = upregulated, negative = downregulated)
5. Run `gseapy.prerank()` with Reactome_2022 gene sets
6. Extract NES, p-values, FDR from results
7. Cache to parquet (one file per drug/concentration/cell_line tuple)

### Treatment discovery:
- Use the treatment index to list available (drug, concentration) for a cell line
- No need to scan full dataset after index is built
- Concentrations vary between treatments (not always 0.05 µM)

---

## Phase 1: Add Dependencies and Create Module Structure

### Overview
Set up the module structure and add gseapy dependency.

### Changes Required:

#### 1. Update pyproject.toml
**File**: `pyproject.toml`
**Changes**: Add gseapy dependency

```toml
dependencies = [
    "datasets>=2.14.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "dask[dataframe]>=2024.1.0",
    "duckdb>=1.0.0",
    "gseapy>=1.0.0",
    "huggingface_hub>=0.20.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "openpyxl>=3.1.5",
]
```

#### 2. Create PathwayEnrichment Module
**File**: `src/l2l_bench/pathway_enrichment.py`
**Changes**: Create new module with class skeleton

```python
"""
Pathway enrichment analysis using GSEApy prerank with Reactome pathways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from l2l_bench.l2l_data import L2LData


@dataclass
class PathwayEnrichment:
    """
    Compute and cache Reactome pathway enrichment scores for treatment conditions.

    Uses GSEApy prerank method with log2FoldChange as the ranking metric.
    Results are cached to data/processed/reactome/ for efficient retrieval.
    """

    data: L2LData
    cache_dir: Path = field(default_factory=lambda: Path("data/processed/reactome"))
    gene_set_library: str = "Reactome_2022"
    min_size: int = 15
    max_size: int = 500
    permutation_num: int = 1000

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
```

#### 3. Update __init__.py
**File**: `src/l2l_bench/__init__.py`
**Changes**: Export new class

```python
from l2l_bench.l2l_data import L2LData, TreatmentCondition
from l2l_bench.pathway_enrichment import PathwayEnrichment

__all__ = ["L2LData", "TreatmentCondition", "PathwayEnrichment"]
```

### Success Criteria:

#### Automated Verification:
- [x] `uv pip install -e .` completes without errors
- [x] `python -c "from l2l_bench import PathwayEnrichment"` works
- [x] `python -c "import gseapy; print(gseapy.__version__)"` works

#### Manual Verification:
- [x] `data/processed/reactome/` directory is created (on first use)

---

## Phase 2: Build Treatment Index

### Overview
Build an index mapping (cell_line, drug, concentration) → file_number. This enables efficient single-file downloads instead of scanning all 1026 files.

### Changes Required:

#### 1. Add index building method to L2LData
**File**: `src/l2l_bench/l2l_data.py`
**Changes**: Add methods to build and use treatment index

```python
INDEX_PATH = Path("data/processed/treatment_index.parquet")

def _get_single_parquet_url(self, file_num: int) -> str:
    """Get URL for a specific parquet file."""
    base_url = f"https://huggingface.co/datasets/{self.dataset_path}/resolve/main/metadata/{self.de_config}"
    return f"{base_url}/train-{file_num:05d}-of-01026.parquet"

def build_treatment_index(self, save_path: Path | None = None) -> pd.DataFrame:
    """
    Build index mapping treatments to file numbers by streaming through data.

    This is slow (~30-60 min) but only needs to be done once.
    The index is saved to data/processed/treatment_index.parquet.

    Returns:
        DataFrame with columns: cell_line, drug, concentration, file_num
    """
    if save_path is None:
        save_path = self.INDEX_PATH

    from datasets import load_dataset

    print("Building treatment index (this takes 30-60 minutes, but only once)...")

    de_stream = load_dataset(
        self.dataset_path,
        name=self.de_config,
        split="train",
        streaming=True
    )

    # track which file we're in based on row count
    # ~4M rows per file
    ROWS_PER_FILE = 4_000_000

    treatments = []
    seen = set()

    for i, row in enumerate(de_stream):
        key = (row['Cell_Name_Vevo'], row['drug'], row['concentration'])

        if key not in seen:
            seen.add(key)
            file_num = i // ROWS_PER_FILE
            treatments.append({
                'cell_line': row['Cell_Name_Vevo'],
                'drug': row['drug'],
                'concentration': row['concentration'],
                'file_num': file_num,
                'row_start': i,
            })

            if len(treatments) % 100 == 0:
                print(f"  Found {len(treatments)} treatments...")

    index_df = pd.DataFrame(treatments)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(save_path, index=False)
    print(f"Saved treatment index with {len(index_df)} treatments to {save_path}")

    return index_df

def load_treatment_index(self) -> pd.DataFrame:
    """Load the treatment index, building it if it doesn't exist."""
    if not self.INDEX_PATH.exists():
        return self.build_treatment_index()
    return pd.read_parquet(self.INDEX_PATH)

def get_expression_data_indexed(
    self,
    drug: str,
    concentration: float,
    cell_line: str,
) -> pd.DataFrame:
    """
    Get DE data using the treatment index for efficient single-file download.

    Much faster than get_expression_data() after index is built.
    """
    index = self.load_treatment_index()

    # find the file containing this treatment
    mask = (
        (index['cell_line'] == cell_line) &
        (index['drug'] == drug) &
        (abs(index['concentration'] - concentration) < 0.001)
    )
    matches = index[mask]

    if len(matches) == 0:
        raise ValueError(f"Treatment not found: {drug}@{concentration} in {cell_line}")

    file_num = matches.iloc[0]['file_num']
    url = self._get_single_parquet_url(file_num)

    print(f"Downloading file {file_num} for {drug}@{concentration} in {cell_line}...")

    # query just this one file
    query = f"""
        SELECT * FROM read_parquet('{url}')
        WHERE drug = '{drug}'
          AND Cell_Name_Vevo = '{cell_line}'
          AND ABS(concentration - {concentration}) < 0.001
    """

    result = self._conn.execute(query).df()
    print(f"  Retrieved {len(result)} rows")
    return result

def get_available_treatments_for_cell_line(self, cell_line: str) -> pd.DataFrame:
    """Get all drug/concentration combinations for a cell line from the index."""
    index = self.load_treatment_index()
    mask = index['cell_line'] == cell_line
    return index[mask][['drug', 'concentration']].drop_duplicates()
```

### Success Criteria:

#### Automated Verification:
- [x] `build_treatment_index()` method added (actual building deferred - takes 30-60 min)
- [x] `load_treatment_index()` method added
- [x] `get_expression_data_indexed()` method added

#### Manual Verification:
- [ ] Index file size is reasonable (~1-5 MB) - will verify after build
- [ ] Single-file download is noticeably faster than multi-file query - will verify after build
- [ ] Index contains expected number of treatments (~50 cell lines × ~400 drugs × concentrations) - will verify after build

---

## Phase 3: Implement Core Enrichment Method

### Overview
Implement `compute_enrichment()` that runs GSEApy prerank on DE data.

### Changes Required:

#### 1. Add compute_enrichment method
**File**: `src/l2l_bench/pathway_enrichment.py`
**Changes**: Implement the core enrichment computation

```python
import gseapy as gp

def _get_cache_path(self, drug: str, concentration: float, cell_line: str) -> Path:
    """Get path to cached enrichment results."""
    # sanitize drug name for filesystem
    safe_drug = drug.replace("/", "_").replace(" ", "_")
    cell_dir = self.cache_dir / cell_line
    cell_dir.mkdir(parents=True, exist_ok=True)
    return cell_dir / f"{safe_drug}_{concentration}.parquet"

def _is_cached(self, drug: str, concentration: float, cell_line: str) -> bool:
    """Check if enrichment results are already cached."""
    return self._get_cache_path(drug, concentration, cell_line).exists()

def _load_cached(self, drug: str, concentration: float, cell_line: str) -> pd.DataFrame:
    """Load cached enrichment results."""
    return pd.read_parquet(self._get_cache_path(drug, concentration, cell_line))

def _save_cache(self, df: pd.DataFrame, drug: str, concentration: float, cell_line: str) -> None:
    """Save enrichment results to cache."""
    df.to_parquet(self._get_cache_path(drug, concentration, cell_line), index=False)

def compute_enrichment(
    self,
    drug: str,
    concentration: float,
    cell_line: str,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute Reactome pathway enrichment for a treatment condition.

    Uses GSEApy prerank with log2FoldChange as ranking metric.
    Results are cached to avoid recomputation.

    Args:
        drug: Drug name
        concentration: Drug concentration
        cell_line: Cell line name
        force_recompute: If True, recompute even if cached

    Returns:
        DataFrame with columns: pathway, nes, pvalue, fdr, leading_edge
    """
    # check cache first
    if not force_recompute and self._is_cached(drug, concentration, cell_line):
        print(f"Loading cached enrichment for {drug}@{concentration} in {cell_line}")
        return self._load_cached(drug, concentration, cell_line)

    # get differential expression data (uses indexed single-file download)
    de_data = self.data.get_expression_data_indexed(
        drug=drug,
        concentration=concentration,
        cell_line=cell_line
    )

    if len(de_data) == 0:
        raise ValueError(f"No expression data found for {drug}@{concentration} in {cell_line}")

    # prepare ranked gene list for prerank
    # use log2FoldChange as ranking metric
    # convert gene names to uppercase (required by Enrichr libraries)
    ranked = de_data[['gene_name', 'log2FoldChange']].dropna()
    ranked['gene_name'] = ranked['gene_name'].str.upper()
    ranked = ranked.drop_duplicates(subset='gene_name')
    ranked = ranked.set_index('gene_name')['log2FoldChange']

    print(f"Running prerank enrichment for {drug}@{concentration} in {cell_line}...")
    print(f"  {len(ranked)} genes in ranked list")

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

    # add treatment metadata
    results['drug'] = drug
    results['concentration'] = concentration
    results['cell_line'] = cell_line

    # cache results
    self._save_cache(results, drug, concentration, cell_line)
    print(f"  Cached {len(results)} pathway scores")

    return results
```

### Success Criteria:

#### Automated Verification:
- [x] `compute_enrichment()` method implemented
- [x] Cache file handling implemented at expected path

#### Manual Verification:
- [ ] Run with real data: `enrichment.compute_enrichment("Trametinib", 0.05, "A549")` (requires treatment index)
- [ ] Verify output DataFrame has expected columns
- [ ] Verify cache file exists and can be reloaded

---

## Phase 4: Implement get_pathway_score Method

### Overview
Add method to retrieve a single pathway's score, computing if not cached.

### Changes Required:

#### 1. Add get_pathway_score method
**File**: `src/l2l_bench/pathway_enrichment.py`
**Changes**: Add single-pathway retrieval method

```python
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
```

### Success Criteria:

#### Automated Verification:
- [x] `get_pathway_score()` method implemented
- [x] `list_pathways()` method implemented
- [x] Partial pathway name matching implemented

#### Manual Verification:
- [ ] `enrichment.get_pathway_score("MAPK", "Trametinib", 0.05, "A549")` returns reasonable score (requires treatment index)
- [ ] `enrichment.list_pathways(...)` returns list of pathway names (requires treatment index)

---

## Phase 5: Add Batch Processing for Cell Lines

### Overview
Add method to batch process all treatments for a cell line. Treatment discovery uses the index built in Phase 2.

### Changes Required:

#### 1. Add batch processing method to PathwayEnrichment
**File**: `src/l2l_bench/pathway_enrichment.py`
**Changes**: Add method to process entire cell line with actual concentrations

```python
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
```

### Success Criteria:

#### Automated Verification:
- [x] `compute_enrichment_for_cell_line()` method implemented
- [x] Cached results skip logic implemented
- [x] Concentration handling implemented in cache paths

#### Manual Verification:
- [ ] Run on one cell line with 2-3 drugs, verify all files cached (requires treatment index)
- [ ] Verify cache file names include actual concentration values
- [ ] Verify combined DataFrame contains results from multiple treatments

---

## Phase 6: Integration and Documentation

### Overview
Update exports, add docstrings, and create usage notebook.

### Changes Required:

#### 1. Create example notebook
**File**: `notebooks/03_pathway_enrichment.ipynb`
**Changes**: Create notebook demonstrating usage

```python
# Cell 1 - Setup
from l2l_bench import L2LData, PathwayEnrichment

data = L2LData()
enrichment = PathwayEnrichment(data)

# Cell 2 - Compute enrichment for one treatment
results = enrichment.compute_enrichment(
    drug="Trametinib",
    concentration=0.05,
    cell_line="A549"
)
print(f"Computed {len(results)} pathway scores")
results.head(10)

# Cell 3 - Get specific pathway score
mapk_score = enrichment.get_pathway_score(
    pathway="MAPK",
    drug="Trametinib",
    concentration=0.05,
    cell_line="A549"
)
print(f"MAPK pathway NES: {mapk_score['nes']:.3f} (FDR: {mapk_score['fdr']:.3e})")

# Cell 4 - List available pathways
pathways = enrichment.list_pathways("Trametinib", 0.05, "A549")
print(f"Available pathways: {len(pathways)}")

# Cell 5 - Filter for significant pathways
sig_pathways = results[results['fdr'] < 0.25].sort_values('nes')
print("Top upregulated pathways:")
print(sig_pathways[sig_pathways['nes'] > 0].tail(5))
print("\nTop downregulated pathways:")
print(sig_pathways[sig_pathways['nes'] < 0].head(5))
```

### Success Criteria:

#### Automated Verification:
- [x] Notebook created at `notebooks/03_pathway_enrichment.ipynb`
- [ ] Notebook executes without errors: `jupyter nbconvert --execute` (requires treatment index)

#### Manual Verification:
- [ ] Results make biological sense (e.g., MEK inhibitor downregulates MAPK pathway) (requires treatment index)
- [ ] Cache files are organized as expected in `data/processed/reactome/`

---

## Testing Strategy

### Unit Tests:
- Test `_get_cache_path()` produces valid paths
- Test `_is_cached()` correctly detects cached files
- Test `compute_enrichment()` with mock DE data
- Test `get_pathway_score()` partial matching logic

### Integration Tests:
- Test full pipeline: download → enrich → cache → retrieve
- Test batch processing with 2-3 drugs

### Manual Testing Steps:
1. Run enrichment for Trametinib/A549, verify MAPK pathway is downregulated
2. Re-run same query, verify it uses cache (faster)
3. Run for different drug, verify new cache file created
4. Run `list_pathways()`, verify reasonable list returned

## Performance Considerations

- GSEApy prerank with 1000 permutations takes ~10-30 seconds per treatment
- Caching is essential for interactive use
- Batch processing by cell line reduces repeated downloads
- Consider reducing `permutation_num` to 100 for faster iteration during development

## File Organization

After implementation:
```
data/processed/reactome/
├── A549/
│   ├── Trametinib_0.05.parquet
│   ├── Trametinib_0.5.parquet      # same drug, different concentration
│   ├── Cobimetinib_0.05.parquet
│   └── ...
├── SW620/
│   ├── Trametinib_0.05.parquet
│   └── ...
└── ...
```

Each parquet file contains columns:
- `pathway`: Reactome pathway name
- `nes`: Normalized Enrichment Score
- `pvalue`: Nominal p-value
- `fdr`: FDR q-value
- `leading_edge`: Genes driving the enrichment
- `drug`, `concentration`, `cell_line`: Treatment metadata

## Dependencies

- gseapy >= 1.0.0 (for prerank)
- Reactome_2022 gene set library (downloaded automatically by gseapy)

## References

- GSEApy documentation: https://gseapy.readthedocs.io/en/latest/
- Current L2LData implementation: `src/l2l_bench/l2l_data.py`
