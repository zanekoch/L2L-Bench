# Single-Pass Pathway Calculation Plan

## Goal

Modify `build_treatment_index()` to compute full GSEA pathway enrichment for each treatment while processing each parquet file, avoiding a second pass through all 1026 files.

## Key Context

- Treatments are self-contained within files (confirmed via index analysis)
- Current index build: 10-20 sec/file → 3-6 hours total
- Full GSEA: 10-30 sec/treatment × ~63 treatments/file = 10-30 min/file → 170-500 hours total
- This is a long-running job but avoids re-downloading files

## Design

Add `compute_pathways: bool = False` flag to `build_treatment_index()`. When enabled:
1. Load full DE data for each file (not just DISTINCT)
2. For each treatment in the file, run full GSEA prerank
3. Cache results using existing PathwayEnrichment cache structure
4. Skip already-cached treatments (enables resume)

---

## Phase 1: Add compute_enrichment_from_df() to PathwayEnrichment

**File:** `src/l2l_bench/pathway_enrichment.py`

Refactor `compute_enrichment()` to extract core GSEA logic into a new method that accepts a pre-loaded DataFrame:

```python
def compute_enrichment_from_df(
    self,
    de_data: pd.DataFrame,
    drug: str,
    concentration: float,
    cell_line: str,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute full GSEA from pre-loaded DataFrame (avoids re-download).
    """
    # check cache
    if not force_recompute and self._is_cached(drug, concentration, cell_line):
        return self._load_cached(drug, concentration, cell_line)

    # ... existing GSEA logic from compute_enrichment() ...
```

Then update `compute_enrichment()` to call this method after fetching data.

### Changes:
- [x] Add `compute_enrichment_from_df()` method
- [x] Refactor `compute_enrichment()` to use it
- [x] Update cache_dir to use `_get_root_dir()` pattern (already done)

---

## Phase 2: Extend build_treatment_index() in L2LData

**File:** `src/l2l_bench/l2l_data.py`

Update signature:
```python
def build_treatment_index(
    self,
    save_path: Path | None = None,
    compute_pathways: bool = False,
) -> pd.DataFrame:
```

Add pathway computation after processing each file:
```python
if compute_pathways:
    from l2l_bench.pathway_enrichment import PathwayEnrichment
    pe = PathwayEnrichment(self)

    # load full DE data for this file
    full_de = self._conn.execute(f"SELECT * FROM read_parquet('{url}')").df()

    # process each treatment
    for (cell_line, drug, conc), treatment_df in full_de.groupby([...]):
        if pe._is_cached(drug, conc, cell_line):
            continue
        pe.compute_enrichment_from_df(treatment_df, drug, conc, cell_line)
```

### Changes:
- [x] Add `compute_pathways` parameter
- [x] Add pathway computation loop inside main file loop
- [x] Update progress reporting to show pathway status

---

## Phase 3: Update PathwayEnrichment cache_dir

Use `L2L_ROOT_DIR` env var for cache directory (same pattern as INDEX_PATH).

**File:** `src/l2l_bench/pathway_enrichment.py`

```python
from l2l_bench.l2l_data import _get_root_dir

@dataclass
class PathwayEnrichment:
    data: L2LData
    cache_dir: Path = field(default_factory=lambda: _get_root_dir() / "data/processed/reactome")
```

---

## Verification

1. Test `compute_enrichment_from_df()` works:
   ```python
   de = data.get_expression_data_indexed("Trametinib", 0.05, "A549")
   result = pe.compute_enrichment_from_df(de, "Trametinib", 0.05, "A549")
   ```

2. Test single-pass build (let run for a few files):
   ```python
   data.build_treatment_index(compute_pathways=True)
   # check data/processed/reactome/ for cached results
   ```

3. Verify resume skips cached treatments

---

## Time Estimate

- ~63 treatments per file × 20 sec GSEA = ~21 min per file
- 1026 files × 21 min = ~360 hours (15 days)
- With resume capability, can run in chunks
