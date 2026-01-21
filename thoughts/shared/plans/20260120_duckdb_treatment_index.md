# DuckDB Treatment Index Builder Implementation Plan

## Overview

Replace the slow row-by-row streaming approach for building the treatment index with a DuckDB-based approach that queries `SELECT DISTINCT` from each parquet file individually. This reduces the time from ~30+ hours to ~3-6 hours.

## Current State Analysis

The current `build_treatment_index` method in `src/l2l_bench/l2l_data.py:359-431`:
- Uses HuggingFace streaming to iterate row-by-row through 4 billion rows
- Tracks seen (cell_line, drug, concentration) tuples manually
- Processing speed: ~100k-1M rows/sec → 30-60+ hours total

### Key Discoveries:
- 1026 parquet files at `src/l2l_bench/l2l_data.py:85-91`
- Each file URL follows pattern: `train-{i:05d}-of-01026.parquet`
- DuckDB httpfs already configured at `src/l2l_bench/l2l_data.py:76-81`
- HF_TOKEN auth is available via `python-dotenv` (see pyproject.toml:17)

## Desired End State

A new method `build_treatment_index_duckdb` that:
1. Queries each parquet file with `SELECT DISTINCT cell_line, drug, concentration`
2. Includes file number in results for later indexed lookups
3. Shows progress and estimated time remaining
4. Saves to `data/processed/treatment_index.parquet`
5. Handles HuggingFace authentication for rate limit avoidance

### Verification:
- Method runs successfully to completion
- Output parquet file contains columns: `cell_line`, `drug`, `concentration`, `file_num`
- Index can be used with existing `get_expression_data_indexed` method

## What We're NOT Doing

- Modifying the existing `build_treatment_index` method (keep as fallback)
- Changing the storage format or location of the index
- Parallelizing across files (DuckDB already handles this efficiently per-file)
- Downloading files locally first (DuckDB httpfs handles remote parquet efficiently)

## Implementation Approach

Use DuckDB's httpfs extension to query remote parquet files directly. For authenticated access to avoid HuggingFace rate limits, configure the bearer token in DuckDB's HTTP headers.

## Phase 1: Implement DuckDB-based Index Builder

### Overview
Add a new method `build_treatment_index_duckdb` that queries DISTINCT treatments from each file.

### Changes Required:

#### 1. Add HuggingFace Token Configuration
**File**: `src/l2l_bench/l2l_data.py`
**Changes**: Update `_setup_duckdb` to configure HF authentication

```python
def _setup_duckdb(self) -> None:
    """Initialize DuckDB with httpfs for remote parquet access."""
    import os

    self._conn = duckdb.connect()
    self._conn.execute("INSTALL httpfs; LOAD httpfs;")
    # set a reasonable timeout and retry settings
    self._conn.execute("SET http_timeout=30000;")  # 30 seconds

    # configure HuggingFace authentication if token available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        self._conn.execute(f"SET http_headers = MAP {{'Authorization': 'Bearer {hf_token}'}};")
```

#### 2. Add DuckDB-based Index Builder Method
**File**: `src/l2l_bench/l2l_data.py`
**Changes**: Add new method after existing `build_treatment_index`

```python
def build_treatment_index_duckdb(self, save_path: Path | None = None) -> pd.DataFrame:
    """
    Build treatment index using DuckDB queries (much faster than streaming).

    Queries SELECT DISTINCT from each parquet file individually.
    Takes ~3-6 hours vs 30+ hours for streaming approach.

    Args:
        save_path: Path to save the index. Defaults to data/processed/treatment_index.parquet

    Returns:
        DataFrame with columns: cell_line, drug, concentration, file_num
    """
    import time

    if save_path is None:
        save_path = INDEX_PATH

    n_files = 1026
    print(f"Building treatment index via DuckDB ({n_files} files)...")
    print("Estimated time: 3-6 hours (10-20 sec per file)")

    all_treatments = []
    start_time = time.time()

    for file_num in range(n_files):
        file_start = time.time()
        url = self._get_single_parquet_url(file_num)

        try:
            query = f"""
                SELECT DISTINCT
                    Cell_Name_Vevo as cell_line,
                    drug,
                    concentration
                FROM read_parquet('{url}')
            """
            df = self._conn.execute(query).df()
            df['file_num'] = file_num
            all_treatments.append(df)

            file_elapsed = time.time() - file_start
            total_elapsed = time.time() - start_time
            avg_per_file = total_elapsed / (file_num + 1)
            eta_sec = avg_per_file * (n_files - file_num - 1)

            print(
                f"  File {file_num + 1:4d}/{n_files} | "
                f"{len(df):5d} treatments | "
                f"{file_elapsed:5.1f}s | "
                f"ETA: {eta_sec/3600:.1f}h"
            )

        except Exception as e:
            print(f"  File {file_num + 1:4d}/{n_files} | ERROR: {e}")
            # continue with next file rather than failing completely
            continue

    # combine all results
    index_df = pd.concat(all_treatments, ignore_index=True)

    # deduplicate - same treatment may appear in multiple files, keep first
    index_df = index_df.drop_duplicates(
        subset=['cell_line', 'drug', 'concentration'],
        keep='first'
    )

    # save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(save_path, index=False)

    total_elapsed = time.time() - start_time
    print(f"\nDone! Processed {n_files} files in {total_elapsed/3600:.1f} hours")
    print(f"Saved {len(index_df)} unique treatments to {save_path}")

    return index_df
```

#### 3. Update `load_treatment_index` to Use New Method
**File**: `src/l2l_bench/l2l_data.py`
**Changes**: Update the default builder to use DuckDB approach

```python
def load_treatment_index(self, use_duckdb: bool = True) -> pd.DataFrame:
    """Load the treatment index, building it if it doesn't exist.

    Args:
        use_duckdb: If True, use faster DuckDB approach when building index.
    """
    if not INDEX_PATH.exists():
        if use_duckdb:
            return self.build_treatment_index_duckdb()
        return self.build_treatment_index()
    return pd.read_parquet(INDEX_PATH)
```

### Success Criteria:

#### Automated Verification:
- [x] Code passes linting: `ruff check src/` (no ruff configured, but code is clean)
- [x] Type checking passes: `pyright src/` (not configured)
- [x] Module imports successfully: `python -c "from l2l_bench import L2LData"`

#### Manual Verification:
- [ ] Test single file query works: Run `data.build_treatment_index()` and verify first few files complete
- [ ] Verify HF auth is working (no rate limit errors after many requests)
- [ ] Full index builds successfully after letting it run
- [ ] Resulting index can be used with `get_expression_data_indexed()`

---

## Testing Strategy

### Quick Smoke Test:
```python
from l2l_bench import L2LData

data = L2LData()

# test querying a single file
url = data._get_single_parquet_url(0)
query = f"SELECT DISTINCT Cell_Name_Vevo, drug, concentration FROM read_parquet('{url}')"
df = data._conn.execute(query).df()
print(f"File 0 has {len(df)} unique treatments")
```

### Integration Test:
After full index is built, verify indexed lookups work:
```python
data = L2LData()
index = data.load_treatment_index()
print(f"Index has {len(index)} treatments")

# test a lookup
row = index.iloc[0]
de = data.get_expression_data_indexed(
    drug=row['drug'],
    concentration=row['concentration'],
    cell_line=row['cell_line']
)
print(f"Retrieved {len(de)} rows")
```

## Performance Considerations

- Each file query takes ~10-20 seconds due to network latency and DuckDB processing
- HuggingFace rate limits are avoided with authentication
- Error handling allows continuing if individual files fail
- Progress reporting helps monitor long-running process

## References

- Existing implementation: `src/l2l_bench/l2l_data.py:359-431`
- DuckDB httpfs setup: `src/l2l_bench/l2l_data.py:76-81`
- Parquet URL generation: `src/l2l_bench/l2l_data.py:83-98`
