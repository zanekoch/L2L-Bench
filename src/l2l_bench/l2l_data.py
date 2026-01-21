"""
L2L Data: Efficient access to Tahoe-100M and similar perturbation datasets.

Uses DuckDB for efficient predicate pushdown queries against remote parquet files,
avoiding the need to download the full 50GB+ dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from datasets import load_dataset # huggingface datasets library


def _get_root_dir() -> Path:
    """Get the project root directory from L2L_ROOT_DIR env var or default to cwd."""
    root = os.environ.get("L2L_ROOT_DIR")
    if root:
        return Path(root)
    return Path.cwd()


INDEX_PATH = _get_root_dir() / "data/processed/treatment_index.parquet"


@dataclass
class TreatmentCondition:
    """Represents a specific treatment condition (drug + concentration + cell line)."""

    drug: str
    concentration: float
    cell_line: str
    concentration_unit: str = "uM"

    def __str__(self) -> str:
        return f"{self.drug}_{self.concentration}{self.concentration_unit}_{self.cell_line}"


@dataclass
class L2LData:
    """
    Efficient access to Tahoe-100M pseudobulk differential expression data.

    Uses DuckDB to query remote parquet files with predicate pushdown,
    meaning only matching rows are downloaded (not the full 50GB dataset).

    This class provides:
    - Automatic loading of metadata tables (genes, drugs, cell lines, samples)
    - Efficient filtered queries to differential expression data via DuckDB
    - Filtering by drug, concentration, cell line, and genes

    Example:
        >>> data = L2LData()
        >>> # get all DE data for a specific treatment (fast! uses predicate pushdown)
        >>> de = data.get_expression_data(drug="Trametinib", concentration=0.05, cell_line="A549")
        >>> # access metadata
        >>> data.drug_metadata[data.drug_metadata['moa-fine'] == 'MEK inhibitor']
    """

    dataset_path: str = "tahoebio/Tahoe-100M"
    de_config: str = "pseudobulk_differential_expression"

    # metadata tables (loaded lazily)
    _gene_metadata: pd.DataFrame | None = field(default=None, repr=False)
    _drug_metadata: pd.DataFrame | None = field(default=None, repr=False)
    _cell_line_metadata: pd.DataFrame | None = field(default=None, repr=False)
    _sample_metadata: pd.DataFrame | None = field(default=None, repr=False)

    # cached lookups
    _valid_drugs: set[str] | None = field(default=None, repr=False)
    _valid_cell_lines: set[str] | None = field(default=None, repr=False)

    # duckdb connection
    _conn: duckdb.DuckDBPyConnection | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize by loading metadata tables and setting up DuckDB."""
        self._setup_duckdb()
        self._load_metadata()

    def _setup_duckdb(self) -> None:
        """Initialize DuckDB with httpfs for remote parquet access."""
        self._conn = duckdb.connect()
        self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        # set a reasonable timeout and retry settings
        self._conn.execute("SET http_timeout=30000;")  # 30 seconds

        # configure HuggingFace authentication if token available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            # use DuckDB's Secrets Manager for HTTP authentication
            self._conn.execute(f"""
                CREATE SECRET hf_auth (
                    TYPE http,
                    BEARER_TOKEN '{hf_token}'
                );
            """)
            print("HuggingFace authentication configured")

    def _get_parquet_urls(self) -> list[str]:
        """Get list of parquet file URLs for the pseudobulk DE data."""
        # there are 1026 parquet files for the DE data
        base_url = f"https://huggingface.co/datasets/{self.dataset_path}/resolve/main/metadata/{self.de_config}"
        n_files = 1026
        return [
            f"{base_url}/train-{i:05d}-of-{n_files:05d}.parquet"
            for i in range(n_files)
        ]

    def _get_parquet_url_sql(self) -> str:
        """Get SQL expression for reading all parquet files."""
        urls = self._get_parquet_urls()
        # DuckDB can read a list of parquet files
        url_list = ", ".join(f"'{url}'" for url in urls)
        return f"read_parquet([{url_list}])"

    def _load_metadata(self) -> None:
        """Load all metadata tables from HuggingFace."""
        print(f"Loading metadata from {self.dataset_path}...")

        # gene metadata
        self._gene_metadata = load_dataset(
            self.dataset_path, name="gene_metadata", split="train"
        ).to_pandas()
        print(f"  Gene metadata: {len(self._gene_metadata)} genes")

        # drug metadata
        self._drug_metadata = load_dataset(
            self.dataset_path, name="drug_metadata", split="train"
        ).to_pandas()
        print(f"  Drug metadata: {len(self._drug_metadata)} drugs")

        # cell line metadata
        self._cell_line_metadata = load_dataset(
            self.dataset_path, name="cell_line_metadata", split="train"
        ).to_pandas()
        print(f"  Cell line metadata: {self._cell_line_metadata['cell_name'].nunique()} unique cell lines")

        # sample metadata
        self._sample_metadata = load_dataset(
            self.dataset_path, name="sample_metadata", split="train"
        ).to_pandas()
        print(f"  Sample metadata: {len(self._sample_metadata)} samples")

        # build lookup caches
        self._valid_drugs = set[str](self._drug_metadata['drug'].unique())
        self._valid_cell_lines = set[str](self._cell_line_metadata['cell_name'].unique())

        print("Metadata loaded successfully.")

    @property
    def gene_metadata(self) -> pd.DataFrame:
        """Gene metadata: symbol, ensembl_id, token_id."""
        if self._gene_metadata is None:
            self._load_metadata()
        return self._gene_metadata

    @property
    def drug_metadata(self) -> pd.DataFrame:
        """Drug metadata: MOA, targets, SMILES, etc."""
        if self._drug_metadata is None:
            self._load_metadata()
        return self._drug_metadata

    @property
    def cell_line_metadata(self) -> pd.DataFrame:
        """Cell line metadata: driver mutations, organ, etc."""
        if self._cell_line_metadata is None:
            self._load_metadata()
        return self._cell_line_metadata

    @property
    def sample_metadata(self) -> pd.DataFrame:
        """Sample metadata: concentrations, QC metrics."""
        if self._sample_metadata is None:
            self._load_metadata()
        return self._sample_metadata

    def get_expression_data(
        self,
        drug: str | None = None,
        concentration: float | None = None,
        cell_line: str | None = None,
        genes: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Get differential expression data with efficient filtering via DuckDB.

        Uses predicate pushdown to only download matching rows from the remote
        parquet files. This is O(result_size) not O(dataset_size).

        Args:
            drug: Drug name to filter by (e.g., "Trametinib"). Optional.
            concentration: Drug concentration to filter by (e.g., 0.05). Optional.
            cell_line: Cell line name to filter by (e.g., "A549"). Optional.
            genes: List of gene symbols to filter to. Optional.
            limit: Maximum number of rows to return. Optional.

        Returns:
            DataFrame with differential expression data for matching rows.
            Columns include: gene_name, log2FoldChange, padj, baseMean, etc.

        Raises:
            ValueError: If drug or cell_line not found in metadata.

        Example:
            >>> data = L2LData()
            >>> # get all genes for one treatment condition
            >>> de = data.get_expression_data(
            ...     drug="Trametinib",
            ...     concentration=0.05,
            ...     cell_line="A549"
            ... )
            >>> print(f"Got {len(de)} genes")

            >>> # get specific genes across all conditions
            >>> de = data.get_expression_data(genes=["KRAS", "BRAF", "MKI67"])

            >>> # get all data for a drug (across all cell lines)
            >>> de = data.get_expression_data(drug="Trametinib", limit=10000)
        """
        # validate inputs
        if drug is not None and drug not in self._valid_drugs:
            similar = [d for d in self._valid_drugs if drug.lower() in d.lower()]
            raise ValueError(
                f"Drug '{drug}' not found. "
                f"Similar drugs: {similar[:5] if similar else 'none'}"
            )

        if cell_line is not None and cell_line not in self._valid_cell_lines:
            similar = [c for c in self._valid_cell_lines if cell_line.lower() in c.lower()]
            raise ValueError(
                f"Cell line '{cell_line}' not found. "
                f"Similar: {similar[:5] if similar else 'none'}"
            )

        # build WHERE clause
        conditions = []
        if drug is not None:
            conditions.append(f"drug = '{drug}'")
        if cell_line is not None:
            conditions.append(f"Cell_Name_Vevo = '{cell_line}'")
        if concentration is not None:
            # use a small tolerance for float comparison
            conditions.append(f"ABS(concentration - {concentration}) < 0.001")
        if genes is not None:
            gene_list = ", ".join(f"'{g}'" for g in genes)
            conditions.append(f"gene_name IN ({gene_list})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""

        # build query using read_parquet with list of URLs
        parquet_source = self._get_parquet_url_sql()
        query = f"""
            SELECT *
            FROM {parquet_source}
            WHERE {where_clause}
            {limit_clause}
        """

        # log what we're doing
        filter_desc = []
        if drug:
            filter_desc.append(f"drug={drug}")
        if concentration:
            filter_desc.append(f"conc={concentration}")
        if cell_line:
            filter_desc.append(f"cell_line={cell_line}")
        if genes:
            filter_desc.append(f"genes=[{len(genes)} genes]")
        filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
        print(f"Querying DE data ({filter_str})...")

        # execute query
        result = self._conn.execute(query).df()
        print(f"  Retrieved {len(result)} rows")

        return result

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a custom SQL query against the DE data.

        The parquet files are available as a table - use {parquet} as placeholder.

        Args:
            sql: SQL query with {parquet} placeholder for the data source.

        Returns:
            Query result as DataFrame.

        Example:
            >>> data = L2LData()
            >>> result = data.query('''
            ...     SELECT drug, Cell_Name_Vevo, AVG(log2FoldChange) as mean_lfc
            ...     FROM {parquet}
            ...     WHERE gene_name = 'MKI67'
            ...     GROUP BY drug, Cell_Name_Vevo
            ...     LIMIT 100
            ... ''')
        """
        parquet_source = self._get_parquet_url_sql()
        query = sql.replace("{parquet}", parquet_source)
        return self._conn.execute(query).df()

    def get_available_treatments(self, cell_line: str | None = None) -> pd.DataFrame:
        """
        Get a summary of available treatment conditions.

        Args:
            cell_line: Optional cell line to filter to.

        Returns:
            DataFrame with drug, cell_line combinations available.
        """
        df = self.sample_metadata[['sample', 'drug', 'drugname_drugconc']].copy()
        return df

    def get_drugs_by_moa(self, moa: str, moa_type: str = "fine") -> list[str]:
        """
        Get list of drugs with a specific mechanism of action.

        Args:
            moa: Mechanism of action to search for (partial match)
            moa_type: "fine" for moa-fine or "broad" for moa-broad

        Returns:
            List of drug names matching the MOA.

        Example:
            >>> data = L2LData()
            >>> mek_inhibitors = data.get_drugs_by_moa("MEK inhibitor")
            >>> print(mek_inhibitors)
            ['Trametinib', 'Cobimetinib', 'Binimetinib', ...]
        """
        col = f"moa-{moa_type}"
        mask = self.drug_metadata[col].str.contains(moa, case=False, na=False)
        return self.drug_metadata[mask]['drug'].tolist()

    def get_cell_lines_by_mutation(self, gene: str) -> pd.DataFrame:
        """
        Get cell lines with mutations in a specific gene.

        Args:
            gene: Gene symbol to search for (e.g., "KRAS", "BRAF")

        Returns:
            DataFrame with cell lines and their mutations in that gene.

        Example:
            >>> data = L2LData()
            >>> kras_mutants = data.get_cell_lines_by_mutation("KRAS")
            >>> print(kras_mutants[['cell_name', 'Driver_ProtEffect_or_CdnaEffect']])
        """
        mask = self.cell_line_metadata['Driver_Gene_Symbol'] == gene
        return self.cell_line_metadata[mask][
            ['cell_name', 'Organ', 'Driver_Gene_Symbol',
             'Driver_ProtEffect_or_CdnaEffect', 'Driver_Mech_InferDM']
        ].drop_duplicates()

    def list_cell_lines(self) -> list[str]:
        """Get list of all available cell line names."""
        return sorted(self._valid_cell_lines)

    def list_drugs(self) -> list[str]:
        """Get list of all available drug names."""
        return sorted(self._valid_drugs)

    def _get_single_parquet_url(self, file_num: int) -> str:
        """Get URL for a specific parquet file."""
        base_url = f"https://huggingface.co/datasets/{self.dataset_path}/resolve/main/metadata/{self.de_config}"
        return f"{base_url}/train-{file_num:05d}-of-01026.parquet"

    def build_treatment_index(
        self,
        save_path: Path | None = None,
        compute_pathways: bool = False,
    ) -> pd.DataFrame:
        """
        Build index mapping treatments to file numbers using DuckDB.

        Queries SELECT DISTINCT from each parquet file individually.
        Takes ~3-6 hours (10-20 sec per file × 1026 files).

        When compute_pathways=True, also computes GSEA pathway enrichment for each
        treatment while processing each file. This avoids a second pass through all
        files but significantly increases processing time per file (~10-30 min/file).

        Writes incrementally to disk and can resume from where it left off.
        The index is saved to data/processed/treatment_index.parquet.

        Args:
            save_path: Path to save the index. Defaults to INDEX_PATH.
            compute_pathways: If True, compute GSEA pathway enrichment for each
                treatment while processing. Results are cached to
                data/processed/reactome/ and already-cached treatments are skipped.

        Returns:
            DataFrame with columns: cell_line, drug, concentration, file_num
        """
        import time

        if save_path is None:
            save_path = INDEX_PATH

        n_files = 1026

        # initialize PathwayEnrichment if computing pathways
        pe = None
        if compute_pathways:
            from l2l_bench.pathway_enrichment import PathwayEnrichment
            pe = PathwayEnrichment(self)
            print("Pathway enrichment enabled - will compute GSEA for each treatment")

        # check for existing partial index to resume from
        start_file = 0
        all_treatments = []
        if save_path.exists():
            existing_df = pd.read_parquet(save_path)
            if 'file_num' in existing_df.columns:
                start_file = existing_df['file_num'].max() + 1
                all_treatments.append(existing_df)
                print(f"Resuming from file {start_file} (found {len(existing_df)} existing treatments)")

        if start_file >= n_files:
            print("Index already complete!")
            return existing_df

        print(f"Building treatment index via DuckDB ({n_files} files)...")
        if start_file == 0 and not compute_pathways:
            print("Estimated time: 3-6 hours (10-20 sec per file)")

        start_time = time.time()
        files_processed = 0
        pathways_computed = 0
        pathways_skipped = 0

        for file_num in range(start_file, n_files):
            file_start = time.time()
            url = self._get_single_parquet_url(file_num)

            try:
                # compute pathways if enabled
                file_pathways_computed = 0
                file_pathways_skipped = 0

                if compute_pathways and pe is not None:
                    # load only columns needed for index + GSEA
                    # index: Cell_Name_Vevo, drug, concentration
                    # GSEA: gene_name, log2FoldChange, padj
                    query = f"""
                        SELECT Cell_Name_Vevo, drug, concentration,
                               gene_name, log2FoldChange, padj
                        FROM read_parquet('{url}')
                    """
                    full_de = self._conn.execute(query).df()

                    # extract distinct treatments from loaded data (no second download)
                    df = full_de[['Cell_Name_Vevo', 'drug', 'concentration']].drop_duplicates()
                    df = df.rename(columns={'Cell_Name_Vevo': 'cell_line'})
                    df['file_num'] = file_num
                    all_treatments.append(df)
                    files_processed += 1

                    # process each treatment in the file
                    for (cell_line, drug, conc), treatment_df in full_de.groupby(
                        ['Cell_Name_Vevo', 'drug', 'concentration']
                    ):
                        # skip if already cached
                        if pe._is_cached(drug, conc, cell_line):
                            file_pathways_skipped += 1
                            continue

                        try:
                            pe.compute_enrichment_from_df(
                                de_data=treatment_df,
                                drug=drug,
                                concentration=conc,
                                cell_line=cell_line,
                                verbose=False,  # reduce noise during batch processing
                            )
                            file_pathways_computed += 1
                        except Exception as e:
                            print(f"    Warning: Failed pathway for {drug}@{conc} in {cell_line}: {e}")

                    pathways_computed += file_pathways_computed
                    pathways_skipped += file_pathways_skipped
                else:
                    # index-only mode: just get distinct treatments
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
                    files_processed += 1

                file_elapsed = time.time() - file_start
                total_elapsed = time.time() - start_time
                avg_per_file = total_elapsed / files_processed
                remaining_files = n_files - file_num - 1
                eta_sec = avg_per_file * remaining_files

                # build progress message
                progress_msg = (
                    f"  File {file_num + 1:4d}/{n_files} | "
                    f"{len(df):5d} treatments | "
                    f"{file_elapsed:5.1f}s | "
                    f"ETA: {eta_sec/3600:.1f}h"
                )
                if compute_pathways:
                    progress_msg += f" | pathways: {file_pathways_computed} new, {file_pathways_skipped} cached"
                print(progress_msg)

                # save incrementally after every file
                self._save_treatment_index(all_treatments, save_path)

            except Exception as e:
                print(f"  File {file_num + 1:4d}/{n_files} | ERROR: {e}")
                # save what we have before continuing
                if all_treatments:
                    self._save_treatment_index(all_treatments, save_path)
                continue

        # final save
        index_df = self._save_treatment_index(all_treatments, save_path)

        total_elapsed = time.time() - start_time
        print(f"\nDone! Processed {files_processed} files in {total_elapsed/3600:.1f} hours")
        print(f"Saved {len(index_df)} unique treatments to {save_path}")
        if compute_pathways:
            print(f"Pathways: {pathways_computed} computed, {pathways_skipped} already cached")

        return index_df

    def _save_treatment_index(self, all_treatments: list[pd.DataFrame], save_path: Path) -> pd.DataFrame:
        """Combine and save treatment index, deduplicating entries."""
        index_df = pd.concat(all_treatments, ignore_index=True)

        # deduplicate - same treatment may appear in multiple files, keep first
        index_df = index_df.drop_duplicates(
            subset=['cell_line', 'drug', 'concentration'],
            keep='first'
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_parquet(save_path, index=False)
        return index_df

    def load_treatment_index(self) -> pd.DataFrame:
        """Load the treatment index, building it if it doesn't exist."""
        if not INDEX_PATH.exists():
            return self.build_treatment_index()
        return pd.read_parquet(INDEX_PATH)

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

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
