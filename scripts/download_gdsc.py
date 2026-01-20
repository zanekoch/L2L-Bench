#!/usr/bin/env python3
"""
Download GDSC (Genomics of Drug Sensitivity in Cancer) data
and check overlaps with Tahoe-100M cell lines and drugs.
"""

import argparse
from pathlib import Path
import urllib.request

try:
    import pandas as pd
except ImportError:
    print("pandas required. Run: uv add pandas openpyxl")
    exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"
GDSC_DIR = DATA_DIR / "external" / "gdsc"

# GDSC release 8.5 URLs
GDSC_URLS = {
    "gdsc2_fitted": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
    "gdsc1_fitted": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
    "cell_lines": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
    "compounds": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv",
}


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file if it doesn't exist."""
    if dest.exists() and not force:
        print(f"  Already exists: {dest.name}")
        return dest

    print(f"  Downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def download_gdsc(force: bool = False):
    """Download GDSC data files."""
    print("Downloading GDSC data...")

    paths = {}
    for name, url in GDSC_URLS.items():
        suffix = ".xlsx" if "xlsx" in url else ".csv"
        dest = GDSC_DIR / f"{name}{suffix}"
        paths[name] = download_file(url, dest, force)

    return paths


def analyze_overlaps():
    """Analyze overlaps between GDSC and Tahoe-100M."""
    print("\n" + "=" * 60)
    print("ANALYZING GDSC ↔ TAHOE-100M OVERLAPS")
    print("=" * 60)

    # load Tahoe-100M metadata
    tahoe_cells = pd.read_parquet(DATA_DIR / "processed" / "cell_line_metadata.parquet")
    tahoe_drugs = pd.read_parquet(DATA_DIR / "processed" / "drug_metadata.parquet")

    tahoe_cell_names = set(tahoe_cells['cell_name'].str.upper().str.replace('-', '').str.replace(' ', ''))
    tahoe_drug_names = set(tahoe_drugs['drug'].str.upper())

    print(f"\nTahoe-100M: {len(tahoe_cells['cell_name'].unique())} cell lines, {len(tahoe_drugs)} drugs")

    # load GDSC data
    gdsc_cells = pd.read_excel(GDSC_DIR / "cell_lines.xlsx")
    gdsc_compounds = pd.read_csv(GDSC_DIR / "compounds.csv")
    gdsc2 = pd.read_excel(GDSC_DIR / "gdsc2_fitted.xlsx")

    # filter to valid string cell names
    gdsc_cells = gdsc_cells[gdsc_cells['Sample Name'].apply(lambda x: isinstance(x, str))]
    gdsc_cell_names = set(gdsc_cells['Sample Name'].str.upper().str.replace('-', '').str.replace(' ', ''))
    gdsc_drug_names = set(gdsc_compounds['DRUG_NAME'].str.upper())

    print(f"GDSC: {len(gdsc_cells)} cell lines, {len(gdsc_compounds)} drugs")

    # find overlaps
    cell_overlap = tahoe_cell_names & gdsc_cell_names
    drug_overlap = tahoe_drug_names & gdsc_drug_names

    print(f"\n--- CELL LINE OVERLAP ---")
    print(f"Overlapping cell lines: {len(cell_overlap)}")

    # show actual matches
    tahoe_cells_lookup = {c.upper().replace('-', '').replace(' ', ''): c
                          for c in tahoe_cells['cell_name'].unique()}
    gdsc_cells_lookup = {str(c).upper().replace('-', '').replace(' ', ''): str(c)
                         for c in gdsc_cells['Sample Name']}

    print("\nMatched cell lines (first 20):")
    for i, norm_name in enumerate(sorted(cell_overlap)[:20]):
        tahoe_name = tahoe_cells_lookup.get(norm_name, '?')
        gdsc_name = gdsc_cells_lookup.get(norm_name, '?')
        print(f"  {tahoe_name:20} ↔ {gdsc_name}")
    if len(cell_overlap) > 20:
        print(f"  ... and {len(cell_overlap) - 20} more")

    print(f"\n--- DRUG OVERLAP ---")
    print(f"Overlapping drugs: {len(drug_overlap)}")

    tahoe_drugs_lookup = {d.upper(): d for d in tahoe_drugs['drug']}
    gdsc_drugs_lookup = {d.upper(): d for d in gdsc_compounds['DRUG_NAME']}

    print("\nMatched drugs (first 30):")
    for i, norm_name in enumerate(sorted(drug_overlap)[:30]):
        tahoe_name = tahoe_drugs_lookup.get(norm_name, '?')
        gdsc_name = gdsc_drugs_lookup.get(norm_name, '?')
        print(f"  {tahoe_name:25} ↔ {gdsc_name}")
    if len(drug_overlap) > 30:
        print(f"  ... and {len(drug_overlap) - 30} more")

    # check MEK inhibitors specifically
    print(f"\n--- MEK INHIBITORS IN GDSC ---")
    mek_drugs = gdsc_compounds[gdsc_compounds['DRUG_NAME'].str.upper().isin([
        'TRAMETINIB', 'COBIMETINIB', 'BINIMETINIB', 'SELUMETINIB', 'TAK-733'
    ])]
    if len(mek_drugs) > 0:
        print(mek_drugs[['DRUG_NAME', 'DRUG_ID', 'TARGET', 'TARGET_PATHWAY']].to_string(index=False))
    else:
        # search more broadly
        mek_related = gdsc_compounds[
            gdsc_compounds['TARGET_PATHWAY'].str.contains('MEK|ERK|MAPK', case=False, na=False) |
            gdsc_compounds['DRUG_NAME'].str.contains('metinib|tinib', case=False, na=False)
        ]
        print("MEK-related drugs found:")
        print(mek_related[['DRUG_NAME', 'DRUG_ID', 'TARGET', 'TARGET_PATHWAY']].head(15).to_string(index=False))

    # check KRAS cell lines specifically
    print(f"\n--- KRAS-MUTANT CELL LINES ---")
    kras_tahoe = tahoe_cells[tahoe_cells['Driver_Gene_Symbol'] == 'KRAS']['cell_name'].unique()
    print(f"Tahoe-100M KRAS-mutant lines: {len(kras_tahoe)}")

    kras_in_gdsc = []
    for cell in kras_tahoe:
        norm = cell.upper().replace('-', '').replace(' ', '')
        if norm in gdsc_cell_names:
            kras_in_gdsc.append(cell)

    print(f"Also in GDSC: {len(kras_in_gdsc)}")
    print(f"  {kras_in_gdsc[:15]}")

    # show sample IC50 data for MEK inhibitors in KRAS lines
    print(f"\n--- SAMPLE IC50 DATA ---")
    print("(Looking for MEK inhibitor response in KRAS-mutant cell lines)")

    # find drug IDs for MEK inhibitors
    mek_ids = gdsc_compounds[
        gdsc_compounds['DRUG_NAME'].str.upper().isin(['TRAMETINIB', 'SELUMETINIB', 'COBIMETINIB'])
    ]['DRUG_ID'].tolist()

    if mek_ids:
        # filter GDSC2 data
        sample_data = gdsc2[
            (gdsc2['DRUG_ID'].isin(mek_ids)) &
            (gdsc2['CELL_LINE_NAME'].str.upper().str.replace('-', '').str.replace(' ', '').isin(
                [c.upper().replace('-', '').replace(' ', '') for c in kras_in_gdsc]
            ))
        ][['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'AUC', 'RMSE']].head(20)

        if len(sample_data) > 0:
            # convert LN_IC50 to IC50 in uM
            sample_data = sample_data.copy()
            sample_data['IC50_uM'] = (2.718281828 ** sample_data['LN_IC50']).round(4)
            print(sample_data.to_string(index=False))
        else:
            print("No matching data found in GDSC2 for these combinations")

    return {
        'cell_overlap': cell_overlap,
        'drug_overlap': drug_overlap,
        'kras_in_gdsc': kras_in_gdsc,
    }


def main():
    parser = argparse.ArgumentParser(description="Download GDSC data")
    parser.add_argument('--force', action='store_true', help='Re-download even if files exist')
    parser.add_argument('--download-only', action='store_true', help='Only download, skip analysis')
    args = parser.parse_args()

    download_gdsc(force=args.force)

    if not args.download_only:
        analyze_overlaps()


if __name__ == "__main__":
    main()
