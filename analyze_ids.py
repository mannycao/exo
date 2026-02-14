import pandas as pd
from pathlib import Path
import re

# Define paths
project_root = Path(__file__).resolve().parent
data_files_dir = project_root / "data_files"

FULL_METADATA_CSV = project_root / "data_files" / "full_metadata.csv"
KEPLER_LOCAL_FILE = project_root / "kep.txt"
TESS_LOCAL_FILE = project_root / "tess.txt"

# --- Analysis Function ---
def analyze_target_ids():
    print(f"Analyzing target IDs across metadata files and light curve filename parsing patterns...")

    # 1. Load full_metadata.csv
    if not FULL_METADATA_CSV.exists():
        print(f"Error: {FULL_METADATA_CSV} not found.")
        return
    df_full_metadata = pd.read_csv(FULL_METADATA_CSV)
    metadata_ids = set(df_full_metadata['target_id'].astype(str).tolist())
    print(f"Found {len(metadata_ids)} unique target_ids in {FULL_METADATA_CSV}")

    # 2. Load kep.txt
    if not KEPLER_LOCAL_FILE.exists():
        print(f"Error: {KEPLER_LOCAL_FILE} not found.")
        return
    df_kep = pd.read_csv(KEPLER_LOCAL_FILE)
    kep_ids = set(df_kep['kepid'].astype(str).tolist())
    print(f"Found {len(kep_ids)} unique target_ids in {KEPLER_LOCAL_FILE}")

    # 3. Load tess.txt
    if not TESS_LOCAL_FILE.exists():
        print(f"Error: {TESS_LOCAL_FILE} not found.")
        return
    df_tess = pd.read_csv(TESS_LOCAL_FILE)
    tess_ids = set(df_tess['tid'].astype(str).tolist())
    print(f"Found {len(tess_ids)} unique target_ids in {TESS_LOCAL_FILE}")

    # --- Comparisons ---
    print("\n--- Comparison: kep.txt IDs vs. full_metadata.csv IDs ---")
    in_kep_not_in_metadata = kep_ids - metadata_ids
    in_metadata_not_in_kep = metadata_ids - kep_ids
    print(f"IDs in {KEPLER_LOCAL_FILE} but NOT in {FULL_METADATA_CSV}: {len(in_kep_not_in_metadata)}")
    if in_kep_not_in_metadata:
        print(f"Sample missing from metadata: {list(in_kep_not_in_metadata)[:5]}")

    print(f"IDs in {FULL_METADATA_CSV} but NOT in {KEPLER_LOCAL_FILE}: {len(in_metadata_not_in_kep)}")
    if in_metadata_not_in_kep:
        print(f"Sample missing from kep.txt: {list(in_metadata_not_in_kep)[:5]}")

    print("\n--- Comparison: tess.txt IDs vs. full_metadata.csv IDs ---")
    in_tess_not_in_metadata = tess_ids - metadata_ids
    in_metadata_not_in_tess = metadata_ids - tess_ids
    print(f"IDs in {TESS_LOCAL_FILE} but NOT in {FULL_METADATA_CSV}: {len(in_tess_not_in_metadata)}")
    if in_tess_not_in_metadata:
        print(f"Sample missing from metadata: {list(in_tess_not_in_metadata)[:5]}")

    print(f"IDs in {FULL_METADATA_CSV} but NOT in {TESS_LOCAL_FILE}: {len(in_metadata_not_in_tess)}")
    if in_metadata_not_in_tess:
        print(f"Sample missing from tess.txt: {list(in_metadata_not_in_tess)[:5]}")

    print("\n--- Comparison: All IDs from kep.txt and tess.txt vs. full_metadata.csv IDs ---")
    all_source_ids = kep_ids.union(tess_ids)
    mismatched_ids = all_source_ids - metadata_ids
    print(f"IDs from kep.txt or tess.txt that are NOT in full_metadata.csv: {len(mismatched_ids)}")
    if mismatched_ids:
        print(f"Sample missing from metadata: {list(mismatched_ids)[:10]}")

    print("\n--- Investigating filenames parsing ---")
    print("Cannot directly analyze light curve filenames without access to the directories.")
    print("Consider adding a logging step in run_full_survey.py to log extracted IDs and metadata lookup results.")


if __name__ == "__main__":
    analyze_target_ids()