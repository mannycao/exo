# paper2_experiments/investigate_fp_metadata.py
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Set

import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    """
    Investigates missing metadata for False Positive FITS files.
    """
    # Configuration
    fp_data_base_dir = "/Users/emmanuel/proj/phd/kep/false_positives"
    metadata_path = os.path.join(project_root, "data_files", "full_metadata_v2.csv")

    # 1. Retrieve indexed_fp_ids
    fp_file_index = {}
    if not os.path.exists(fp_data_base_dir):
        print(f"Error: False Positive FITS directory not found at {fp_data_base_dir}")
        return

    print(f"Indexing files in {fp_data_base_dir}...")
    for root, _, files in os.walk(fp_data_base_dir):
        for file in files:
            if file.endswith(".fits"):
                match = re.search(r'kplr(\d+)', file)
                if match:
                    target_id = int(match.group(1))
                    fp_file_index[target_id] = os.path.join(root, file)
    print(f"Found {len(fp_file_index)} local False Positive FITS files.")
    indexed_fp_ids = set(fp_file_index.keys())

    # 2. Retrieve verified_fp_metadata_ids
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    print(f"Loading metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path)
    
    # Filter for rows where 'true_label' contains "FALSE" or "POSITIVE"
    verified_fp_metadata = metadata_df[
        metadata_df['true_label'].astype(str).str.contains("FALSE|POSITIVE", na=False, regex=True)
    ]
    print(f"Metadata contains {len(verified_fp_metadata)} verified False Positives.")
    verified_fp_metadata_ids = set(verified_fp_metadata['target_id'].astype(int))

    # 3. Identify Missing IDs
    missing_ids_in_metadata = indexed_fp_ids - verified_fp_metadata_ids
    print(f"""
Found {len(missing_ids_in_metadata)} FITS files that are on disk but not in the filtered metadata as 'False Positive'.""")
    # print(f"Missing IDs: {sorted(list(missing_ids_in_metadata))}")

    # 4. Investigate Missing IDs
    print("\nInvestigating missing IDs in the full metadata...")
    found_in_full_metadata_with_different_label = []
    completely_absent_from_metadata = []

    for missing_id in sorted(list(missing_ids_in_metadata)):
        matching_rows = metadata_df[metadata_df['target_id'] == missing_id]
        if not matching_rows.empty:
            # Check if there are entries with 'true_label' that don't match our 'FALSE|POSITIVE' filter
            if not matching_rows['true_label'].astype(str).str.contains("FALSE|POSITIVE", na=False, regex=True).any():
                found_in_full_metadata_with_different_label.append({
                    'target_id': missing_id,
                    'true_label': matching_rows['true_label'].iloc[0] # Take the first label if multiple
                })
        else:
            completely_absent_from_metadata.append(missing_id)

    if found_in_full_metadata_with_different_label:
        print(f"""
{len(found_in_full_metadata_with_different_label)} IDs found in full metadata but with different or missing 'true_label':""")
        for item in found_in_full_metadata_with_different_label:
            print(f"  Target ID: {item['target_id']}, True Label: {item['true_label']}")
    else:
        print("All missing IDs (from disk) not classified as 'FALSE' or 'POSITIVE' in filtered metadata are either completely absent or have non-matching 'true_label' values.")

    if completely_absent_from_metadata:
        print(f"""
{len(completely_absent_from_metadata)} IDs are completely absent from the metadata file:""")
        print(f"  {sorted(list(completely_absent_from_metadata))}")
    else:
        print("\nAll missing IDs (from disk) are present in the full metadata, just not as 'FALSE' or 'POSITIVE'.")

if __name__ == "__main__":
    main()
