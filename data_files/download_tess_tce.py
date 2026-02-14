import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def process_tess_data_from_manual_download(tess_input_csv_path, output_tess_metadata_path):
    print(f"Starting TESS data processing from manual download: {tess_input_csv_path}")

    if not os.path.exists(tess_input_csv_path):
        print(f"Error: TESS input CSV '{tess_input_csv_path}' not found. Please ensure it has been downloaded manually.")
        sys.exit(1)

    try:
        tess_df = pd.read_csv(tess_input_csv_path, comment='#') # Assuming '#' can be a comment character
        print(f"Loaded TESS data from {tess_input_csv_path}. Shape: {tess_df.shape}")

        # --- Column mapping and standardization for TESS TCE data ---
        # Map TIC ID to target_id
        if 'tic_id' in tess_df.columns:
            tess_df.rename(columns={'tic_id': 'target_id'}, inplace=True)
        elif 'TICID' in tess_df.columns: # Common alternative for TIC ID
            tess_df.rename(columns={'TICID': 'target_id'}, inplace=True)
        elif 'TID' in tess_df.columns: # Another common alternative for TIC ID
            tess_df.rename(columns={'TID': 'target_id'}, inplace=True)
        else:
            print("Error: Could not find 'tic_id', 'TICID', or 'TID' column in TESS input CSV for target_id mapping.")
            sys.exit(1)

        # Map period
        if 'period' not in tess_df.columns:
            print("Error: 'period' column not found in TESS input CSV.")
            sys.exit(1)

        # Map tce_disposition or tce_tags to true_label
        if 'tce_disposition' in tess_df.columns:
            tess_df['true_label'] = tess_df['tce_disposition'].map({
                'CANDIDATE': 'CANDIDATE',
                'CONFIRMED': 'CANDIDATE', # Treat confirmed as candidate for consistency with Kepler
                'FALSE POSITIVE': 'FALSE_POSITIVE'
            })
        elif 'tce_tags' in tess_df.columns:
            # This is more complex, tce_tags might contain multiple flags
            # A common approach is to look for 'FP' for false positives, otherwise assume candidate
            tess_df['true_label'] = np.where(tess_df['tce_tags'].str.contains('FP', na=False), 'FALSE_POSITIVE', 'CANDIDATE')
        else:
            print("Warning: Neither 'tce_disposition' nor 'tce_tags' found. Defaulting 'true_label' to 'CANDIDATE' for all TESS entries.")
            tess_df['true_label'] = 'CANDIDATE'
        
        # Select and ensure necessary columns exist
        required_columns = ['target_id', 'period', 'true_label']
        if not all(col in tess_df.columns for col in required_columns):
            print(f"Error: Missing one or more required columns after processing TESS data: {required_columns}")
            sys.exit(1)

        tess_final = tess_df[required_columns].copy()
        tess_final['source'] = 'TESS'

        # Ensure target_id is string for consistent merging later
        tess_final['target_id'] = tess_final['target_id'].astype(str)

        # Deduplicate based on target_id (TIC ID) and optionally period, keeping the most confident/highest SNR
        # Assuming higher SNR or smaller period implies better candidate for deduplication if needed
        # For simplicity, let's deduplicate on target_id, keeping the first occurrence
        tess_final.drop_duplicates(subset=['target_id'], keep='first', inplace=True)
        print(f"Processed TESS data with {len(tess_final)} unique targets.")

        tess_final.to_csv(output_tess_metadata_path, index=False)
        print(f"Processed TESS metadata saved to {output_tess_metadata_path}")
        print(f"Breakdown of labels in {output_tess_metadata_path}:\n{tess_final['true_label'].value_counts()}")

    except Exception as e:
        print(f"An error occurred during TESS data processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    tess_manual_csv = current_dir / 'tess_tce_catalog.csv' # Expected manual download
    output_tess_metadata = current_dir / 'tess_metadata.csv'

    process_tess_data_from_manual_download(tess_manual_csv, output_tess_metadata)