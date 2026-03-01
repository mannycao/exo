import pandas as pd
import os
from discovery_stack.sentinel import DataSentinel
from discovery_stack.registry import TargetRegistry

def download_kepler_files_from_sh(kepler_sh_path="data/Kepler.sh", csv_path="kepler_confirmed.csv"):
    """
    Reads confirmed Kepler IDs from CSV, then uses DataSentinel to parse
    Kepler.sh and download corresponding FITS files.
    """
    
    # --- Step 1: Read confirmed Kepler IDs from CSV ---
    print(f"Reading confirmed Kepler IDs from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, skiprows=58) # skiprows=58 for the header
        if 'koi_disposition' in df.columns:
            confirmed_kepler_df = df[df['koi_disposition'] == 'CONFIRMED']
        else:
            print("Error: 'koi_disposition' column not found in kepler_confirmed.csv. Please check your CSV.")
            return

        kepler_id_column = 'kepid'
        if kepler_id_column not in confirmed_kepler_df.columns:
            print(f"Error: '{kepler_id_column}' column not found in the filtered CSV.")
            print("Available columns:", confirmed_kepler_df.columns.tolist())
            return
        
        # Convert to list of strings for DataSentinel
        confirmed_kic_ids_list = [str(int(kid)) for kid in confirmed_kepler_df[kepler_id_column].dropna().unique()]
        print(f"Found {len(confirmed_kic_ids_list)} unique confirmed Kepler IDs in CSV.")

        if not confirmed_kic_ids_list:
            print("No confirmed Kepler IDs found after filtering from CSV.")
            return

    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please ensure the CSV is in the correct directory.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {csv_path}: {e}")
        return

    # --- Step 2: Instantiate DataSentinel and trigger download ---
    print("\nInitializing DataSentinel for downloading...")
    DOWNLOAD_FOLDER = "kepler_fits_files" # Hardcoded download folder for this utility
    REGISTRY_FILE = "processed_kepler_sh_downloads.json" # Use a separate registry for these downloads

    registry = TargetRegistry(REGISTRY_FILE)
    sentinel = DataSentinel(registry, DOWNLOAD_FOLDER) # max_disk_usage_pct uses default
    
    print(f"DataSentinel initialized. Downloads will go to '{DOWNLOAD_FOLDER}'.")
    print(f"Download progress will be logged in '{REGISTRY_FILE}'.")

    try:
        sentinel.poll_from_kepler_sh_and_download_kepler(kepler_sh_path, confirmed_kic_ids_list)
        print("\nKepler FITS file download process initiated.")
    except Exception as e:
        print(f"An error occurred during the download process: {e}")

if __name__ == "__main__":
    download_kepler_files_from_sh()
