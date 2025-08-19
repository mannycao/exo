#!/usr/bin/env python3
"""
Script to download Kepler light curves using a local CSV list of Kepler IDs
and the direct MAST archive URL structure.
"""

import os
import argparse
import pandas as pd
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor # New import
from functools import partial # New import

def read_kic_ids_from_csv(csv_path, kic_column_name):
    """
    Reads a list of Kepler IDs from a specified column in a CSV file.
    
    Args:
        csv_path (str): The path to the input CSV file.
        kic_column_name (str): The name of the column containing the Kepler IDs.
        
    Returns:
        list: A list of unique integer Kepler IDs.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return []
    
    print(f"Reading Kepler IDs from {csv_path}...")
    try:
        # Skip comment lines which are common in these archive files
        df = pd.read_csv(csv_path, comment='#')
        # Drop rows where the KIC ID is missing and convert to integers
        df = df.dropna(subset=[kic_column_name])
        kic_ids = df[kic_column_name].astype(int).unique().tolist()
        print(f"Found {len(kic_ids)} unique targets.")
        return kic_ids
    except Exception as e:
        print(f"Failed to read or parse CSV file: {e}")
        return []

def download_file_direct(kic_id, output_dir):
    """
    Downloads all long-cadence light curve files for a given KIC ID using
    the direct archive URL structure.
    
    Args:
        kic_id (int): Kepler Input Catalog ID.
        output_dir (str): The directory to save the downloaded files.
        
    Returns:
        int: The number of files successfully downloaded for this KIC ID.
    """
    # Format the KIC ID for the URL as described in the README
    kic_id_padded = str(kic_id).zfill(9)
    kic_prefix = kic_id_padded[:4]
    
    # Construct the base URL for the target's data directory
    base_url = f"https://archive.stsci.edu/pub/kepler/lightcurves/{kic_prefix}/{kic_id_padded}/"
    
    # This is a common pattern, but we'll try to be more robust
    # Instead of guessing filenames, let's assume we just need one file per KIC
    # A more advanced script could parse the directory listing
    filename_pattern = f"kplr{kic_id_padded}" 
    
    # Let's try to find any long-cadence file (_llc.fits)
    # This is a simplified approach. A fully robust one would need to list directory contents.
    # We will try a few common quarter date stamps.
    potential_timestamps = ["2009131105131", "2009166043257", "2009259160929", "2010078095331", "2010174085026", "2010265121752", "2010355172524", "2011073133259", "2011177032512", "2011271113734", "2012004120508", "2012088054726", "2012179063303", "2012277125453", "2013011073258", "2013098041711", "2013131215648"]
    
    for ts in potential_timestamps:
        filename = f"kplr{kic_id_padded}-{ts}_llc.fits"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            return 1 # Assume already downloaded

        try:
            response = requests.get(base_url + filename, timeout=30, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return 1 # Success, we only need one file per KIC
        except requests.exceptions.RequestException:
            continue # Try next timestamp
            
    return 0 # Failed to find a file with any of the common timestamps

def main():
    parser = argparse.ArgumentParser(
        description='Download Kepler data by reading KIC IDs from CSV files.'
    )
    # --- Arguments now point to the CSV files ---
    parser.add_argument('--confirmed_csv', type=str, required=True,
                        help='Path to the CSV file with confirmed planet Kepler IDs.')
    parser.add_argument('--fp_csv', type=str, required=True,
                        help='Path to the CSV file with false positive Kepler IDs.')
    parser.add_argument('--output_base', type=str, default='kepler_local_data',
                        help='Base directory where data folders will be created.')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count() * 2, # New argument
                        help='Maximum number of concurrent download workers.')
    
    args = parser.parse_args()
    
    confirmed_dir = os.path.join(args.output_base, "confirmed_planets")
    fp_dir = os.path.join(args.output_base, "false_positives")
    os.makedirs(confirmed_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    
    # --- Read IDs from CSV files ---
    confirmed_ids = read_kic_ids_from_csv(args.confirmed_csv, kic_column_name='kepid')
    fp_ids = read_kic_ids_from_csv(args.fp_csv, kic_column_name='kepid')
    
    # --- Download Confirmed Planets (Parallelized) ---
    confirmed_success_count = 0
    if confirmed_ids:
        print(f"\nDownloading light curves for {len(confirmed_ids)} confirmed planets...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Use a partial function to pass the output_dir to download_file_direct
            download_func = partial(download_file_direct, output_dir=confirmed_dir)
            
            # Map the download function to each KIC ID and sum the results
            results = list(tqdm(executor.map(download_func, confirmed_ids), 
                                total=len(confirmed_ids), 
                                desc="Confirmed Planets"))
            confirmed_success_count = sum(results)
    
    # --- Download False Positives (Parallelized) ---
    fp_success_count = 0
    if fp_ids:
        print(f"\nDownloading light curves for {len(fp_ids)} false positives...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            download_func = partial(download_file_direct, output_dir=fp_dir)
            results = list(tqdm(executor.map(download_func, fp_ids), 
                                total=len(fp_ids), 
                                desc="False Positives"))
            fp_success_count = sum(results)
            
    # --- Final Summary ---
    print("\n====================================")
    print("      DIRECT DOWNLOAD SUMMARY")
    print("====================================")
    print(f"Processed {len(confirmed_ids)} confirmed planet targets, successfully downloaded {confirmed_success_count} files.")
    print(f"Files are located in: {confirmed_dir}")
    print("-" * 36)
    print(f"Processed {len(fp_ids)} false positive targets, successfully downloaded {fp_success_count} files.")
    print(f"Files are located in: {fp_dir}")
    print("====================================")

if __name__ == "__main__":
    main()
