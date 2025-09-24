#!/usr/bin/env python3
"""
Script to download Kepler light curves using a local CSV list of Kepler IDs
and the astroquery library for robust MAST archive access.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from astroquery.mast import Observations

def read_kic_ids_from_csv(csv_path, kic_column_name):
    """
    Reads a list of Kepler IDs from a specified column in a CSV file.
    
    Args:
        csv_path (str): The path to the input CSV file.
        kic_column_name (str): The name of the column containing the Kepler IDs.
        
    Returns:
        list: A list of unique integer Kepler IDs.
    """
    if not csv_path or not os.path.exists(csv_path):
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

def download_kepler_light_curves(kic_id, output_dir):
    """
    Downloads all long-cadence Kepler light curve files for a given KIC ID using astroquery.
    
    Args:
        kic_id (int): Kepler Input Catalog ID.
        output_dir (str): The directory to save the downloaded files.
        
    Returns:
        int: The number of files successfully downloaded for this KIC ID.
    """
    try:
        print(f"DEBUG: Querying MAST for KIC {kic_id}...")
        obs_table = Observations.query_object(f"KIC {kic_id}", radius=".001 deg")
        print(f"DEBUG: obs_table for KIC {kic_id}: {len(obs_table)} observations found.")
        
        if len(obs_table) == 0:
            print(f"DEBUG: No observations found for KIC {kic_id}.")
            return 0

        kepler_products = Observations.get_product_list(obs_table)
        print(f"DEBUG: kepler_products for KIC {kic_id}: {len(kepler_products)} products found.")

        if len(kepler_products) == 0:
            print(f"DEBUG: No Kepler products found for KIC {kic_id}.")
            return 0
        
        products_to_download = Observations.filter_products(kepler_products,
                                                            productSubGroupDescription="LC")
        print(f"DEBUG: products_to_download for KIC {kic_id}: {len(products_to_download)} LC products found.")
        
        if len(products_to_download) == 0:
            print(f"DEBUG: No long-cadence light curves found for KIC {kic_id}.")
            return 0

        manifest = Observations.download_products(products_to_download,
                                                  download_dir=output_dir)
        
        downloaded_count = len(manifest) if manifest else 0
        print(f"DEBUG: Successfully downloaded {downloaded_count} files for KIC {kic_id}.")
        return downloaded_count
    except Exception as e:
        print(f"ERROR: Exception during download for KIC {kic_id}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description='Download Kepler data by reading KIC IDs from CSV files.'
    )
    parser.add_argument('--confirmed_csv', type=str,
                        help='Path to the CSV file with confirmed planet Kepler IDs.')
    parser.add_argument('--fp_csv', type=str,
                        help='Path to the CSV file with false positive Kepler IDs.')
    parser.add_argument('--output_base', type=str, default='kepler_local_data',
                        help='Base directory where data folders will be created.')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count() * 2,
                        help='Maximum number of concurrent download workers.')
    parser.add_argument('--type', type=str, default='all', choices=['all', 'confirmed', 'fp'],
                        help='Type of data to download.')
    
    args = parser.parse_args()
    
    confirmed_dir = os.path.join(args.output_base, "confirmed_planets")
    fp_dir = os.path.join(args.output_base, "false_positives")
    os.makedirs(confirmed_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    
    confirmed_ids = []
    fp_ids = []
    
    if args.type in ['all', 'confirmed'] and args.confirmed_csv:
        confirmed_ids = read_kic_ids_from_csv(args.confirmed_csv, kic_column_name='kepid')
    
    if args.type in ['all', 'fp'] and args.fp_csv:
        fp_ids = read_kic_ids_from_csv(args.fp_csv, kic_column_name='kepid')

    confirmed_success_count = 0
    if confirmed_ids:
        print(f"\nDownloading light curves for {len(confirmed_ids)} confirmed planets...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            download_func = partial(download_kepler_light_curves, output_dir=confirmed_dir)
            results = list(tqdm(executor.map(download_func, confirmed_ids), 
                                total=len(confirmed_ids), 
                                desc="Confirmed Planets"))
            confirmed_success_count = sum(results)
    
    fp_success_count = 0
    if fp_ids:
        print(f"\nDownloading light curves for {len(fp_ids)} false positives...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            download_func = partial(download_kepler_light_curves, output_dir=fp_dir)
            results = list(tqdm(executor.map(download_func, fp_ids), 
                                total=len(fp_ids), 
                                desc="False Positives"))
            fp_success_count = sum(results)
            
    print("\n====================================")
    print("      ASTROQUERY DOWNLOAD SUMMARY")
    print("====================================")
    if confirmed_ids:
        print(f"Processed {len(confirmed_ids)} confirmed planet targets, successfully downloaded {confirmed_success_count} files.")
        print(f"Files are located in subdirectories within: {confirmed_dir}")
    if confirmed_ids and fp_ids:
        print("-" * 36)
    if fp_ids:
        print(f"Processed {len(fp_ids)} false positive targets, successfully downloaded {fp_success_count} files.")
        print(f"Files are located in subdirectories within: {fp_dir}")
    print("====================================")

if __name__ == "__main__":
    main()