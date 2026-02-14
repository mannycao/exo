import pandas as pd
import os
import re
import subprocess
from pathlib import Path
import logging
import time
import requests
from bs4 import BeautifulSoup
import lightkurve as lk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kepler_lcs(metadata_path: Path, output_dir: Path):
    """
    Downloads Kepler Light Curve FITS files based on target_ids from metadata
    using the lightkurve library.
    """
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path, index_col='target_id')
    logger.info(f"Loaded Kepler metadata from {metadata_path}. Shape: {metadata_df.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for Kepler targets
    kepler_targets = metadata_df[metadata_df['source'] == 'KEPLER']
    logger.info(f"Found {len(kepler_targets)} Kepler targets in metadata.")

    if kepler_targets.empty:
        logger.warning("No Kepler targets found in the metadata to download.")
        return

    for target_id in kepler_targets.index:
        kic_id_str = str(target_id)
        
        # Define expected filename for checking existing files
        expected_filename_pattern = f"kplr{kic_id_str}*.fits"
        existing_files = list(output_dir.glob(expected_filename_pattern))
        if existing_files:
            # Check if any existing file is non-empty
            if any(f.is_file() and os.path.getsize(f) > 0 for f in existing_files):
                logger.info(f"Skipping KIC ID {target_id}: valid FITS file(s) already exist ({[f.name for f in existing_files]}).")
                continue
            else:
                # Clean up any 0-byte files that match the pattern
                for f in existing_files:
                    if f.is_file() and os.path.getsize(f) == 0:
                        logger.info(f"Deleting existing 0-byte file: {f.name}")
                        f.unlink()

        logger.info(f"Searching for light curve files for KIC ID {kic_id_str} using Lightkurve...")
        try:
            # Search for Kepler long cadence (LC) data
            # The warning suggests `search_lightcurve()` which returns LightCurve objects,
            # not LightCurveFile objects needed to save as FITS. So, we'll keep `search_lightcurvefile`
            # but access its elements correctly for logging.
            lcf_collection = lk.search_lightcurvefile(f"KIC {kic_id_str}", mission='Kepler', cadence='long')
            
            if not lcf_collection:
                logger.warning(f"No Kepler light curve files found for KIC ID {kic_id_str}. Skipping.")
                continue
            
            downloaded_lc_file = None
            # Iterate through the returned LightCurveFileCollection to find a suitable one
            # lcf_collection is a SearchResult, which is a table. Access items as rows.
            for i, result_row in enumerate(lcf_collection):
                # LightCurveFile objects are obtained by indexing the collection
                # For logging, we can access attributes of the result_row (which is like a dictionary or object)
                # or download the file and then access its attributes.
                # Let's directly download the first suitable one.
                
                # lightkurve.download() on a search result row often does the right thing.
                # The attributes like .object, .exptime, .mission, .quarter are present in the *table rows*,
                # but direct access like `lcf.object` might be better on the actual LightCurveFile object
                # after it's downloaded, or from the metadata table.
                # Let's simplify and just download the first one, then log if successful.
                
                try:
                    # Download the specific LightCurveFile object from the collection
                    # .download() directly on the SearchResult table will download the "best" one
                    # which is usually the one with the most data (longest baseline).
                    logger.info(f"Attempting to download best available light curve for KIC ID {kic_id_str} (Row {i+1}/{len(lcf_collection)}) to {output_dir}...")
                    
                    # Call .download() on the specific row (which is effectively a LightCurveFile object wrapper)
                    downloaded_lc_file_path = lcf_collection[i].download(directory=output_dir)
                    
                    if downloaded_lc_file_path:
                        logger.info(f"Successfully downloaded light curve file for KIC ID {kic_id_str}: {downloaded_lc_file_path.name}")
                        downloaded_lc_file = True # Indicate success
                        break # Download the first suitable one and move on
                    else:
                        logger.warning(f"Download returned None for KIC ID {kic_id_str} (Row {i+1}/{len(lcf_collection)}).")
                except Exception as e:
                    logger.warning(f"Error downloading light curve for KIC ID {kic_id_str} (Row {i+1}/{len(lcf_collection)}): {e}")

            if not downloaded_lc_file:
                logger.warning(f"Could not download a suitable light curve file for KIC ID {kic_id_str} after searching available files.")

        except Exception as e:
            logger.error(f"An error occurred while downloading light curve for KIC ID {kic_id_str} using Lightkurve: {e}", exc_info=True)
        
        # Be a good citizen, don't hammer the server
        time.sleep(1) # 1 second delay between downloads

if __name__ == "__main__":
    METADATA_PATH = Path("/Users/emmanuel/proj/phd/data_files/full_metadata.csv")
    OUTPUT_DIR = Path("/Users/emmanuel/proj/phd/data_files/light_curves")
    
    # Clean up any remaining 0-byte files before starting a new download attempt
    for f in OUTPUT_DIR.glob("*.fits"):
        if f.is_file() and os.path.getsize(f) == 0:
            logger.info(f"Deleting existing 0-byte file: {f}")
            f.unlink()

    download_kepler_lcs(METADATA_PATH, OUTPUT_DIR)
