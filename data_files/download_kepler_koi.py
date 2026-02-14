import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import re

# --- Logging Setup ---
LOG_FILE = Path(__file__).resolve().parent / "download_kepler_koi.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Paths ---
# Assuming project root is two levels up from data_files/
project_root = Path(__file__).resolve().parents[1]
DATA_FILES_DIR = project_root / "data_files"
CUMULATIVE_KOI_CSV = DATA_FILES_DIR / "cumulative_koi.csv"

# URL for Kepler KOI bulk data
KEPLER_KOI_BULK_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv"

def download_and_process_kepler_koi():
    logger.info(f"Starting download and processing of Kepler KOI data from {KEPLER_KOI_BULK_URL}")

    # 1. Download Kepler KOI Bulk CSV
    try:
        kepler_koi_df = pd.read_csv(KEPLER_KOI_BULK_URL, comment='#') # comment='#' to skip header comments
        logger.info(f"Downloaded Kepler KOI data. Shape: {kepler_koi_df.shape}")
        logger.debug(f"Kepler KOI columns: {kepler_koi_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error downloading Kepler KOI bulk file: {e}", exc_info=True)
        sys.exit(1)

    # 2. Clean & Map Kepler KOI Data
    # Expected: kepid, koi_period, koi_disposition
    
    # Rename kepid to target_id
    if 'kepid' in kepler_koi_df.columns:
        kepler_koi_df.rename(columns={'kepid': 'target_id'}, inplace=True)
    else:
        logger.error("'kepid' column not found in Kepler KOI data. Cannot proceed.")
        sys.exit(1)

    # Rename koi_period to period
    if 'koi_period' in kepler_koi_df.columns:
        kepler_koi_df.rename(columns={'koi_period': 'period'}, inplace=True)
    else:
        logger.error("'koi_period' column not found in Kepler KOI data. Cannot proceed.")
        sys.exit(1)

    # Map koi_disposition to true_label
    kepler_koi_df['true_label'] = 'FALSE_POSITIVE' # Default value

    if 'koi_disposition' in kepler_koi_df.columns:
        kepler_koi_df['true_label'] = kepler_koi_df['koi_disposition'].apply(
            lambda x: 'CANDIDATE' if pd.notna(x) and ('CANDIDATE' in str(x).upper() or 'CONFIRMED' in str(x).upper()) else 'FALSE_POSITIVE'
        )
        logger.info("Used 'koi_disposition' for 'true_label' mapping.")
    else:
        logger.warning("'koi_disposition' column not found in Kepler KOI data. Defaulting all to 'FALSE_POSITIVE'.")
    
    # Ensure target_id is string for consistent merging
    kepler_koi_df['target_id'] = kepler_koi_df['target_id'].astype(str)

    # Select only relevant columns
    kepler_koi_df = kepler_koi_df[['target_id', 'period', 'true_label']]
    logger.info(f"Cleaned Kepler KOI data. Shape: {kepler_koi_df.shape}")

    # 3. Save Processed Kepler KOI Data
    kepler_koi_df.to_csv(CUMULATIVE_KOI_CSV, index=False)
    logger.info(f"Kepler KOI data saved to: {CUMULATIVE_KOI_CSV}")
    logger.info(f"Total Kepler KOIs: {len(kepler_koi_df)}")


if __name__ == "__main__":
    # Add project root to sys.path if not already there, for general robustness
    sys.path.insert(0, str(project_root))
    download_and_process_kepler_koi()
