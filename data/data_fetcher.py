# data/data_fetcher.py

import logging
import os
import numpy as np
from astropy.io import fits
import pandas as pd

# Import the centralized dependencies and availability flags
from utils.dependencies import (
    lk, LIGHTKURVE_AVAILABLE,
    ExoplanetArchive, Observations, ASTROQUERY_AVAILABLE
)

logger = logging.getLogger(__name__)


def create_mock_fits_file(filepath, time_points=2000):
    """Creates a fake FITS file with a plausible light curve structure."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    primary_hdu = fits.PrimaryHDU()
    time_col = fits.Column(name='TIME', format='D', array=np.linspace(0, 100, time_points))
    flux_col = fits.Column(name='PDCSAP_FLUX', format='D', array=np.random.normal(1.0, 0.01, size=time_points))
    
    cols = fits.ColDefs([time_col, flux_col])
    hdu = fits.BinTableHDU.from_columns(cols)
    
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(filepath, overwrite=True)
    hdul.close()


def generate_sample_light_curves(count, output_dir):
    """
    Generates a specified number of mock FITS files for testing the pipeline.
    This function is intended for use with the --synthetic-data flag.
    """
    logger.info(f"--- Generating {count} synthetic light curve files ---")
    os.makedirs(output_dir, exist_ok=True)

    typed_files = []
    for i in range(count):
        # Alternate between creating mock "planet" and "false positive" files
        label = 'confirmed_planet' if i % 2 == 0 else 'false_positive'
        filename = f"synthetic_{label}_{i}.fits"
        filepath = os.path.join(output_dir, filename)
        
        create_mock_fits_file(filepath)
        
        typed_files.append({
            "file_path": filepath,
            "type": label
        })
        logger.debug(f"Created synthetic file: {filepath}")
        
    logger.info(f"Generated {len(typed_files)} synthetic files in {output_dir}")
    return typed_files


def fetch_exoplanet_labels(use_cache=True):
    """
    Uses astroquery to fetch a DataFrame of exoplanet data, including labels.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not available. Cannot fetch exoplanet labels.")
        return pd.DataFrame() # Return empty DataFrame

    try:
        logger.info("Querying NASA Exoplanet Archive for catalog labels...")
        # Fetch a comprehensive table of confirmed exoplanets
        labels_df = ExoplanetArchive.query_criteria(
            table="cumulative",
            select="pl_name, kepid, koi_disposition, default_flag",
            where="default_flag = 1"
        )
        if labels_df is None:
            logger.error("Failed to retrieve data from Exoplanet Archive.")
            return pd.DataFrame()
            
        return labels_df.to_pandas()

    except Exception as e:
        logger.error(f"An error occurred while querying for exoplanet labels: {e}", exc_info=True)
        return pd.DataFrame()

# You can keep your other data fetching functions (like get_kepler_koi_targets) here