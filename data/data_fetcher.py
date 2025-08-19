# data/data_fetcher.py

import logging
import os
import numpy as np
import pandas as pd
from astropy.io import fits
# Removed unused imports: from astroquery.mast import Observations, from astroquery.utils.tap.core import TapPlus

# Import the centralized dependencies and availability flags
# Removed unused imports: lk, LIGHTKURVE_AVAILABLE, ExoplanetArchive, ASTROQUERY_AVAILABLE

logger = logging.getLogger(__name__)

# --- Consolidated Data Fetching Functions (Only keeping mock data generation) ---

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
    """
    logger.info(f"--- Generating {count} synthetic light curve files ---")
    os.makedirs(output_dir, exist_ok=True)

    typed_files = []
    for i in range(count):
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