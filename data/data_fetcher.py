# data/data_fetcher.py

import logging
import os
import numpy as np
from astropy.io import fits

# We keep the dependency checks for completeness, but won't rely on them for the mock run
from utils.dependencies import ASTROQUERY_AVAILABLE, LIGHTKURVE_AVAILABLE

logger = logging.getLogger(__name__)

def create_mock_fits_file(filepath, time_points=2000):
    """Creates a fake FITS file with a plausible light curve structure."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a primary HDU (Header Data Unit) - often empty
    primary_hdu = fits.PrimaryHDU()

    # Create a binary table HDU for the light curve data
    time_col = fits.Column(name='TIME', format='D', array=np.linspace(0, 100, time_points))
    flux_col = fits.Column(name='PDCSAP_FLUX', format='D', array=np.random.normal(1.0, 0.01, size=time_points))
    
    cols = fits.ColDefs([time_col, flux_col])
    hdu = fits.BinTableHDU.from_columns(cols)
    
    # Create an HDU list and write to file
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(filepath, overwrite=True)
    hdul.close()


def get_mock_target_lists_and_data(config):
    """
    Generates mock target lists and creates corresponding fake FITS files.
    This function replaces the need for astroquery and lightkurve for a test run.
    """
    logger.info("--- RUNNING IN MOCK DATA MODE ---")
    logger.info("Skipping real data fetching and generating fake FITS files.")

    mock_download_dir = os.path.join(config.DATA_DIR, "mock_data")
    os.makedirs(mock_download_dir, exist_ok=True)

    num_mock_files = getattr(config, 'DOWNLOAD_LIMIT', 10) * 2  # Confirmed and false positives
    
    mock_files_info = []
    for i in range(num_mock_files):
        label = 'confirmed' if i % 2 == 0 else 'false_positive'
        filename = f"mock_kic_{i}.fits"
        filepath = os.path.join(mock_download_dir, filename)
        
        create_mock_fits_file(filepath)
        
        mock_files_info.append({
            "file_path": filepath,
            "label": label
        })
        logger.debug(f"Created mock file: {filepath}")
        
    logger.info(f"Generated {len(mock_files_info)} mock data files in {mock_download_dir}")
    return mock_files_info

# --- Original Functions (kept for reference but will be bypassed) ---

def get_kepler_target_lists():
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not available. Cannot fetch target lists.")
        return None, None
    # ... (original astroquery logic)
    return [], []

def download_light_curves(target_ids, label, download_dir):
    if not LIGHTKURVE_AVAILABLE:
        logger.error("Lightkurve is not available. Cannot download light curves.")
        return []
    # ... (original lightkurve logic)
    return []