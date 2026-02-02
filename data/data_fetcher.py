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

def create_mock_fits_file(filepath, time_points=2000, injected_params=None):
    """Creates a fake FITS file with a plausible light curve structure and provenance."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    primary_hdu = fits.PrimaryHDU()
    # --- Add Provenance to Header ---
    primary_hdu.header['TELESCOP'] = 'Synthetic'
    primary_hdu.header['TICID'] = np.random.randint(1000000, 9999999)
    primary_hdu.header['SECTOR'] = 1
    
    time_array = np.linspace(0, 100, time_points)
    flux_array = np.random.normal(1.0, 0.01, size=time_points)

    if injected_params:
        primary_hdu.header['INJECT'] = (True, 'Data was injected synthetically')
        # Simple transit injection
        period = injected_params.get('period', 10)
        epoch = injected_params.get('epoch', 5)
        depth = injected_params.get('depth', 0.01)
        duration = injected_params.get('duration', 0.1)
        
        for t_transit in np.arange(epoch, time_array.max(), period):
            transit_mask = np.abs(time_array - t_transit) < (duration / 2)
            flux_array[transit_mask] -= depth

    else:
        primary_hdu.header['INJECT'] = (False, 'No data injected')

    time_col = fits.Column(name='TIME', format='D', array=time_array)
    flux_col = fits.Column(name='PDCSAP_FLUX', format='D', array=flux_array)
    
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
        is_planet = (i % 2 == 0)
        label = 'confirmed_planet' if is_planet else 'false_positive'
        filename = f"synthetic_{label}_{i}.fits"
        filepath = os.path.join(output_dir, filename)
        
        injected_params = None
        if is_planet:
            injected_params = {'period': np.random.uniform(5, 20), 'depth': np.random.uniform(0.005, 0.02)}

        create_mock_fits_file(filepath, injected_params=injected_params)
        
        typed_files.append({
            "file_path": filepath,
            "type": label
        })
        logger.debug(f"Created synthetic file: {filepath}")
        
    logger.info(f"Generated {len(typed_files)} synthetic files in {output_dir}")
    return typed_files