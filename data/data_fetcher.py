# data/data_fetcher.py

import logging
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.mast import Observations
from astroquery.utils.tap.core import TapPlus

# Import the centralized dependencies and availability flags
from utils.dependencies import (
    lk, LIGHTKURVE_AVAILABLE,
    ExoplanetArchive, ASTROQUERY_AVAILABLE
)

logger = logging.getLogger(__name__)

# --- Consolidated Data Fetching Functions ---

def fetch_kepler_data(max_records=50, use_cache=True):
    """Fetches light curve data from Kepler mission with local caching."""
    METADATA_DIR = "data/metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)
    cache_file = os.path.join(METADATA_DIR, "kepler_observation_list.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached Kepler observation list")
        obs_table = pd.read_pickle(cache_file)
        return obs_table[:max_records]
    
    print(f"Fetching Kepler observations from MAST")
    # Query for general Kepler time-series data to avoid resolver errors
    obs_table = Observations.query_criteria(
        obs_collection='Kepler', 
        dataproduct_type="timeseries"
    )
    
    obs_table.to_pandas().to_pickle(cache_file)
    return obs_table[:max_records]

def download_product(product, use_cache=True):
    """Downloads a data product with caching."""
    LIGHT_CURVE_DIR = "data/light_curves"
    os.makedirs(LIGHT_CURVE_DIR, exist_ok=True)
    filename = f"{product['obs_id']}_{product['dataproduct_type']}.fits"
    local_path = os.path.join(LIGHT_CURVE_DIR, filename)
    
    if use_cache and os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return local_path
    
    print(f"Downloading: {product['dataURI']}")
    try:
        download_path = Observations.download_file(product['dataURI'], local_path=local_path)
        return download_path[0] if isinstance(download_path, list) else download_path
    except Exception as e:
        print(f"Error downloading {product['dataURI']}: {e}")
        return None

def download_light_curves(obs_table, use_cache=True):
    """Downloads light curves from an observation table."""
    light_curve_files = []
    for obs in obs_table:
        try:
            data_products = Observations.get_product_list(obs)
            light_curve_products = [p for p in data_products if 'LIGHTCURVE' in p['dataURI']]
            
            if light_curve_products:
                file_path = download_product(light_curve_products[0], use_cache=use_cache)
                if file_path:
                    light_curve_files.append(file_path)
        except Exception as e:
            print(f"Error processing observation {obs['obs_id']}: {e}")
    return light_curve_files

def fetch_exoplanet_labels(use_cache=True):
    """Fetches confirmed exoplanet data for training labels using the updated TAP service."""
    METADATA_DIR = "data/metadata"
    os.makedirs(METADATA_DIR, exist_ok=True)
    cache_file = os.path.join(METADATA_DIR, "exoplanet_labels.csv")
    
    if use_cache and os.path.exists(cache_file):
        print("Loading cached exoplanet labels")
        return pd.read_csv(cache_file)
    
    print("Fetching exoplanet data from NASA Exoplanet Archive TAP service")
    tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP")
    query = "SELECT pl_name, hostname, pl_orbper, pl_rade, pl_masse, disc_year, discoverymethod FROM ps WHERE default_flag = 1"
    
    result = tap.launch_job(query)
    exoplanet_data = result.get_results()
    
    exoplanet_data.to_pandas().to_csv(cache_file, index=False)
    return exoplanet_data.to_pandas()

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