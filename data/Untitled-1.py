# data/sensor_fusion_data_loader.py

import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Assuming your existing light curve processing functions are in a utils file
from legacy_pipeline import preprocess_light_curve, create_image_representations, extract_transit_features

# --- NEW: Import the real data fetching functions ---
from data.archive_fetcher import get_star_info, get_rv_data, get_imaging_data

logger = logging.getLogger(__name__)

def get_kepler_id_from_path(file_path):
    """Extracts Kepler ID from the light curve file path."""
    try:
        # Assumes filename format like 'kplr006278762-2009166043257_llc.fits'
        return Path(file_path).stem.split('-')[0].replace('kplr', '').lstrip('0')
    except Exception:
        return None

def load_and_align_sensor_data(light_curve_files, image_size, rv_data_dir=None, imaging_data_dir=None):
    """
    Loads and aligns data from all three sensor types by querying the NASA Exoplanet Archive.
    1. Transit Photometry (from local light curve files)
    2. Radial Velocity (from archive)
    3. High-Resolution Imaging (from archive)

    Args:
        light_curve_files (list): A list of dictionaries with 'file_path' and 'type'.
        image_size (tuple): The target size for the 2D image representations.
        rv_data_dir (Path, optional): Not used for archive fetching, kept for compatibility.
        imaging_data_dir (Path, optional): Not used for archive fetching, kept for compatibility.

    Returns:
        A tuple containing the aligned data arrays (X_img, X_ts, X_rv, X_imaging, y).
    """
    logger.info("Starting data loading and alignment for all sensors...")
    all_photometry_img = []
    all_photometry_ts = []
    all_rv = []
    all_imaging = []
    all_labels = []

    for item in tqdm(light_curve_files, desc="Processing Stars and Fetching Archive Data"):
        file_path = item['file_path']
        label_type = item['type']
        kepler_id = get_kepler_id_from_path(file_path)

        if not kepler_id:
            logger.warning(f"Could not extract Kepler ID from {file_path}. Skipping.")
            continue

        # --- Step 1: Get Star Hostname from Kepler ID ---
        star_info = get_star_info(kepler_id)
        if not star_info:
            logger.warning(f"Could not find hostname for KIC {kepler_id}. Skipping.")
            continue
        hostname = star_info['hostname']
        
        # --- Step 2: Load and process local transit photometry data ---
        time, flux = preprocess_light_curve(file_path)
        if time is None or flux is None:
            continue
        transit_segments = extract_transit_features(time, flux)
        if not transit_segments:
            continue
        
        segment_1d = transit_segments[0]
        image_2d = create_image_representations([segment_1d], img_size=image_size)
        if image_2d is None:
            continue

        # --- Step 3: Fetch and process radial velocity data from archive ---
        rv_data = get_rv_data(hostname)
        if rv_data is None:
            logger.warning(f"Could not load RV data for {hostname} (KIC {kepler_id}). Skipping.")
            continue

        # --- Step 4: Fetch and process high-resolution imaging data from archive ---
        imaging_data = get_imaging_data(hostname)
        if imaging_data is None:
            logger.warning(f"Could not load imaging data for {hostname} (KIC {kepler_id}). Skipping.")
            continue
            
        # --- Step 5: If all data is present, add it to our lists ---
        all_photometry_img.append(image_2d)
        all_photometry_ts.append(segment_1d)
        all_rv.append(rv_data)
        all_imaging.append(imaging_data)
        all_labels.append(1 if label_type == 'confirmed_planet' else 0)

    logger.info(f"Successfully loaded and aligned data for {len(all_labels)} targets.")

    if not all_labels:
        logger.error("Failed to load any complete data sets. Check logs for details.")
        return None, None, None, None, None

    return (
        np.array(all_photometry_img),
        np.array(all_photometry_ts),
        np.array(all_rv),
        np.array(all_imaging),
        np.array(all_labels)
    )
