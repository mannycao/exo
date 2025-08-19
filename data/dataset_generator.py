# data/dataset_generator.py

import logging
import os
import numpy as np
import pandas as pd
import re
from astropy.io import fits
from skimage.transform import resize
from data.augmentation_utils import advanced_phase_folding
from concurrent.futures import ThreadPoolExecutor # New import
from functools import partial # New import

logger = logging.getLogger(__name__)

def process_single_file(file_info, metadata_df, image_size, FIXED_LENGTH):
    """Helper function to process a single light curve file."""
    file_path, current_label = file_info
    
    try:
        with fits.open(file_path, mode='readonly') as hdul:
            data = hdul[1].data
            time = data.field('TIME')
            flux = data.field('PDCSAP_FLUX')

            finite_mask = np.isfinite(flux) & np.isfinite(time)
            time = time[finite_mask]
            flux = flux[finite_mask]

            if len(flux) < 100:
                logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points.")
                return None

            period = None
            t0 = 0.0

            if current_label == 'confirmed_planet':
                match = re.search(r'kplr(\d+)', os.path.basename(file_path))
                if match:
                    kepler_id_num = int(match.group(1))
                    metadata_df_copy = metadata_df.copy()
                    metadata_df_copy['pl_name_num_id'] = metadata_df_copy['pl_name'].apply(
                        lambda x: int(re.search(r'Kepler-(\d+)', x).group(1)) if pd.notna(x) and re.search(r'Kepler-(\d+)', x) else None
                    )
                    matching_rows = metadata_df_copy[metadata_df_copy['pl_name_num_id'] == kepler_id_num]
                    if not matching_rows.empty:
                        period = matching_rows['pl_orbper'].iloc[0]
                        if pd.isna(period):
                            period = np.random.uniform(1, 100)
                            logger.warning(f"Period for Kepler ID {kepler_id_num} is NaN. Using random period: {period:.2f}")
                    else:
                        period = np.random.uniform(1, 100)
                        logger.warning(f"No period found for Kepler ID {kepler_id_num} in metadata. Using random period: {period:.2f}")
                else:
                    period = np.random.uniform(1, 100)
                    logger.warning(f"Could not parse Kepler ID from {os.path.basename(file_path)}. Using random period: {period:.2f}")
            elif current_label == 'false_positive':
                period = np.random.uniform(1, 100)
            
            if period is None:
                period = np.random.uniform(1, 100)
                logger.warning(f"Period is None after all attempts. Using random period: {period:.2f}")

            flux_norm = (flux - np.median(flux)) / np.std(flux)
            start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
            segment = flux_norm[start : start + FIXED_LENGTH]
            if len(segment) < FIXED_LENGTH:
                segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
            
            if len(time) != len(flux):
                logger.warning(f"Time and flux length mismatch for {os.path.basename(file_path)}. Skipping phase folding for 2D image.")
                img_1d = segment[:image_size[0] * image_size[1]]
                if len(img_1d) < image_size[0] * image_size[1]:
                    img_1d = np.pad(img_1d, (0, image_size[0] * image_size[1] - len(img_1d)), 'constant', constant_values=0)
                img_2d = img_1d.reshape(image_size)
            else:
                binned_flux = advanced_phase_folding(time, flux, period, n_bins=image_size[0] * image_size[1])
                img_2d = binned_flux.reshape(image_size)

            img_norm = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
            
            return segment, img_norm, (1 if current_label == 'confirmed_planet' else 0)

    except Exception as e:
        logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.")
        return None

def create_dataset(file_paths, labels, output_dir, metadata_df, image_size=(64, 64), max_workers=os.cpu_count()):
    """
    Creates a multimodal dataset (1D time-series and 2D image) from FITS files.
    Includes phase folding for 2D image representation.
    Parallelized using ThreadPoolExecutor.
    """
    logger.info(f"Starting multimodal dataset creation with {len(file_paths)} files using {max_workers} workers.")
    
    all_timeseries = []
    all_images = []
    all_labels = []
    
    FIXED_LENGTH = 2048 # Define a fixed length for time-series segments

    file_info_list = [(fp, lbl) for fp, lbl in zip(file_paths, labels)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use partial to pass fixed arguments to the helper function
        process_func = partial(process_single_file, metadata_df=metadata_df, image_size=image_size, FIXED_LENGTH=FIXED_LENGTH)
        
        # Map the function to the list of file info
        results = list(executor.map(process_func, file_info_list))

    for result in results:
        if result is not None:
            segment, img_norm, label = result
            all_timeseries.append(segment)
            all_images.append(img_norm)
            all_labels.append(label)

    if not all_timeseries:
        logger.error("CRITICAL: No files were successfully processed.")
        return

    # Convert to NumPy arrays and add channel dimensions
    X_ts = np.array(all_timeseries)
    X_img = np.array(all_images)[..., np.newaxis]
    y = np.array(all_labels)

    np.save(os.path.join(output_dir, 'X_timeseries.npy'), X_ts)
    np.save(os.path.join(output_dir, 'X_images.npy'), X_img)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y)
    
    logger.info(f"Multimodal dataset created successfully with {len(y)} samples.")