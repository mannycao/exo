# data/dataset_generator.py

import logging
import os
import numpy as np
import pandas as pd
import re
from astropy.io import fits
from skimage.transform import resize
from data.augmentation_utils import advanced_phase_folding, augment_data
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from astropy.timeseries import LombScargle
import config

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

            period = np.random.uniform(1, 100)

            flux_norm = (flux - np.median(flux)) / np.std(flux)
            start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
            segment = flux_norm[start : start + FIXED_LENGTH]
            if len(segment) < FIXED_LENGTH:
                segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
            
            # Feature Engineering
            frequency, power = LombScargle(time, flux).autopower()
            autocorr = np.correlate(flux_norm, flux_norm, mode='full')[len(flux_norm)-1:]

            # Resize features to a fixed length
            power = resize(power, (FIXED_LENGTH,), preserve_range=True, anti_aliasing=False)
            autocorr = resize(autocorr, (FIXED_LENGTH,), preserve_range=True, anti_aliasing=False)

            feature_vector = np.hstack([power, autocorr])

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
            
            return segment, img_norm, feature_vector, (1 if current_label == 'confirmed_planet' else 0)

    except Exception as e:
        logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.")
        return None

def create_dataset(file_paths, labels, output_dir, metadata_df, image_size=(64, 64), max_workers=os.cpu_count()):
    """
    Creates a multimodal dataset (1D time-series, 2D image, and engineered features) from FITS files.
    """
    logger.info(f"Starting multimodal dataset creation with {len(file_paths)} files using {max_workers} workers.")
    
    all_timeseries = []
    all_images = []
    all_features = []
    all_labels = []
    
    FIXED_LENGTH = 2048 # Define a fixed length for time-series segments

    file_info_list = [(fp, lbl) for fp, lbl in zip(file_paths, labels)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_single_file, metadata_df=metadata_df, image_size=image_size, FIXED_LENGTH=FIXED_LENGTH)
        
        results = list(executor.map(process_func, file_info_list))

    for result in results:
        if result is not None:
            segment, img_norm, feature_vector, label = result
            all_timeseries.append(segment)
            all_images.append(img_norm)
            all_features.append(feature_vector)
            all_labels.append(label)

    if not all_timeseries:
        logger.error("CRITICAL: No files were successfully processed.")
        return

    X_ts = np.array(all_timeseries)
    X_img = np.array(all_images)[..., np.newaxis]
    X_features = np.array(all_features)
    y = np.array(all_labels)

    # Augment data
    X_img_aug, X_ts_aug, X_features_aug, y_aug = augment_data([X_img, X_ts, X_features], y, augmentation_factor=config.AUGMENTATION_FACTOR)

    np.save(os.path.join(output_dir, 'X_timeseries.npy'), X_ts_aug)
    np.save(os.path.join(output_dir, 'X_images.npy'), X_img_aug)
    np.save(os.path.join(output_dir, 'X_features.npy'), X_features_aug)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y_aug)
    
    logger.info(f"Multimodal dataset created and augmented successfully with {len(y_aug)} samples.)")
