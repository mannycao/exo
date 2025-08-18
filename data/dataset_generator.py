# data/dataset_generator.py

import logging
import os
import numpy as np
import pandas as pd # Added import
import re # Added for regex
from astropy.io import fits
from imblearn.over_sampling import SMOTE
from skimage.transform import resize
from data.augmentation_utils import advanced_phase_folding

logger = logging.getLogger(__name__)

def balance_dataset(X, y):
    """Balances the dataset using SMOTE."""
    if len(np.unique(y)) < 2:
        logger.error(f"Cannot balance data with only one class.")
        return X, y
    
    if isinstance(X, list):
        # Flatten and combine features for balancing strategy calculation
        X_reshaped_for_smote = np.hstack([arr.reshape(arr.shape[0], -1) for arr in X])
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_reshaped_for_smote, y)
        
        # Reconstruct the separate input arrays
        X_balanced = []
        current_col = 0
        for arr in X:
            num_features = np.prod(arr.shape[1:])
            balanced_arr_flat = X_res[:, current_col:current_col + num_features]
            X_balanced.append(balanced_arr_flat.reshape(len(y_res), *arr.shape[1:]))
            current_col += num_features
        return X_balanced, y_res
    else: # Standard single input
        X_reshaped = X.reshape(X.shape[0], -1)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_reshaped, y)
        return X_res.reshape(len(y_res), *X.shape[1:]), y_res


def phase_fold(time, flux, period, t0):
    """
    Phase-folds a light curve given time, flux, period, and reference epoch.
    """
    if period <= 0:
        raise ValueError("Period must be positive.")
    
    phase = ((time - t0) % period) / period
    return phase, flux

def create_dataset(file_paths, labels, output_dir, metadata_df, image_size=(64, 64)):
    """
    Creates a multimodal dataset (1D time-series and 2D image) from FITS files.
    Includes phase folding for 2D image representation.
    """
    logger.info(f"Starting multimodal dataset creation with {len(file_paths)} files.")
    
    all_timeseries = []
    all_images = []
    all_labels = []
    
    FIXED_LENGTH = 2048 # Define a fixed length for time-series segments

    for i, file_path in enumerate(file_paths):
        try:
            with fits.open(file_path, mode='readonly') as hdul:
                data = hdul[1].data
                time = data.field('TIME') # Assuming TIME field exists
                flux = data.field('PDCSAP_FLUX')

                finite_mask = np.isfinite(flux) & np.isfinite(time)
                time = time[finite_mask]
                flux = flux[finite_mask]

                if len(flux) < 100: # Ensure there's enough data
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points.")
                    continue
                
                # --- Determine Period and t0 for Phase Folding ---
                current_label = labels[i]
                period = None
                t0 = 0.0 # Default t0 for now

                if current_label == 'confirmed_planet':
                    # Extract Kepler ID from filename (e.g., kplr005966154-2009131105131_llc.fits)
                    match = re.search(r'kplr(\d+)', os.path.basename(file_path))
                    if match:
                        kepler_id_num = int(match.group(1))
                        
                        # Create a temporary column in metadata_df for numerical Kepler ID from pl_name
                        # This is to handle cases like "Kepler-61 b" where we need to extract "61"
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

                # --- 1D Time-Series Preparation (from original light curve) ---
                flux_norm = (flux - np.median(flux)) / np.std(flux)
                start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
                segment = flux_norm[start : start + FIXED_LENGTH]
                if len(segment) < FIXED_LENGTH:
                    segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
                all_timeseries.append(segment)
                
                # --- 2D Image Preparation (Phase-folded) ---
                # Ensure time array is also processed for finite values and matches flux length
                if len(time) != len(flux):
                    logger.warning(f"Time and flux length mismatch for {os.path.basename(file_path)}. Skipping phase folding for 2D image.")
                    # Fallback to original 1D segment reshaping if phase folding is not possible
                    img_1d = segment[:image_size[0] * image_size[1]]
                    if len(img_1d) < image_size[0] * image_size[1]:
                        img_1d = np.pad(img_1d, (0, image_size[0] * image_size[1] - len(img_1d)), 'constant', constant_values=0)
                    img_2d = img_1d.reshape(image_size)
                else:
                    # Use advanced phase folding to create a binned and averaged 2D representation
                    binned_flux = advanced_phase_folding(time, flux, period, n_bins=image_size[0] * image_size[1])
                    img_2d = binned_flux.reshape(image_size)

                # Normalize image to [0, 1] for the CNN
                img_norm = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
                
                all_images.append(img_norm)
                all_labels.append(1 if current_label == 'confirmed_planet' else 0)

        except Exception as e:
            logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.")

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