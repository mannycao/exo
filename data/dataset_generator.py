# data/dataset_generator.py

import logging
import os
import numpy as np
from astropy.io import fits
from imblearn.over_sampling import SMOTE
from skimage.transform import resize

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


def create_dataset(file_paths, labels, output_dir, image_size=(64, 64)):
    """
    Creates a multimodal dataset (1D time-series and 2D image) from FITS files.
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
                flux = data.field('PDCSAP_FLUX')

                finite_mask = np.isfinite(flux)
                flux = flux[finite_mask]

                if len(flux) < 100: # Ensure there's enough data
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points.")
                    continue
                
                # --- 1D Time-Series Preparation ---
                # Normalize the entire light curve first
                flux_norm = (flux - np.median(flux)) / np.std(flux)
                
                # For simplicity, we'll take a center chunk of the light curve
                # A more advanced approach would use the transit detection logic
                start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
                segment = flux_norm[start : start + FIXED_LENGTH]

                # Pad if segment is shorter than FIXED_LENGTH
                if len(segment) < FIXED_LENGTH:
                    segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
                
                all_timeseries.append(segment)
                
                # --- 2D Image Preparation (IMPROVED METHOD) ---
                # Convert the 1D segment directly into a 2D representation
                # This preserves the transit shape information
                img_1d = segment[:image_size[0] * image_size[1]] # Ensure it fits
                if len(img_1d) < image_size[0] * image_size[1]:
                    img_1d = np.pad(img_1d, (0, image_size[0] * image_size[1] - len(img_1d)), 'constant', constant_values=0)
                
                img_2d = img_1d.reshape(image_size)

                # Normalize image to [0, 1] for the CNN
                img_norm = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
                
                all_images.append(img_norm)
                all_labels.append(1 if labels[i] == 'confirmed_planet' else 0)

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