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
    # If X is a list of multiple inputs (for multimodal), balance each one
    if isinstance(X, list):
        # Flatten and combine features for balancing strategy calculation
        X_reshaped_for_smote = np.hstack([arr.reshape(arr.shape[0], -1) for arr in X])
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_reshaped_for_smote, y)
        
        # Now, we need to reconstruct the separate input arrays
        X_balanced = []
        current_col = 0
        for arr in X:
            num_features = np.prod(arr.shape[1:])
            # Take the corresponding slice and reshape it back to its original feature shape
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

    for i, file_path in enumerate(file_paths):
        try:
            with fits.open(file_path, mode='readonly') as hdul:
                data = hdul[1].data
                time = data.field('TIME')
                flux = data.field('PDCSAP_FLUX')

                finite_mask = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[finite_mask], flux[finite_mask]

                if len(time) == 0:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: No finite data.")
                    continue
                
                # --- 1D Time-Series Preparation ---
                processed_flux = (flux - np.mean(flux)) / (np.std(flux) if np.std(flux) > 0 else 1)
                fixed_length = 2048
                if len(processed_flux) > fixed_length:
                    processed_flux = processed_flux[:fixed_length]
                else:
                    processed_flux = np.pad(processed_flux, (0, fixed_length - len(processed_flux)), 'constant')
                
                # --- 2D Image Preparation ---
                # A simple 2D representation: phase-folded plot
                period = 10.0 # Placeholder period
                phase = (time % period) / period
                binned_image, _, _ = np.histogram2d(phase, flux, bins=image_size[0])
                # Resize and normalize
                img = resize(binned_image, image_size, anti_aliasing=True)
                img = (img - np.min(img)) / (np.max(img) - np.min(img) if np.max(img) > np.min(img) else 1)

                all_timeseries.append(processed_flux)
                all_images.append(img)
                all_labels.append(1 if labels[i] == 'confirmed_planet' else 0)

        except Exception as e:
            logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.")

    if not all_timeseries:
        logger.error("CRITICAL: No files were successfully processed.")
        return

    # Convert to NumPy arrays and add channel dimensions
    X_ts = np.array(all_timeseries)[..., np.newaxis]
    X_img = np.array(all_images)[..., np.newaxis]
    y = np.array(all_labels)

    # Save all three arrays
    np.save(os.path.join(output_dir, 'X_timeseries.npy'), X_ts)
    np.save(os.path.join(output_dir, 'X_images.npy'), X_img)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y)
    
    logger.info(f"Multimodal dataset created successfully with {len(y)} samples.")