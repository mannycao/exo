# In data/dataset_generator.py

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.ndimage import shift

# ... (keep your existing functions in this file)

def balance_dataset(X, y):
    """
    Balances the dataset using SMOTE for oversampling the minority class.

    Args:
        X (np.ndarray): The feature data.
        y (np.ndarray): The labels.

    Returns:
        tuple: A tuple containing the balanced X and y arrays.
    """
    # Reshape X for SMOTE if it's 3D (e.g., (n_samples, timesteps, features))
    original_shape = X.shape
    if len(original_shape) > 2:
        X_reshaped = X.reshape(original_shape[0], -1)
    else:
        X_reshaped = X

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

    # Reshape X back to its original 3D shape if needed
    if len(original_shape) > 2:
        X_resampled = X_resampled.reshape(-1, *original_shape[1:])

    return X_resampled, y_resampleda# data/dataset_generator.py

import logging
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift

logger = logging.getLogger(__name__)

def _process_fits_file(file_path):
    """
    Reads a FITS file, cleans the light curve data, and converts it into
    a 2D image representation through phase folding and binning.
    """
    try:
        with fits.open(file_path, mode='readonly') as hdul:
            data = hdul[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX']

            # Clean up NaN/infinite values from the data
            finite_mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[finite_mask], flux[finite_mask]

            if len(time) == 0:
                logger.warning(f"No finite data found in {file_path}")
                return None

            # --- Phase Folding and Binning ---
            # NOTE: This is a simplified example.
            period = 10.0  # Placeholder period in days
            phase = (time % period) / period
            
            bins = 256
            binned_flux, _, _ = np.histogram2d(phase, flux, bins=[bins, bins])
            
            if np.max(binned_flux) > np.min(binned_flux):
                binned_flux = (binned_flux - np.min(binned_flux)) / (np.max(binned_flux) - np.min(binned_flux))
            
            return binned_flux.T

    except Exception as e:
        logger.error(f"Could not process FITS file {file_path}: {e}")
        return None

def _augment_single_image(image):
    """Applies simple augmentations to a single image."""
    if image is None or image.size == 0:
        return image
    noise = np.random.normal(0, 0.005 * np.std(image), image.shape)
    augmented_image = image + noise
    h_shift, w_shift = np.random.uniform(-1.5, 1.5, 2)
    augmented_image = shift(augmented_image, (h_shift, w_shift), mode='reflect')
    return augmented_image

def create_dataset(file_paths, labels, output_dir):
    """
    Creates a dataset from FITS files, processes them into augmented images,
    and saves them as NumPy arrays.
    """
    logger.info(f"Starting dataset creation with {len(file_paths)} files.")
    
    all_images, all_labels = [], []

    for i, file_path in enumerate(file_paths):
        label = labels[i]
        logger.info(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")

        image = _process_fits_file(file_path)
        if image is not None:
            augmented_image = _augment_single_image(image)
            all_images.append(augmented_image)
            all_labels.append(1 if label == 'confirmed' else 0)

    if not all_images:
        logger.error("No images were successfully processed. Cannot create dataset.")
        return

    X = np.array(all_images)
    y = np.array(all_labels)
    X = X[..., np.newaxis]

    X_path = os.path.join(output_dir, 'X_data.npy')
    y_path = os.path.join(output_dir, 'y_labels.npy')
    np.save(X_path, X)
    np.save(y_path, y)
    
    logger.info(f"Dataset created successfully. Data shape: {X.shape}, Labels shape: {y.shape}")
    logger.info(f"Saved data to {X_path} and labels to {y_path}")


def augment_dataset(X, y, augmentation_factor=2, shift_range=5):
    """
    Augments the dataset by creating shifted copies of the minority class samples.

    Args:
        X (np.ndarray): The feature data.
        y (np.ndarray): The labels.
        augmentation_factor (int): The number of augmented samples to create for each minority sample.
        shift_range (int): The maximum number of timesteps to shift the data.

    Returns:
        tuple: A tuple containing the augmented X and y arrays.
    """
    augmented_X = list(X)
    augmented_y = list(y)
    
    minority_class_indices = np.where(y == 1)[0]
    
    for i in minority_class_indices:
        for _ in range(augmentation_factor):
            # Create a shifted version of the light curve
            shift_amount = np.random.randint(-shift_range, shift_range)
            shifted_sample = shift(X[i], (0, shift_amount), mode='nearest') # Assuming 2D data (samples, timesteps)
            
            augmented_X.append(shifted_sample)
            augmented_y.append(y[i])
            
    return np.array(augmented_X), np.array(augmented_y)