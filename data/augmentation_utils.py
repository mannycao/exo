
# data/augmentation_utils.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_transit(length=2048, transit_depth=0.1, transit_duration=50, transit_period=1000, noise_level=0.01):
    """Generates a synthetic transit light curve with a more realistic shape."""
    time = np.arange(length)
    flux = np.ones(length)
    
    # Improved transit shape (U-shape)
    half_duration = transit_duration / 2
    transit_points = np.abs(time - (length / 2)) < half_duration
    limb_darkening = 1 - (np.abs(time[transit_points] - (length / 2)) / half_duration)**2
    flux[transit_points] = 1 - transit_depth * limb_darkening
    
    # Add noise
    flux += np.random.normal(0, noise_level, length)
    return flux

def generate_eclipsing_binary(length=2048, primary_depth=0.5, secondary_depth=0.2, duration=100, noise_level=0.01):
    """Generates a synthetic eclipsing binary light curve."""
    time = np.arange(length)
    flux = np.ones(length)
    
    # Primary eclipse
    primary_points = np.abs(time - (length / 4)) < duration / 2
    flux[primary_points] = 1 - primary_depth
    
    # Secondary eclipse
    secondary_points = np.abs(time - (3 * length / 4)) < duration / 2
    flux[secondary_points] = 1 - secondary_depth
    
    # Add noise
    flux += np.random.normal(0, noise_level, length)
    return flux

def augment_data(X_list, y, augmentation_factor=2):
    """Augments the dataset with synthetic transits and eclipsing binaries for multimodal data."""
    X_img_original, X_ts_original = X_list[0], X_list[1]

    X_img_augmented = list(X_img_original)
    X_ts_augmented = list(X_ts_original)
    y_augmented = list(y)

    # Get target shapes for synthetic data
    ts_length = X_ts_original.shape[1] # e.g., 2048
    img_height, img_width = X_img_original.shape[1], X_img_original.shape[2] # e.g., 64, 64
    img_channels = X_img_original.shape[3] # e.g., 1

    positive_indices = np.where(y == 1)[0]

    for _ in range(augmentation_factor):
        for i in positive_indices:
            # Augment existing positive examples by duplicating them
            X_img_augmented.append(X_img_original[i])
            X_ts_augmented.append(X_ts_original[i])
            y_augmented.append(1)

            # Add synthetic transit (positive example)
            synthetic_ts = generate_synthetic_transit(length=ts_length)
            # Phase-fold synthetic time series to create image
            # Need a period for phase folding. Random for now.
            synthetic_img_2d = advanced_phase_folding(np.arange(ts_length), synthetic_ts, period=np.random.uniform(1, 100), n_bins=img_height * img_width)
            synthetic_img = synthetic_img_2d.reshape(img_height, img_width, img_channels) # Reshape and add channel

            X_img_augmented.append(synthetic_img)
            X_ts_augmented.append(synthetic_ts)
            y_augmented.append(1)

            # Add eclipsing binary as a false positive
            eclipsing_ts = generate_eclipsing_binary(length=ts_length)
            eclipsing_img_2d = advanced_phase_folding(np.arange(ts_length), eclipsing_ts, period=np.random.uniform(1, 100), n_bins=img_height * img_width)
            eclipsing_img = eclipsing_img_2d.reshape(img_height, img_width, img_channels) # Reshape and add channel

            X_img_augmented.append(eclipsing_img)
            X_ts_augmented.append(eclipsing_ts)
            y_augmented.append(0)

    return np.array(X_img_augmented), np.array(X_ts_augmented), np.array(y_augmented)

def advanced_phase_folding(time, flux, period, n_bins=128):
    """
    Performs phase folding with binning and averaging to create a cleaner 2D representation.
    """
    phase = (time % period) / period
    binned_flux = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(len(phase)):
        bin_index = int(phase[i] * n_bins)
        if 0 <= bin_index < n_bins:
            binned_flux[bin_index] += flux[i]
            bin_counts[bin_index] += 1
            
    # Avoid division by zero
    bin_counts[bin_counts == 0] = 1
    
    return binned_flux / bin_counts
