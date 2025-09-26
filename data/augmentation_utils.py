# data/augmentation_utils.py

import numpy as np
import logging
from astropy.timeseries import LombScargle
from skimage.transform import resize

logger = logging.getLogger(__name__)

def generate_synthetic_transit(length=2048, transit_depth=0.1, transit_duration=50, noise_level=0.01):
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
    return time, flux

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
    return time, flux

def augment_data(X_list, y, pipeline_results_raw, augmentation_factor=2):
    """Augments the dataset with synthetic transits and eclipsing binaries for multimodal data."""
    X_img_original, X_ts_original, X_features_original = X_list[0], X_list[1], X_list[2]

    X_img_augmented = list(X_img_original)
    X_ts_augmented = list(X_ts_original)
    X_features_augmented = list(X_features_original)
    y_augmented = list(y)
    pipeline_results_augmented = list(pipeline_results_raw) # Initialize augmented pipeline results

    # Get target shapes for synthetic data
    ts_length = X_ts_original.shape[1] # e.g., 2048
    img_height, img_width = X_img_original.shape[1], X_img_original.shape[2] # e.g., 64, 64
    img_channels = X_img_original.shape[3] # e.g., 1
    feature_length = X_features_original.shape[1] # e.g., 4096

    positive_indices = np.where(y == 1)[0]

    for _ in range(augmentation_factor):
        for i in positive_indices:
            # Augment existing positive examples by duplicating them
            X_img_augmented.append(X_img_original[i])
            X_ts_augmented.append(X_ts_original[i])
            X_features_augmented.append(X_features_original[i])
            y_augmented.append(1)
            pipeline_results_augmented.append(pipeline_results_raw[i]) # Duplicate corresponding result

            # Add synthetic transit (positive example)
            synthetic_time, synthetic_ts = generate_synthetic_transit(length=ts_length)
            synthetic_img_2d = advanced_phase_folding(synthetic_time, synthetic_ts, period=np.random.uniform(1, 100), n_bins=img_height * img_width)
            synthetic_img = synthetic_img_2d.reshape(img_height, img_width, img_channels)

            frequency, power = LombScargle(synthetic_time, synthetic_ts).autopower()
            autocorr = np.correlate(synthetic_ts, synthetic_ts, mode='full')[len(synthetic_ts)-1:]
            power = resize(power, (ts_length,), preserve_range=True, anti_aliasing=False)
            autocorr = resize(autocorr, (ts_length,), preserve_range=True, anti_aliasing=False)
            synthetic_features = np.hstack([power, autocorr])
            synthetic_features = resize(synthetic_features, (feature_length,), preserve_range=True, anti_aliasing=False)

            X_img_augmented.append(synthetic_img)
            X_ts_augmented.append(synthetic_ts)
            X_features_augmented.append(synthetic_features)
            y_augmented.append(1)
            pipeline_results_augmented.append({ # Dummy entry for synthetic transit
                'file_path': 'synthetic_transit',
                'success': True,
                'transit_count': 1,
                'periodicity': np.random.uniform(1, 100),
                'planet_properties': {'radius_earth': np.random.uniform(1, 10), 'orbital_period_days': np.random.uniform(1, 100), 'semi_major_axis_au': np.random.uniform(0.1, 1), 'equilibrium_temp_k': np.random.uniform(200, 2000)},
                'light_curve_plot_path': 'N/A'
            })

            # Add eclipsing binary as a false positive
            eclipsing_time, eclipsing_ts = generate_eclipsing_binary(length=ts_length)
            eclipsing_img_2d = advanced_phase_folding(eclipsing_time, eclipsing_ts, period=np.random.uniform(1, 100), n_bins=img_height * img_width)
            eclipsing_img = eclipsing_img_2d.reshape(img_height, img_width, img_channels)

            frequency, power = LombScargle(eclipsing_time, eclipsing_ts).autopower()
            autocorr = np.correlate(eclipsing_ts, eclipsing_ts, mode='full')[len(eclipsing_ts)-1:]
            power = resize(power, (ts_length,), preserve_range=True, anti_aliasing=False)
            autocorr = resize(autocorr, (ts_length,), preserve_range=True, anti_aliasing=False)
            eclipsing_features = np.hstack([power, autocorr])
            eclipsing_features = resize(eclipsing_features, (feature_length,), preserve_range=True, anti_aliasing=False)

            X_img_augmented.append(eclipsing_img)
            X_ts_augmented.append(eclipsing_ts)
            X_features_augmented.append(eclipsing_features)
            y_augmented.append(0)
            pipeline_results_augmented.append({ # Dummy entry for eclipsing binary
                'file_path': 'eclipsing_binary',
                'success': True,
                'transit_count': 1,
                'periodicity': np.random.uniform(1, 100),
                'planet_properties': {'radius_earth': np.random.uniform(1, 10), 'orbital_period_days': np.random.uniform(1, 100), 'semi_major_axis_au': np.random.uniform(0.1, 1), 'equilibrium_temp_k': np.random.uniform(200, 2000)},
                'light_curve_plot_path': 'N/A'
            })

    return np.array(X_img_augmented), np.array(X_ts_augmented), np.array(X_features_augmented), np.array(y_augmented), pipeline_results_augmented

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