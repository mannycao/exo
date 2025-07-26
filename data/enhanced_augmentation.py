"""
Enhanced data augmentation for exoplanet detection using real data from TESS and Kepler.
"""

import os
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Import your project modules
import config
from data.light_curve_processor import preprocess_light_curve, extract_transit_features
from data.data_fetcher import fetch_kepler_data, fetch_exoplanet_labels, download_product

logger = logging.getLogger(__name__)


def fetch_tess_data(max_records=50, use_cache=True):
    """
    Fetches light curve data from TESS mission with local caching.
    
    Args:
        max_records: Maximum number of records to fetch
        use_cache: Whether to use cached data
    
    Returns:
        astropy.table.Table: Table of TESS observations
    """
    from astroquery.mast import Observations
    
    cache_file = os.path.join(config.METADATA_DIR, "tess_observation_list.pkl")
    
    # Check if we have cached data
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached TESS observation list")
        obs_table = pd.read_pickle(cache_file)
        return obs_table
    
    logger.info(f"Fetching TESS observations from MAST")
    obs_table = Observations.query_criteria(
        obs_collection='TESS',
        dataproduct_type="timeseries",
        objectname="exoplanet"
    )
    
    # Cache results for future use
    pd.DataFrame(obs_table).to_pickle(cache_file)
    
    return obs_table[:max_records]  # Limit for development/testing


def download_tess_light_curves(obs_table, max_records=20, use_cache=True):
    """
    Downloads TESS light curve files based on observation table.
    
    Args:
        obs_table: Table of TESS observations
        max_records: Maximum number of records to download
        use_cache: Whether to use cached files
    
    Returns:
        list: Paths to downloaded light curve files
    """
    from astroquery.mast import Observations
    
    light_curve_files = []
    
    # Limit the number of observations for development/testing
    for i, obs in enumerate(obs_table[:max_records]):
        try:
            data_products = Observations.get_product_list(obs)
            
            # Filter for light curve products
            light_curve_products = [p for p in data_products if 'LC' in p['dataURI']]
            
            if light_curve_products:
                # Choose the first light curve product
                file_path = download_product(light_curve_products[0], use_cache)
                if file_path:
                    light_curve_files.append(file_path)
                    logger.info(f"Downloaded TESS light curve: {file_path}")
        except Exception as e:
            logger.error(f"Error with TESS observation {obs.get('obsid', i)}: {e}")
    
    return light_curve_files


def augment_transit_segment(segment, augmentation_type='noise'):
    """
    Apply augmentation to a transit segment.
    
    Args:
        segment: Transit segment time series
        augmentation_type: Type of augmentation to apply
            - 'noise': Add Gaussian noise
            - 'jitter': Add time jitter
            - 'scale': Scale transit depth
            - 'smooth': Apply smoothing
            - 'random': Apply random combination
    
    Returns:
        numpy.ndarray: Augmented transit segment
    """
    augmented = segment.copy()
    
    if augmentation_type == 'random':
        # Choose a random augmentation type
        augmentation_type = np.random.choice([
            'noise', 'jitter', 'scale', 'smooth'
        ])
    
    if augmentation_type == 'noise':
        # Add Gaussian noise with random amplitude
        noise_level = np.random.uniform(0.05, 0.2) * np.std(segment)
        augmented += np.random.normal(0, noise_level, size=len(segment))
    
    elif augmentation_type == 'jitter':
        # Add random jitter to time coordinates
        x = np.arange(len(segment))
        jitter = np.random.normal(0, 1, size=len(segment))
        x_jittered = x + jitter
        x_jittered = np.clip(x_jittered, 0, len(segment)-1)
        
        # Interpolate to get values at jittered positions
        interp_func = interp1d(x, segment, bounds_error=False, fill_value='extrapolate')
        augmented = interp_func(x_jittered)
        
    elif augmentation_type == 'scale':
        # Scale the transit depth
        # First, estimate the baseline level
        baseline = np.median(segment)
        
        # Find the approximate transit depth
        min_val = np.min(segment)
        depth = baseline - min_val
        
        # Scale the depth by a random factor
        scale_factor = np.random.uniform(0.7, 1.3)
        
        # Apply scaling only to the transit part
        transit_mask = segment < (baseline - depth * 0.3)
        if np.any(transit_mask):
            # Calculate how much to adjust the transit points
            adjustment = (scale_factor - 1) * (segment[transit_mask] - baseline)
            augmented[transit_mask] = segment[transit_mask] + adjustment
    
    elif augmentation_type == 'smooth':
        # Apply random smoothing
        sigma = np.random.uniform(0.5, 2.0)
        augmented = gaussian_filter1d(segment, sigma)
    
    return augmented


def create_mixed_domain_features(time_series, image):
    """
    Create enhanced features that combine time series and image domain information.
    
    Args:
        time_series: Transit time series data
        image: Image representation of transit
    
    Returns:
        tuple: (enhanced_time_series, enhanced_image)
    """
    # Calculate time domain features
    # Apply LombScargle to extract frequency domain info
    freq, power = LombScargle(np.arange(len(time_series)), time_series).autopower()
    
    # Get top frequencies
    sorted_idx = np.argsort(power)[::-1]
    top_freqs_idx = sorted_idx[:3]  # Top 3 frequencies
    top_powers = power[top_freqs_idx]
    
    # Calculate statistical features
    mean = np.mean(time_series)
    std = np.std(time_series)
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    
    # Create enhanced time series with additional features
    enhanced_ts = time_series.copy()
    
    # Add marker for frequency content
    for i, (f_idx, p) in enumerate(zip(top_freqs_idx, top_powers)):
        # Add subtle markers at positions corresponding to top frequencies
        pos = int((f_idx / len(freq)) * len(time_series))
        if pos < len(time_series):
            enhanced_ts[pos] = enhanced_ts[pos] * (1 + 0.01 * (i+1))
    
    # Enhanced image with time domain features
    enhanced_img = image.copy()
    
    # Add time domain information to specific rows
    if enhanced_img.shape[0] >= 5:
        # Add statistical markers in specific rows
        # Row 0: mean-related info
        enhanced_img[0, :] = enhanced_img[0, :] * (1 + 0.02 * mean)
        
        # Row 1: std-related info
        enhanced_img[1, :] = enhanced_img[1, :] * (1 + 0.05 * std)
    
    return enhanced_ts, enhanced_img


def generate_realistic_synthetic_transit(length=100, transit_width_range=(5, 15), 
                                         depth_range=(0.005, 0.05), noise_level_range=(0.001, 0.02)):
    """
    Generate realistic synthetic transit light curves based on real transit shapes.
    
    Args:
        length: Length of the time series
        transit_width_range: Range of transit widths in samples
        depth_range: Range of transit depths
        noise_level_range: Range of noise levels
    
    Returns:
        numpy.ndarray: Synthetic transit light curve
    """
    time = np.arange(length)
    flux = np.ones(length)
    
    # Set transit parameters
    transit_center = length // 2
    transit_width = np.random.randint(transit_width_range[0], transit_width_range[1])
    transit_depth = np.random.uniform(depth_range[0], depth_range[1])
    
    # Create transit shape with limb darkening
    for i in range(length):
        # Distance from transit center in units of width
        d = 2 * abs(i - transit_center) / transit_width
        
        # Points inside the transit
        if d < 1:
            # Apply limb darkening effect
            if d < 0.25:
                # Flat bottom
                flux[i] = 1 - transit_depth
            else:
                # Limb transition with quartic limb darkening
                # This creates a smoother, more realistic transit shape
                limb_factor = (d - 0.25) / 0.75
                darkening = 1 - (1 - limb_factor**2)**2
                flux[i] = 1 - transit_depth * (1 - darkening)
    
    # Add stellar variability (optional)
    if np.random.random() < 0.7:
        period = np.random.uniform(3, length/2)
        amplitude = np.random.uniform(0.0005, 0.003)
        flux += amplitude * np.sin(2 * np.pi * time / period)
    
    # Add noise
    noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])
    flux += np.random.normal(0, noise_level, length)
    
    return flux


def augment_and_balance_dataset(X_image, X_timeseries, y, 
                               augmentation_factor=2,
                               balance_ratio=0.5):
    """
    Augment and balance the dataset using real data characteristics.
    
    Args:
        X_image: Image features
        X_timeseries: Time series features
        y: Labels
        augmentation_factor: Factor by which to augment positive examples
        balance_ratio: Target ratio of positive to negative examples
    
    Returns:
        tuple: (augmented_X_image, augmented_X_timeseries, augmented_y)
    """
    # Count positive and negative examples
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    total_count = len(y)
    
    logger.info(f"Original dataset: {positive_count} positive, {negative_count} negative examples")
    
    # Indices of positive and negative examples
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    
    # Initialize lists for augmented data
    augmented_images = list(X_image)
    augmented_timeseries = list(X_timeseries)
    augmented_labels = list(y)
    
    # Augment positive examples
    for _ in range(augmentation_factor - 1):
        for idx in positive_indices:
            # Apply random augmentation
            aug_type = np.random.choice(['noise', 'jitter', 'scale', 'smooth'])
            aug_ts = augment_transit_segment(X_timeseries[idx], aug_type)
            
            # Create updated image representation
            aug_img = X_image[idx].copy()
            
            # Apply slight variations to image
            aug_img += np.random.normal(0, 0.05, aug_img.shape)
            
            # Apply domain-specific feature enhancement
            aug_ts, aug_img = create_mixed_domain_features(aug_ts, aug_img)
            
            # Add to augmented datasets
            augmented_images.append(aug_img)
            augmented_timeseries.append(aug_ts)
            augmented_labels.append(1)  # Positive example
    
    # Count after augmentation
    aug_positive_count = np.sum(np.array(augmented_labels) == 1)
    aug_negative_count = np.sum(np.array(augmented_labels) == 0)
    
    # Balance the dataset if needed
    current_ratio = aug_positive_count / len(augmented_labels)
    
    # Add synthetic negative examples if needed to achieve balance_ratio
    if current_ratio > balance_ratio:
        # Need to add more negative examples
        synthetic_negative_count = int(aug_positive_count / balance_ratio) - len(augmented_labels)
        
        logger.info(f"Adding {synthetic_negative_count} synthetic negative examples")
        
        for _ in range(synthetic_negative_count):
            # Create synthetic non-transit light curve
            length = X_timeseries[0].shape[0]
            
            # Generate realistic non-transit data (stellar variability with noise)
            time = np.arange(length)
            flux = np.ones(length)
            
            # Add stellar variability
            period1 = np.random.uniform(length/10, length/2)
            amplitude1 = np.random.uniform(0.001, 0.01)
            flux += amplitude1 * np.sin(2 * np.pi * time / period1)
            
            # Sometimes add a second variability component
            if np.random.random() < 0.5:
                period2 = np.random.uniform(length/5, length/1.5)
                amplitude2 = np.random.uniform(0.0005, 0.005)
                flux += amplitude2 * np.sin(2 * np.pi * time / period2)
            
            # Add noise
            noise_level = np.random.uniform(0.001, 0.02)
            flux += np.random.normal(0, noise_level, length)
            
            # Create image representation
            img = np.zeros((X_image[0].shape[0], X_image[0].shape[1]))
            for j in range(img.shape[0]):
                img[j, :] = flux
            
            augmented_timeseries.append(flux)
            augmented_images.append(img)
            augmented_labels.append(0)  # Negative example
    
    # Convert to numpy arrays
    augmented_X_image = np.array(augmented_images)
    augmented_X_timeseries = np.array(augmented_timeseries)
    augmented_y = np.array(augmented_labels)
    
    # Final counts
    final_positive = np.sum(augmented_y == 1)
    final_negative = np.sum(augmented_y == 0)
    
    logger.info(f"Augmented dataset: {final_positive} positive, {final_negative} negative examples")
    
    return augmented_X_image, augmented_X_timeseries, augmented_y