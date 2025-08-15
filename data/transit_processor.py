# data/transit_processor.py

import logging
import lightkurve as lk
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from skimage.transform import resize

# --- Setup Loggers ---
logger = logging.getLogger(__name__)
# The lightkurve library is very chatty by default. This quiets it down.
lk_logger = logging.getLogger('lightkurve')
lk_logger.setLevel(logging.WARNING)

def process_light_curve(file_path):
    """
    Processes a single FITS file into multimodal views (2D image and 1D time series).

    This function is designed to be run in a separate process for parallel execution.

    Args:
        file_path (str): The path to the FITS file.

    Returns:
        tuple: A tuple containing (local_view_image, global_view_timeseries), or (None, None) on error.
    """
    image_dim = 64
    global_view_bins = 201
    
    try:
        # Read, normalize, and flatten the light curve
        lc = lk.read(file_path)
        lc = lc.remove_nans().normalize().flatten(window_length=401)
        
        # Find the best period using Box Least Squares (BLS) and fold the light curve
        period = lc.to_periodogram(method='bls').period_at_max_power
        folded_lc = lc.fold(period=period)
        
        # --- 1. Create the Global View (1D Time Series) ---
        phase = folded_lc.phase.value
        flux = folded_lc.flux.value
        
        sort_mask = np.argsort(phase)
        phase, flux = phase[sort_mask], flux[sort_mask]
        
        bin_edges = np.linspace(phase.min(), phase.max(), global_view_bins + 1)
        bin_indices = np.digitize(phase, bin_edges)
        
        binned_flux = np.array([flux[bin_indices == i].mean() for i in range(1, len(bin_edges))])
        
        if np.isnan(binned_flux).any():
            binned_flux_df = pd.Series(binned_flux)
            binned_flux_df.interpolate(method='linear', inplace=True, limit_direction='both')
            binned_flux = binned_flux_df.values

        global_view_timeseries = binned_flux.reshape((global_view_bins, 1))

        # --- 2. Create the Local View (2D Image) ---
        phase_min, phase_max = -0.1, 0.1
        local_mask = (phase >= phase_min) & (phase <= phase_max)
        
        if np.sum(local_mask) < 10:
            phase_local, flux_local = phase, flux
        else:
            phase_local, flux_local = phase[local_mask], flux[local_mask]

        local_view_image, _, _ = np.histogram2d(phase_local, flux_local, bins=(image_dim, image_dim))
        
        local_view_image = resize(local_view_image, (image_dim, image_dim), anti_aliasing=True)
        if np.max(local_view_image) > np.min(local_view_image):
            local_view_image = (local_view_image - np.min(local_view_image)) / (np.max(local_view_image) - np.min(local_view_image))
        
        local_view_image = local_view_image.reshape((image_dim, image_dim, 1))

        return local_view_image, global_view_timeseries

    except Exception as e:
        # Suppress error logging for individual files in parallel mode to avoid spam,
        # but return None so the file is skipped.
        return None, None
