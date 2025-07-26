# data/light_curve_processor.py

import logging
import numpy as np
from scipy.signal import savgol_filter
from astropy.io import fits

logger = logging.getLogger(__name__)

def preprocess_light_curve(file_path, detrend_window_days=2.0, savgol_polyorder=2):
    """
    Loads a FITS file, cleans it, and applies a Savitzky-Golay filter to detrend it.

    Args:
        file_path (str): The path to the FITS file.
        detrend_window_days (float): The window size for the detrending filter in days.
        savgol_polyorder (int): The polynomial order for the Savitzky-Golay filter.

    Returns:
        tuple: A tuple of (time, detrended_flux, original_flux) or (None, None, None) if failed.
    """
    try:
        with fits.open(file_path, mode='readonly') as hdul:
            data = hdul[1].data
            time = data.field('TIME')
            flux = data.field('PDCSAP_FLUX')

            # Clean up NaN/infinite values from the data
            finite_mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[finite_mask], flux[finite_mask]

            if len(time) == 0:
                logger.warning(f"Skipping {file_path}: No finite data found after cleaning.")
                return None, None, None
            
            # --- Detrending Logic ---
            # Calculate the window length in samples (must be an odd number)
            time_interval = np.median(np.diff(time))
            window_length = int(detrend_window_days / time_interval)
            if window_length % 2 == 0:
                window_length += 1 # Ensure the window is odd

            # The filter window must be smaller than the data length
            if window_length >= len(flux):
                logger.warning(f"Detrending window is too large for the light curve in {file_path}. Skipping detrending.")
                trend = np.median(flux) # Fallback to a simple median
            else:
                # Apply the Savitzky-Golay filter to find the long-term trend
                trend = savgol_filter(flux, window_length, savgol_polyorder)

            # Subtract the trend to get the detrended flux
            detrended_flux = flux - trend
            # Normalize the detrended flux to have a standard deviation of 1
            detrended_flux = (detrended_flux - np.median(detrended_flux)) / np.std(detrended_flux)

            return time, detrended_flux, flux
    except Exception as e:
        logger.error(f"Failed to preprocess FITS file {file_path}: {e}")
        return None, None, None