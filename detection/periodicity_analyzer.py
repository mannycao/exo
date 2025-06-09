"""
Functions for analyzing periodicity in light curve data,
including Lomb-Scargle periodograms and phase-folding.
"""
import logging
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

logger = logging.getLogger(__name__)

def analyze_periodicity(time, flux, transit_info):
    """
    Analyzes the periodicity of detected transits using a Lomb-Scargle periodogram.
    Now safely accesses transit_info properties.
    """
    if transit_info is None or not transit_info.get('peak_indices'):
        logger.debug("analyze_periodicity: No transit information provided.")
        return None

    # Use the times of the detected transits to look for periodicity
    transit_times = np.asarray(transit_info.get('times', []))
    if len(transit_times) < 3: # Need at least 3 points for a meaningful periodogram
        logger.info(f"Cannot perform periodicity analysis: only {len(transit_times)} transits detected.")
        return None

    logger.debug(f"Analyzing periodicity for {len(transit_times)} transit times.")
    
    try:
        # Use Lomb-Scargle on the transit times themselves
        # Frequencies can be auto-determined
        ls = LombScargle(transit_times, 1) # Using a dummy y=1 as we only care about times
        frequency, power = ls.autopower(minimum_frequency=0.05, maximum_frequency=1.0) # Search for periods from 1 to 20 days
        
        # Find the peak periods
        best_frequency = frequency[np.argmax(power)]
        best_period = 1.0 / best_frequency
        
        # You could also find multiple significant peaks if needed
        # from scipy.signal import find_peaks
        # peaks, _ = find_peaks(power, height=0.1) # Example: peaks with power > 0.1
        # peak_periods = 1.0 / frequency[peaks] if len(peaks) > 0 else []

        periodogram_data = {
            'period': (1.0 / frequency).tolist(),
            'power': power.tolist(),
            'best_period': best_period,
            'peak_periods': [best_period] # Storing the best one as a list for consistency
        }
        
        # For this function, let's assume it primarily finds the period.
        # We can calculate a median period if multiple peaks were found, but best_period is often sufficient.
        
        logger.info(f"Periodicity analysis found best period: {best_period:.4f} days.")
        return {
            'median_period': best_period, # Use 'best_period' as the representative median
            'periodogram': periodogram_data
        }

    except Exception as e:
        logger.error(f"Error during Lomb-Scargle periodogram analysis: {e}", exc_info=True)
        return None


def calculate_folded_lightcurve(time, flux, period, epoch=0):
    """
    Folds the light curve to the given period.
    """
    if time is None or flux is None or period <= 0:
        return None, None
    
    # phase = (time - epoch) % period / period
    phase = np.mod(time - epoch, period) / period
    # Sort by phase
    sort_indices = np.argsort(phase)
    return phase[sort_indices], flux[sort_indices]


def bin_folded_lightcurve(phase, flux, n_bins=100):
    """
    Bins the phase-folded light curve to make the signal clearer.
    """
    if phase is None or flux is None or len(phase) == 0:
        return None, None, None
        
    # Define bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Use pandas for robust binning, which handles empty bins
    df = pd.DataFrame({'phase': phase, 'flux': flux})
    df['bin'] = pd.cut(df['phase'], bins=bin_edges, labels=False, include_lowest=True)
    
    # Group by bin and calculate mean flux and standard error of the mean
    binned_data = df.groupby('bin')['flux'].agg(['mean', 'sem']).reset_index()
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_data['bin_center'] = bin_centers[binned_data['bin']]
    
    # Create final arrays, filling missing bins with NaNs
    final_bin_centers = bin_centers
    final_mean_flux = np.full(n_bins, np.nan)
    final_sem_flux = np.full(n_bins, np.nan)
    
    final_mean_flux[binned_data['bin']] = binned_data['mean']
    final_sem_flux[binned_data['bin']] = binned_data['sem']

    return final_bin_centers, final_mean_flux, final_sem_flux
