"""
Functions for analyzing periodicity in light curve data.
"""

import logging
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def periodogram_analysis(time, flux):
    """
    Performs periodogram analysis to detect periodic signals.
    
    Args:
        time: Time series data
        flux: Normalized flux data
    
    Returns:
        dict: Periodogram analysis results
    """
    try:
        from astropy.timeseries import LombScargle
        
        # Remove NaN values
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]
        
        # Compute the Lomb-Scargle periodogram
        frequency, power = LombScargle(time, flux).autopower()
        
        # Convert frequency to period
        period = 1/frequency
        
        # Find peaks in the periodogram
        peak_indices, _ = find_peaks(power, height=0.1)
        
        # Handle case with no peaks
        if len(peak_indices) == 0:
            return {
                'period': period,
                'power': power,
                'peak_periods': np.array([]),
                'peak_powers': np.array([])
            }
            
        peak_periods = period[peak_indices]
        peak_powers = power[peak_indices]
        
        # Sort by power
        sort_idx = np.argsort(peak_powers)[::-1]
        peak_periods = peak_periods[sort_idx]
        peak_powers = peak_powers[sort_idx]
        
        return {
            'period': period,
            'power': power,
            'peak_periods': peak_periods[:10],  # Top 10 peaks
            'peak_powers': peak_powers[:10]
        }
    except Exception as e:
        logger.error(f"Error in periodogram analysis: {e}")
        return None


def analyze_periodicity(time, flux, transit_info):
    """
    Analyzes periodicity of detected transits.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        transit_info: Dictionary with transit information
    
    Returns:
        dict: Periodicity analysis results or None if analysis fails
    """
    if transit_info is None or len(transit_info['peak_indices']) < 2:
        return None
    
    try:
        # Calculate time differences between consecutive transits
        transit_times = transit_info['times']
        time_diffs = np.diff(transit_times)
        
        # Calculate the median time difference (potential orbital period)
        median_period = np.median(time_diffs)
        
        # Perform periodogram analysis
        periodogram = periodogram_analysis(time, flux)
        if periodogram is None:
            return {'median_period': median_period}
        
        # Check if the detected period from transits matches any peak in the periodogram
        period_matches = []
        for peak_period in periodogram['peak_periods']:
            # Check if the peak period is close to the median period or its harmonics
            if abs(peak_period - median_period) / median_period < 0.1:
                period_matches.append(peak_period)
            elif abs(peak_period - 2*median_period) / (2*median_period) < 0.1:
                period_matches.append(peak_period / 2)  # Half the harmonic
        
        return {
            'median_period': median_period,
            'period_matches': period_matches,
            'periodogram': periodogram
        }
    except Exception as e:
        logger.error(f"Error in periodicity analysis: {e}")
        return None


def calculate_folded_lightcurve(time, flux, period):
    """
    Calculates phase-folded light curve based on the detected period.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        period: Orbital period in days
    
    Returns:
        tuple: (phase, folded_flux) arrays
    """
    if period <= 0:
        return None, None
    
    # Calculate phase (between 0 and 1)
    phase = (time % period) / period
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    return phase_sorted, flux_sorted


def bin_folded_lightcurve(phase, flux, bins=100):
    """
    Bins the folded light curve for smoother visualization and analysis.
    
    Args:
        phase: Phase values (0-1)
        flux: Corresponding flux values
        bins: Number of bins to use
    
    Returns:
        tuple: (binned_phase, binned_flux, binned_error)
    """
    if phase is None or flux is None:
        return None, None, None
    
    binned_flux = np.zeros(bins)
    binned_error = np.zeros(bins)
    bin_counts = np.zeros(bins)
    
    # Create phase bins
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Bin the flux values
    for i in range(len(phase)):
        bin_idx = int(phase[i] * bins)
        if bin_idx == bins:  # Handle edge case
            bin_idx = bins - 1
        binned_flux[bin_idx] += flux[i]
        bin_counts[bin_idx] += 1
    
    # Calculate average flux and error for each bin
    for i in range(bins):
        if bin_counts[i] > 0:
            binned_flux[i] /= bin_counts[i]
        else:
            # Fill empty bins with adjacent values
            if i > 0 and i < bins - 1:
                binned_flux[i] = (binned_flux[i-1] + binned_flux[i+1]) / 2
            elif i > 0:
                binned_flux[i] = binned_flux[i-1]
            elif i < bins - 1:
                binned_flux[i] = binned_flux[i+1]
    
    # Calculate standard error for each bin
    for i in range(len(phase)):
        bin_idx = int(phase[i] * bins)
        if bin_idx == bins:
            bin_idx = bins - 1
        binned_error[bin_idx] += (flux[i] - binned_flux[bin_idx])**2
    
    for i in range(bins):
        if bin_counts[i] > 1:
            binned_error[i] = np.sqrt(binned_error[i] / (bin_counts[i] - 1)) / np.sqrt(bin_counts[i])
    
    return bin_centers, binned_flux, binned_error