"""
Functions for analyzing periodicity in light curve data,
including Lomb-Scargle periodograms and phase-folding.
"""
import logging
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from .periodicity_filters import validate_period_detection, get_period_quality_metrics
from utils.visualization import visualize_periodicity_analysis
from analysis.period_statistics import generate_analysis_report
import os

logger = logging.getLogger(__name__)

# Global list to collect results for statistical analysis
_period_results = []

def clear_period_results():
    """Clear the global results list."""
    global _period_results
    _period_results = []

def get_period_results():
    """Get the current list of period results."""
    return _period_results.copy()

def generate_period_statistics(output_dir="results/period_analysis"):
    """Generate statistical analysis of all period detections."""
    if not _period_results:
        logger.warning("No period results available for statistical analysis")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    return generate_analysis_report(_period_results, output_dir)

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
        # Create time series for Lomb-Scargle
        # Instead of using dummy y=1 values, we want to emphasize the transit events
        # by setting y=1 at transit times and y=0 elsewhere
        duration = max(transit_times) - min(transit_times)
        times = np.linspace(min(transit_times), max(transit_times), 1000)
        y = np.zeros_like(times)
        
        # Mark each transit with a 1, using a small window
        window = duration / 100  # Small enough to avoid aliasing but large enough to be detected
        for t in transit_times:
            y[(times >= t - window/2) & (times <= t + window/2)] = 1
            
        ls = LombScargle(times, y)
        
        # Search for periods from 0.5 to 200 days
        frequency, power = ls.autopower(
            minimum_frequency=1/200,  # 200 days maximum period
            maximum_frequency=2.0,    # 0.5 days minimum period
            samples_per_peak=10)      # Ensure good frequency sampling
        
        # Find the peak periods and calculate confidence metrics
        peak_power_idx = np.argmax(power)
        best_frequency = frequency[peak_power_idx]
        best_period = 1.0 / best_frequency
        
        # Calculate False Alarm Probability (FAP)
        fap = ls.false_alarm_probability(power[peak_power_idx])
        
        # Calculate power significance
        mean_power = np.mean(power)
        power_snr = power[peak_power_idx] / mean_power
        
        # Calculate signal-to-noise ratio of the peak
        # Use local noise level (standard deviation of power in window around peak)
        window = 5  # Points on each side
        start_idx = max(0, peak_power_idx - window)
        end_idx = min(len(power), peak_power_idx + window + 1)
        local_noise = np.std(power[start_idx:end_idx])
        peak_snr = power[peak_power_idx] / local_noise if local_noise > 0 else 0
        
        # Calculate confidence score (0-1) based on multiple metrics
        confidence_score = (1 - fap) * np.minimum(1.0, power_snr / 10) * np.minimum(1.0, peak_snr / 5)

        periodogram_data = {
            'period': (1.0 / frequency).tolist(),
            'power': power.tolist(),
            'best_period': best_period,
            'peak_periods': [best_period],
            'fap': float(fap),
            'power_snr': float(power_snr),
            'peak_snr': float(peak_snr),
            'confidence_score': float(confidence_score)
        }
        
        # For this function, let's assume it primarily finds the period.
        # We can calculate a median period if multiple peaks were found, but best_period is often sufficient.
        
        # Get quality metrics and validate the detection
        quality_metrics = get_period_quality_metrics(periodogram_data)
        data_span = max(transit_times) - min(transit_times)
        is_valid, reason, adjusted_confidence = validate_period_detection(
            periodogram_data,
            data_span=data_span
        )
        
        # Update confidence score with validation results
        confidence_score = adjusted_confidence
        periodogram_data['confidence_score'] = float(confidence_score)
        periodogram_data.update(quality_metrics)
        
        # Generate visualization if period is potentially valid
        if confidence_score > 0.3:  # Low threshold for visualization
            try:
                output_dir = "results/periodicity_analysis"
                import os
                os.makedirs(output_dir, exist_ok=True)
                visualize_periodicity_analysis(
                    periodogram_data,
                    title=f"Period: {best_period:.2f} days (Confidence: {confidence_score:.2f})",
                    filename=f"periodogram_{len(transit_times)}_transits.png",
                    output_dir=output_dir
                )
            except Exception as e:
                logger.warning(f"Failed to generate periodicity visualization: {e}")
        
        # Log results with validation info
        if is_valid:
            logger.info(f"Valid period detected: {best_period:.4f} days (confidence: {confidence_score:.2f})")
        else:
            logger.warning(f"Potential period {best_period:.4f} days marked as invalid: {reason} (confidence: {confidence_score:.2f})")
        
        result = {
            'median_period': best_period,
            'confidence': confidence_score,
            'validation': {'is_valid': is_valid, 'reason': reason},
            'quality_metrics': quality_metrics,
            'periodogram': periodogram_data,
            'data_span': data_span,
            'n_transits': len(transit_times),
            'transit_times': transit_times.tolist()
        }
        
        # Add to global results list for statistical analysis
        _period_results.append(result)
        
        return result

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
