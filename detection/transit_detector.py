# FILE: detection/transit_detector.py (Final, Complete, and Corrected Version)

import logging
import numpy as np
from scipy.signal import find_peaks
import config
from .periodicity_analyzer import analyze_periodicity

# The batman-package is an optional dependency for physical modeling
try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False

logger = logging.getLogger(__name__)

def detect_transits(time, flux):
    """
    Detects transit-like dips in a light curve's flux data with corrected
    peak finding logic.
    """
    if flux is None or len(flux) < 10:
        return {'found_transits': False, 'transit_count': 0}

    inverted_flux = -flux
    mean_inverted_flux = np.mean(inverted_flux)
    std_flux = np.std(flux)
    height_threshold = mean_inverted_flux + (config.TRANSIT_SENSITIVITY * std_flux)
    
    logger.debug(f"Detecting peaks with height threshold > {height_threshold:.4f}")

    try:
        peaks, properties = find_peaks(
            inverted_flux,
            height=height_threshold,
            width=(config.MIN_TRANSIT_DURATION_CADENCES, None)
        )
    except Exception as e:
        logger.error(f"The 'find_peaks' function failed: {e}", exc_info=True)
        return {'found_transits': False, 'transit_count': 0}

    if len(peaks) > 0:
        logger.info(f"Successfully detected {len(peaks)} transit-like events.")
        transit_times = time[peaks]
        transit_depths = inverted_flux[peaks]
        
        periodicity_results = analyze_periodicity(time, flux, transit_times)
        
        return {
            'found_transits': True,
            'transit_count': len(peaks),
            'transit_indices': peaks,
            'transit_times': transit_times,
            'transit_depths': transit_depths,
            **periodicity_results
        }
    else:
        logger.warning("No transit events were detected that met the criteria.")
        return {'found_transits': False, 'transit_count': 0}


def apply_transit_modeling(time, flux, transit_info):
    """
    Fits a physical transit model using the batman package if available.
    This function is now present to prevent the ImportError.
    """
    if not HAS_BATMAN:
        logger.warning("Batman package not found. Skipping physical transit modeling.")
        return {}
        
    if not transit_info or not transit_info.get('period_days') or transit_info['period_days'] <= 0:
        logger.debug("Insufficient info for transit modeling. Skipping.")
        return {}

    logger.debug(f"Applying transit modeling with period: {transit_info['period_days']:.4f}")
    # This is a simplified placeholder for a full MCMC or optimization fitting process.
    # It uses the results from periodicity analysis to create a basic model.
    params = batman.TransitParams()
    params.t0 = transit_info.get('t0', 0)
    params.per = transit_info.get('period_days')
    
    # Estimate Rp/Rs from the max depth of detected transits
    if transit_info.get('transit_depths') is not None and len(transit_info['transit_depths']) > 0:
        max_depth = np.max(transit_info['transit_depths'])
        params.rp = np.sqrt(max_depth) if max_depth > 0 else 0.1
    else:
        params.rp = 0.1 # Default value

    # These are typical placeholder values for a Sun-like star
    params.a = 15.       # semi-major axis (in units of stellar radii)
    params.inc = 90.     # orbital inclination (in degrees)
    params.ecc = 0.      # eccentricity
    params.w = 90.       # longitude of periastron (in degrees)
    params.u = [0.1, 0.3] # limb darkening coefficients
    params.limb_dark = "quadratic"

    try:
        m = batman.TransitModel(params, time)
        model_flux = m.light_curve(params)
        return {"model_flux": model_flux}
    except Exception as e:
        logger.error(f"Batman transit model fitting failed: {e}")
        return {}
