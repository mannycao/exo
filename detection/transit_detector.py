
"""
Functions for modeling transits and estimating planet properties.
"""
import logging
import numpy as np
from astropy.timeseries import BoxLeastSquares
import config
from datetime import datetime

# Use a try-except block for batman import as it might not be installed
try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    logging.warning(
        "Batman library not found. Transit modeling capabilities will be limited. "
        "Please install batman-package (`pip install batman-package`)."
    )

logger = logging.getLogger(__name__)

def apply_transit_modeling(time, flux, transit_info, periodicity_data):
    """
    Fits a transit model to the light curve using the 'batman' package.
    Safely accesses transit_info and periodicity_data.
    """
    if not BATMAN_AVAILABLE:
        logger.error("Cannot apply transit modeling: batman package is not installed.")
        return None
        
    period = periodicity_data.get('median_period')
    if period is None or period <= 0:
        logger.debug("apply_transit_modeling: Invalid or no period provided.")
        return None
    
    # Get necessary parameters. Use .get() for safe access.
    # We need an initial epoch (t0) and depth.
    # Let's take the time and depth of the strongest transit as initial guesses.
    
    transit_times = np.asarray(transit_info.get('times', []))
    # Use prominences from find_peaks as our 'depths'
    transit_depths = np.asarray(transit_info.get('depths', [])) 
    
    if len(transit_times) == 0 or len(transit_depths) == 0:
        logger.warning("apply_transit_modeling: No transit times or depths available.")
        return None
        
    # Find the deepest transit to use for t0 and initial depth
    deepest_transit_idx = np.argmax(transit_depths)
    t0_initial = transit_times[deepest_transit_idx]
    depth_initial = transit_depths[deepest_transit_idx]
    
    # Estimate planet radius to star radius ratio (rp/rs) from depth
    rp_rs_initial = np.sqrt(depth_initial) if depth_initial > 0 else 0.1
    
    # Estimate semi-major axis over stellar radius (a/rs) using Kepler's Third Law (approx.)
    # This requires stellar parameters (mass, radius), which might not be available.
    # Batman can take `a` (in stellar radii) as a parameter.
    # For now, let's use a typical value or derive it if possible.
    # A common value for short-period planets is between 5 and 20.
    a_rs_initial = 15.0 

    # Estimate inclination from transit duration (this is complex).
    # For now, assume a central transit (i=90 degrees).
    inclination_initial = 90.0
    
    logger.debug(f"Initializing batman model with: period={period:.4f}, t0={t0_initial:.4f}, rp/rs={rp_rs_initial:.4f}")

    try:
        params = batman.TransitParams()
        params.t0 = t0_initial                # time of inferior conjunction
        params.per = period                   # orbital period
        params.rp = rp_rs_initial             # planet radius (in units of stellar radii)
        params.a = a_rs_initial               # semi-major axis (in units of stellar radii)
        params.inc = inclination_initial      # orbital inclination (in degrees)
        params.ecc = 0.                       # eccentricity
        params.w = 90.                        # longitude of periastron (in degrees)
        params.limb_dark = "quadratic"        # limb darkening model
        params.u = [0.1, 0.3]                 # limb darkening coefficients

        # Initialize the model and generate the light curve
        m = batman.TransitModel(params, time)
        model_flux = m.light_curve(params)

        return {
            'model_flux': model_flux.tolist(),
            'model_params': {
                't0': params.t0,
                'per': params.per,
                'rp': params.rp,
                'a': params.a,
                'inc': params.inc
            }
        }
    except Exception as e:
        logger.error(f"Error during batman transit modeling: {e}", exc_info=True)
        return None


def estimate_planet_properties(transit_info, periodicity_data, stellar_properties=None):
    """
    Provides rough estimates of planet properties based on detection results.
    Safely accesses transit_info and periodicity_data dictionaries.
    """
    if stellar_properties is None:
        stellar_properties = {'radius': 1.0, 'mass': 1.0, 'temperature': 5778.0}
        logger.debug("Using default stellar properties (Sun-like) for planet property estimation.")

    # Use .get() for safe dictionary access with defaults
    orbital_period_days = periodicity_data.get('median_period')
    if orbital_period_days is None or orbital_period_days <= 0:
        logger.warning("Cannot estimate planet properties: invalid orbital period.")
        return None
        
    # Use prominences from find_peaks as our 'depths'
    depths = transit_info.get('depths')
    if depths is None or len(depths) == 0:
        logger.warning("Cannot estimate planet properties: no transit depths available.")
        return None

    median_depth = np.median(depths)
    if median_depth <= 0:
        logger.warning(f"Cannot estimate radius: median transit depth is non-positive ({median_depth}).")
        return None
    
    st_rad_solar = stellar_properties.get('radius', 1.0)
    st_mass_solar = stellar_properties.get('mass', 1.0)
    st_teff_k = stellar_properties.get('temperature', 5778.0)
    
    # Constants
    R_sun_to_R_earth = 109.2
    AU_to_R_sun = 215.0

    try:
        # Estimate planet radius
        # depth = (Rp/R*)^2 => Rp = R* * sqrt(depth)
        planet_radius_solar = st_rad_solar * np.sqrt(median_depth)
        planet_radius_earth = planet_radius_solar * R_sun_to_R_earth

        # Estimate semi-major axis (a) using Kepler's Third Law
        # P^2 = a^3 / M* (in years, AU, solar masses) -> a = (M* * P^2)^(1/3)
        orbital_period_years = orbital_period_days / 365.25
        semi_major_axis_au = (st_mass_solar * orbital_period_years**2)**(1/3.)
        
        # Estimate equilibrium temperature (assuming zero albedo and perfect energy redistribution)
        # Teq = T* * sqrt(R* / (2a))
        semi_major_axis_rsun = semi_major_axis_au * AU_to_R_sun
        equilibrium_temp_k = st_teff_k * np.sqrt(st_rad_solar / (2 * semi_major_axis_rsun))

        properties = {
            "radius_earth": planet_radius_earth,
            "orbital_period_days": orbital_period_days,
            "semi_major_axis_au": semi_major_axis_au,
            "equilibrium_temp_k": equilibrium_temp_k
        }
        logger.debug(f"Estimated planet properties: {properties}")
        return properties

    except Exception as e:
        logger.error(f"Error during planet property estimation: {e}", exc_info=True)
        return None

def find_transits_bls(time, flux):
    """
    Performs Box-Least-Squares (BLS) transit detection.
    """
    logger.info("Running BLS transit detection.")

    if len(time) == 0 or len(flux) == 0:
        logger.warning("Skipping BLS (find_transits_bls): time or flux array is empty.")
        return None, None

    if np.isnan(time).any() or np.isinf(time).any():
        logger.warning("Skipping BLS (find_transits_bls): time array contains NaN or Inf values after finite filter.")
        return None, None

    if np.isnan(flux).any() or np.isinf(flux).any():
        logger.warning("Skipping BLS (find_transits_bls): flux array contains NaN or Inf values after finite filter.")
        return None, None

    if np.std(flux) < 1e-6: # Check for nearly constant flux
        logger.warning("Skipping BLS (find_transits_bls): flux array is nearly constant.")
        return None, None

    if not np.all(np.diff(time) > 0): # Check for monotonic increasing time
        logger.warning("Skipping BLS (find_transits_bls): time array is not monotonically increasing.")
        return None, None

    logger.debug(f"BLS input time shape: {time.shape}, min: {time.min():.2f}, max: {time.max():.2f}, has NaNs: {np.isnan(time).any()}")
    logger.debug(f"BLS input flux shape: {flux.shape}, min: {flux.min():.2f}, max: {flux.max():.2f}, has NaNs: {np.isnan(flux).any()}")

    # Define minimum and maximum transit durations from config
    min_duration = config.MIN_TRANSIT_DURATION
    max_duration = config.MAX_TRANSIT_DURATION

    # Create a BLS object
    model = BoxLeastSquares(time, flux)

    # Calculate the periodogram
    # Use a more refined period grid based on the data duration
    min_period = max(config.MAX_TRANSIT_DURATION * 2, 0.5) # At least twice max_duration, and not less than 0.5 days
    max_period = (time[-1] - time[0]) / 2.0 # Max period is half the observation span

    if max_period <= min_period:
        logger.warning(f"Skipping BLS (find_transits_bls): max_period ({max_period:.2f}) is not greater than min_period ({min_period:.2f}). Observation span too short.")
        return None, None

    # Ensure a reasonable range for periods to avoid issues with np.linspace
    if (max_period - min_period) < 1e-5: # Arbitrary small threshold
        logger.warning(f"Skipping BLS (find_transits_bls): Period range ({max_period - min_period:.2e}) is too small.")
        return None, None

    periods = np.linspace(min_period, max_period, 1000)

    # Create an array of durations to test
    durations = np.linspace(config.MIN_TRANSIT_DURATION, config.MAX_TRANSIT_DURATION, 10)

    try:
        results = model.power(periods, durations, oversample=10)
    except Exception as e:
        logger.error(f"BLS model.power() failed for current file due to: {type(e).__name__}: {e}. Skipping.", exc_info=True)
        return None, None

    # Find the period with the highest power
    best_period_idx = np.argmax(results.power)
    best_period = results.period[best_period_idx]
    best_t0 = results.transit_time[best_period_idx]
    best_duration = results.duration[best_period_idx]
    best_depth = results.depth[best_period_idx]

    if results.power[best_period_idx] < config.BLS_POWER_THRESHOLD:
        logger.warning(f"Skipping BLS (find_transits_bls): No significant transit detected (max power: {results.power[best_period_idx]:.2f} below threshold {config.BLS_POWER_THRESHOLD:.2e}).")
        return None, None

    # Populate transit_info and periodicity_data
    transit_info = {
        'times': np.array([best_t0]),
        'depths': np.array([best_depth]),
        'durations': np.array([best_duration]),
        'peak_indices': np.array([np.argmin(np.abs(time - best_t0))]) # Approximate index
    }
    periodicity_data = {
        'median_period': best_period,
        'periodogram': {
            'period': results.period.tolist(),
            'power': results.power.tolist(),
            'best_period': best_period,
            'peak_periods': [best_period]
        },
        'max_power': results.power[best_period_idx]
    }
    logger.info(f"BLS detected transit: Period={best_period:.4f}, t0={best_t0:.4f}, Duration={best_duration:.4f}, Depth={best_depth:.4f}")
    return transit_info, periodicity_data
