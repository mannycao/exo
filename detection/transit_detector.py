"""
Functions for modeling transits and estimating planet properties.
"""
import logging
import numpy as np

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

