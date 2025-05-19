"""
Transit detection and characterization functions.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_transit_modeling(time, flux, transit_info, periodicity_info):
    """
    Applies transit modeling to characterize the potential exoplanet.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        transit_info: Dictionary with transit information
        periodicity_info: Dictionary with periodicity information
    
    Returns:
        dict: Transit model information or None if modeling fails
    """
    if transit_info is None or periodicity_info is None:
        return None
    
    try:
        import batman
        
        # Use the detected period from periodicity analysis
        period = periodicity_info['median_period']
        if period <= 0:
            return None
        
        # Create a transit model
        params = batman.TransitParams()
        params.t0 = transit_info['times'][0]  # Time of first transit
        params.per = period  # Orbital period
        params.rp = 0.1  # Planet radius (relative to stellar radius)
        params.a = 15.0  # Semi-major axis (in units of stellar radii)
        params.inc = 87.0  # Orbital inclination (in degrees)
        params.ecc = 0.0  # Eccentricity
        params.w = 90.0  # Longitude of periastron (in degrees)
        params.u = [0.1, 0.3]  # Limb darkening coefficients
        params.limb_dark = "quadratic"  # Limb darkening model
        
        # Initialize the model
        m = batman.TransitModel(params, time)
        
        # Generate the model light curve
        model_flux = m.light_curve(params)
        
        # Calculate residuals
        residuals = flux - model_flux
        residual_std = np.std(residuals)
        
        # Simplified fit quality metric
        fit_quality = np.mean(np.abs(residuals))
        
        return {
            'model_params': params,
            'model_flux': model_flux,
            'residuals': residuals,
            'residual_std': residual_std,
            'fit_quality': fit_quality
        }
    except Exception as e:
        logger.error(f"Error in transit modeling: {e}")
        return None


def estimate_planet_properties(transit_info, periodicity_info, stellar_properties=None):
    """
    Estimates basic planet properties from transit and periodicity data.
    
    Args:
        transit_info: Dictionary with transit information
        periodicity_info: Dictionary with periodicity information
        stellar_properties: Dictionary with stellar properties (optional)
    
    Returns:
        dict: Estimated planet properties or None if estimation fails
    """
    if transit_info is None or periodicity_info is None:
        return None
    
    # Default stellar properties if not provided
    if stellar_properties is None:
        stellar_properties = {
            'radius': 1.0,  # Solar radius
            'mass': 1.0,    # Solar mass
            'temperature': 5778  # Solar temperature (K)
        }
    
    # Extract transit depth and duration
    depths = transit_info['depths']
    mean_depth = np.mean(depths)
    
    # Estimate planet radius (in Earth radii)
    # Planet radius / star radius = sqrt(transit depth)
    planet_radius_ratio = np.sqrt(abs(mean_depth))
    planet_radius = planet_radius_ratio * stellar_properties['radius'] * 109.2  # Convert to Earth radii
    
    # Estimate orbital period (in days)
    orbital_period = periodicity_info['median_period']
    
    # Estimate semi-major axis using Kepler's Third Law
    # a^3 / P^2 = G(M + m) / 4π^2 ≈ GM / 4π^2 (since M >> m)
    # a = (GM * P^2 / 4π^2)^(1/3)
    G = 6.67430e-11  # Gravitational constant
    M_sun = 1.989e30  # Solar mass in kg
    star_mass_kg = stellar_properties['mass'] * M_sun
    P_seconds = orbital_period * 86400  # Convert days to seconds
    
    semi_major_axis_m = (G * star_mass_kg * P_seconds**2 / (4 * np.pi**2))**(1/3)
    semi_major_axis_au = semi_major_axis_m / 1.496e11  # Convert to AU
    
    # Estimate equilibrium temperature
    # T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^(1/4)
    albedo = 0.3  # Assumed albedo
    star_radius_m = stellar_properties['radius'] * 6.957e8  # Convert to meters
    equilibrium_temp = stellar_properties['temperature'] * np.sqrt(star_radius_m / (2 * semi_major_axis_m)) * (1 - albedo)**(1/4)
    
    return {
        'radius_earth': planet_radius,
        'orbital_period_days': orbital_period,
        'semi_major_axis_au': semi_major_axis_au,
        'equilibrium_temp_k': equilibrium_temp
    }
