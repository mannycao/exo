import logging
import numpy as np
from discovery_stack.types import TransitHypothesis

logger = logging.getLogger(__name__)

def check_orbital_stability(hypothesis: TransitHypothesis) -> bool:
    """
    Checks if the proposed orbit is physically stable using Roche limits
    and collision geometries.
    """
    params = hypothesis.signal_params
    
    # Stellar and planetary properties (extracted from signal_params or estimated)
    # R_star in Solar radii, M_star in Solar masses
    R_star = params.get('stellar_radius', 1.0) 
    M_star = params.get('stellar_mass', 1.0)
    # R_p in Earth radii
    R_p = params.get('radius_earth', 1.0)
    # Estimate planet mass based on radius (M ~ R^2 is a very rough proxy for rocky/sub-Neptune)
    M_p = params.get('planet_mass', R_p**2.0) 
    
    # Semi-major axis in AU
    a_au = params.get('semi_major_axis_au', 0.1)
    # Convert AU to Solar Radii for comparison (1 AU approx 215 Rsun)
    a_rsol = a_au * 215.032
    
    # 1. Collision Geometry Check
    # If the semi-major axis is less than the stellar radius, it's a collision
    if a_rsol <= R_star:
        hypothesis.log(f"VETO (Orbital Stability): Collision geometry detected. a={a_rsol:.2f} Rsun, R*={R_star:.2f} Rsun")
        return False

    # 2. Roche Limit Check (Fluid Body Approximation)
    # Simplified Roche limit: d_Roche approx 2.44 * R_star * (rho_star / rho_p)^(1/3)
    # rho_star in g/cm^3 (Sun approx 1.41)
    # rho_p in g/cm^3 (Earth approx 5.51)
    rho_star = 1.41 * (M_star / R_star**3)
    rho_p = 5.51 * (M_p / R_p**3)
    
    roche_limit = 2.44 * R_star * (rho_star / rho_p)**(1/3)
    
    if a_rsol < roche_limit:
        hypothesis.log(f"VETO (Orbital Stability): Within Roche limit. a={a_rsol:.2f} Rsun, Roche={roche_limit:.2f} Rsun")
        return False

    logger.info(f"Orbital stability check passed for {hypothesis.id}")
    return True
