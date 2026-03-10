import logging
import numpy as np
from discovery_stack.types import TransitHypothesis

logger = logging.getLogger(__name__)

def check_transit_shape(hypothesis: TransitHypothesis) -> bool:
    """
    Checks if the transit shape and depth are physically plausible for a planet.
    This includes checking for excessive depths (likely EB) and analyzing
    implied densities.
    """
    params = hypothesis.signal_params
    depth = params.get('depth', 0.0)

    # 1. Depth Constraint (Proxy for V-shape/EB check)
    # A transit depth > 5% is very rare for planets orbiting main-sequence stars
    # (excluding giants), and depth > 30% is almost certainly an eclipsing binary.
    if depth > 0.3:
        hypothesis.log(f"VETO (Transit Geometry): Depth={depth:.2%} > 30% indicates Eclipsing Binary.")
        return False
    elif depth > 0.05:
        logger.info(f"WARNING (Transit Geometry): High depth {depth:.2%} for {hypothesis.id}")

    # 2. Implied Density Constraint
    # We can estimate the implied stellar density from the transit duration and period
    # and compare it to the catalog value.
    # Simplified transit equation for circular orbits:
    # rho_implied approx (3 * period) / (G * pi^3 * duration^3)
    # If the implied density is orders of magnitude off from catalog R*/M*, it's likely a FP.
    
    period_days = params.get('orbital_period_days', 0.0)
    duration_hours = params.get('duration_hours', 0.0)
    
    if period_days > 0 and duration_hours > 0:
        # Convert to SI units for a rough density estimate
        # (This is a simplified scaling, real pipelines use more rigorous transit models)
        duration_days = duration_hours / 24.0
        rho_solar = (period_days / duration_days**3) * 0.0134 # Simple scaling factor for rho_star
        
        R_star = params.get('stellar_radius', 1.0)
        M_star = params.get('stellar_mass', 1.0)
        rho_catalog = M_star / R_star**3
        
        # Allow for a factor of 10 discrepancy (generous for eccentricity/noise)
        if rho_solar / rho_catalog > 10.0 or rho_catalog / rho_solar > 10.0:
            hypothesis.log(f"VETO (Transit Geometry): Implied density ({rho_solar:.2f} rho_sun) "
                          f"is inconsistent with catalog ({rho_catalog:.2f} rho_sun).")
            return False

    logger.info(f"Transit geometry check passed for {hypothesis.id}")
    return True
