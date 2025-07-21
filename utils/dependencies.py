# utils/dependencies.py

import logging

logger = logging.getLogger(__name__)

# --- Astroquery Import ---
try:
    from astroquery.mast import Observations
    from astroquery.exoplanet_archive import ExoplanetArchive
    ASTROQUERY_AVAILABLE = True
    logger.info("Successfully imported astroquery.")
except ImportError:
    Observations = None
    ExoplanetArchive = None
    ASTROQUERY_AVAILABLE = False
    logger.warning("Astroquery library not found. Real data fetching will be limited.")

# --- Lightkurve Import ---
try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
    logger.info("Successfully imported lightkurve.")
except ImportError:
    lk = None
    LIGHTKURVE_AVAILABLE = False
    logger.warning("Lightkurve library not found. Light curve processing will be unavailable.")