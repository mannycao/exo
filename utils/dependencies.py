# FILE: utils/dependencies.py (Corrected and Complete)

import logging

try:
    import lightkurve
    HAS_LIGHTKURVE = True
    logging.info(f"Found 'lightkurve' version {lightkurve.__version__}. Real data downloading is enabled.")
except ImportError:
    HAS_LIGHTKURVE = False
    logging.error("The 'lightkurve' library was not found. Real data fetching is disabled.")

try:
    import astroquery
    HAS_ASTROQUERY = True
    logging.info(f"Found 'astroquery' version {astroquery.__version__}. Catalog searching is enabled.")
except ImportError:
    HAS_ASTROQUERY = False
    logging.error("The 'astroquery' library was not found. Automated target finding is disabled.")

# Add a check for scikit-image as well for completeness
try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
