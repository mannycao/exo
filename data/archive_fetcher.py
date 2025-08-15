# data/archive_fetcher.py

import logging
import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

def get_star_info(kepler_id):
    """
    Queries the NASA Exoplanet Archive to get the primary star name for a Kepler ID.
    The archive often uses star names (e.g., 'Kepler-62') for queries, not just IDs.

    Args:
        kepler_id (str): The Kepler ID of the star (e.g., '6278762').

    Returns:
        A dictionary with 'hostname' and 'kepid' if found, otherwise None.
    """
    try:
        star_info = NasaExoplanetArchive.query_criteria(
            table="keplernames",
            select="kepid, kepoi_name, kepler_name",
            where=f"kepid = {kepler_id}"
        )
        if len(star_info) > 0:
            # Prefer the 'kepler_name' if available, otherwise use the KOI name
            hostname = star_info['kepler_name'][0] or star_info['kepoi_name'][0].split('.')[0]
            return {'hostname': hostname, 'kepid': star_info['kepid'][0]}
        return None
    except Exception as e:
        logger.error(f"Failed to query star info for KIC {kepler_id}: {e}")
        return None

def get_rv_data(hostname, required_points=100):
    """
    Fetches and processes radial velocity data for a given star from the NASA Exoplanet Archive.

    Args:
        hostname (str): The name of the host star (e.g., 'Kepler-62').
        required_points (int): The number of data points for the output time series.

    Returns:
        A numpy array of the processed RV time series, or None if not found.
    """
    try:
        logger.debug(f"Querying RV data for {hostname}")
        rv_data = NasaExoplanetArchive.query_criteria(
            table="rv_file",
            select="rv, rv_err",
            where=f"hostname = '{hostname}'"
        )
        if len(rv_data) < 10:  # Require a minimum number of observations
            logger.warning(f"Insufficient RV data found for {hostname} ({len(rv_data)} points).")
            return None

        rv_values = rv_data['rv'].data.astype(float)
        
        # --- Preprocessing ---
        # 1. Normalize the data
        rv_normalized = (rv_values - np.mean(rv_values)) / (np.std(rv_values) + 1e-8)
        
        # 2. Resample to a fixed length using interpolation
        x_original = np.linspace(0, 1, len(rv_normalized))
        x_new = np.linspace(0, 1, required_points)
        f = interp1d(x_original, rv_normalized, kind='linear', fill_value="extrapolate")
        rv_resampled = f(x_new)

        return rv_resampled.reshape((required_points, 1)) # Shape for LSTM

    except Exception as e:
        logger.error(f"Could not fetch or process RV data for {hostname}: {e}")
        return None


def get_imaging_data(hostname):
    """
    Fetches high-resolution imaging data (contrast and separation) for a given star.

    Args:
        hostname (str): The name of the host star.

    Returns:
        A numpy array containing [contrast, separation], or None if not found.
    """
    try:
        logger.debug(f"Querying imaging data for {hostname}")
        # We query the Planetary Systems Composite Parameters table for imaging-specific columns
        imaging_data = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_letter, im_contrast, im_sep",
            where=f"hostname = '{hostname}' and discoverymethod = 'Imaging'"
        )

        if len(imaging_data) == 0:
            logger.warning(f"No direct imaging data found for {hostname}. Returning default values.")
            # Return a default array representing no detected companion
            return np.array([0.0, 10.0]) # [low_contrast, large_separation]

        # Use the first available data point
        contrast = imaging_data['im_contrast'][0]
        separation = imaging_data['im_sep'][0]
        
        # Handle missing data by providing sensible defaults
        if np.isnan(contrast): contrast = 0.0
        if np.isnan(separation): separation = 10.0

        return np.array([contrast, separation])

    except Exception as e:
        logger.error(f"Could not fetch or process imaging data for {hostname}: {e}")
        return None
