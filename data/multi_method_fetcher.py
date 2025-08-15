# data/multi_method_fetcher.py

import logging
import pandas as pd
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import numpy as np
from pathlib import Path

# Setup logger for this module
logger = logging.getLogger(__name__)

def get_target_coords(kepler_id):
    """Gets the celestial coordinates (RA, Dec) for a given Kepler ID, using a local cache for performance."""
    cache_path = Path("results/keplerstellar_cache.csv")
    if not cache_path.exists():
        logger.info("Downloading full Kepler stellar table for coordinate cache (one-time operation)...")
        try:
            stellar_info = NasaExoplanetArchive.query_criteria(
                table="keplerstellar",
                select="kepid, ra, dec"
            )
            df = stellar_info.to_pandas()
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached Kepler stellar table to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to download Kepler stellar table: {e}")
            return None, None
    else:
        df = pd.read_csv(cache_path)
    row = df[df['kepid'] == kepler_id]
    if not row.empty:
        ra, dec = float(row.iloc[0]['ra']), float(row.iloc[0]['dec'])
        logger.info(f"Found coordinates for KIC {kepler_id}: RA={ra}, Dec={dec} (from cache)")
        return ra, dec
    else:
        logger.warning(f"No coordinates found for KIC {kepler_id} in cached keplerstellar table.")
        return None, None

def fetch_rv_data(ra, dec, rv_timeseries_length=100):
    """Fetches radial velocity data via a cone search around the target's coordinates."""
    try:
        # Perform a cone search in the RV table around the target's location
        # Increased cone search radius to 0.01 degrees
        rv_table = NasaExoplanetArchive.query_criteria(
            table="rvm_fw_curves",
            where=f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01))" # Increased radius
        )
        if len(rv_table) > 0:
            rv_df = rv_table.to_pandas()
            rv_df = rv_df[['jd', 'detrended_rv']].dropna().sort_values('jd')
            
            if len(rv_df) > 1:
                time_norm = (rv_df['jd'] - rv_df['jd'].min()) / (rv_df['jd'].max() - rv_df['jd'].min())
                rv_norm = (rv_df['detrended_rv'] - rv_df['detrended_rv'].mean()) / rv_df['detrended_rv'].std()
                
                return rv_interpolated.reshape(-1, 1), 1.0 # Return 1.0 for mask if data exists
    except Exception:
        pass # Fail silently on individual query errors
    return np.zeros((rv_timeseries_length, 1)), 0.0 # Return 0.0 for mask if no data

def fetch_imaging_data(ra, dec):
    """Fetches high-resolution imaging data via a cone search."""
    try:
        imaging_table = NasaExoplanetArchive.query_criteria(
            table="exofop_contr_curves",
            where=f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01))" # Increased radius
        )
        if len(imaging_table) > 0:
            img_df = imaging_table.to_pandas()
            mean_contrast = img_df['delta_mag'].mean()
            mean_separation = img_df['separation_arcsec'].mean()
            return np.array([mean_contrast, mean_separation]), 1.0 # Return 1.0 for mask
    except Exception:
        pass
    return np.zeros(2), 0.0 # Return 0.0 for mask

def fetch_all_data_for_target(file_path):
    """Orchestrates fetching all data types for a single target."""
    try:
        filename = Path(file_path).name
        kepler_id = int(filename.split('-')[0].replace('kplr', ''))
    except (ValueError, IndexError) as e:
        logger.error(f"Could not parse Kepler ID from file path: {file_path} ({e})")
        return None

    ra, dec = get_target_coords(kepler_id)
    if ra is None or dec is None:
        logger.warning(f"Skipping target: Could not find coordinates for KIC {kepler_id} (file: {file_path})")
        return None # Cannot proceed without coordinates

    rv_data, rv_mask = fetch_rv_data(ra, dec)
    imaging_data, imaging_mask = fetch_imaging_data(ra, dec)

    return {
        "kepler_id": kepler_id,
        "transit_file_path": file_path,
        "rv_data": rv_data,
        "rv_mask": rv_mask,
        "imaging_data": imaging_data,
        "imaging_mask": imaging_mask
    }