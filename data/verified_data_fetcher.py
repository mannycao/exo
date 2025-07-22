# data/verified_data_fetcher.py

import os
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np

import config
from utils.dependencies import ASTROQUERY_AVAILABLE, Observations, NasaExoplanetArchive

logger = logging.getLogger(__name__)


def fetch_verified_transit_data(num_transits=50, use_cache=True):
    """Fetch real confirmed transit light curves."""
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery not available. Cannot fetch transit data.")
        return []

    cache_dir = Path(config.DATA_DIR) / "verified_transits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # This function's logic is generally fine as the list of confirmed planets is smaller.
    # For consistency, we could add shuffling here too, but the main issue is with non-transits.
    # ... (code for fetching transits remains the same) ...
    # This is a placeholder for your existing transit fetching logic which is okay.
    return []


def fetch_verified_non_transit_data(num_non_transits=50, use_cache=True):
    """
    Fetch verified non-transit light curves (e.g., false positives) efficiently
    by shuffling the candidate list before downloading.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery not available. Cannot fetch non-transit data.")
        return []

    cache_dir = Path(config.DATA_DIR) / "verified_non_transits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Querying Kepler Objects of Interest for {num_non_transits} false positives...")
        
        false_positives_table = NasaExoplanetArchive.query_criteria(
            table="koi",
            select="kepid, kepoi_name",
            where="koi_disposition='FALSE POSITIVE'"
        )
        
        if false_positives_table is None:
            logger.error("Failed to query for false positives.")
            return []

        # --- THIS IS THE FIX ---
        # Convert to a list of dictionaries and shuffle it randomly
        candidate_list = false_positives_table.to_pandas().to_dict('records')
        np.random.shuffle(candidate_list)
        logger.info(f"Found and shuffled {len(candidate_list)} total potential false positive candidates.")
        # ^^^^^^^^^^^^^^^^^^^^^^^
        
        non_transit_files = []
        count = 0
        
        # Iterate through the SHUFFLED list
        for i, row in enumerate(candidate_list):
            if count >= num_non_transits:
                logger.info(f"Reached target of {num_non_transits} non-transit files. Stopping download.")
                break
                
            try:
                kep_id = row['kepid']
                
                logger.info(f"Fetching light curve for KIC {kep_id} (Goal: {count+1}/{num_non_transits})")
                
                output_file = cache_dir / f"KIC_{kep_id}_false_positive.fits"
                if use_cache and output_file.exists():
                    logger.debug(f"Found cached file for KIC {kep_id}")
                    non_transit_files.append(str(output_file))
                    count += 1
                    continue

                obs = Observations.query_criteria(
                    target_name=f"KIC {kep_id}",
                    obs_collection='Kepler',
                    dataproduct_type="timeseries"
                )
                
                if len(obs) > 0:
                    products = Observations.get_product_list(obs[0])
                    lc_products = [p for p in products if 'LIGHTCURVE' in p['dataURI']]
                    
                    if lc_products:
                        Observations.download_file(lc_products[0]['dataURI'], local_path=str(output_file))
                        non_transit_files.append(str(output_file))
                        count += 1
                        time.sleep(1) # Be kind to the server
            
            except Exception as e:
                logger.error(f"Error fetching light curve for KIC {kep_id}: {e}")

        logger.info(f"Successfully downloaded {len(non_transit_files)} verified non-transit light curves.")
        return non_transit_files
        
    except Exception as e:
        logger.error(f"A major error occurred while fetching non-transit data: {e}", exc_info=True)
        return []

# You should also have fetch_verified_training_data, etc. in this file.
# The change above is the only one needed to fix the performance issue.