"""
Real astronomical data fetcher with robust caching for large-scale exoplanet detection.
"""

import os
import logging
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astroquery.mast import Observations, Catalogs
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

import config

logger = logging.getLogger(__name__)


def smart_data_fetcher(target_list, data_type="lightcurve", base_dir=None, force_download=False):
    """
    Smart data fetcher that checks for existing files before downloading.
    
    Args:
        target_list: List of target IDs or names
        data_type: Type of data to fetch ('lightcurve', 'target', etc.)
        base_dir: Base directory for data storage
        force_download: Whether to force download even if file exists
    
    Returns:
        dict: Mapping of target IDs to local file paths
    """
    # Set up directories
    base_dir = base_dir or config.DATA_DIR
    cache_dir = os.path.join(base_dir, f"{data_type}_data")
    metadata_dir = os.path.join(base_dir, "metadata")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Create cache index file path
    cache_index_file = os.path.join(metadata_dir, f"{data_type}_cache_index.json")
    
    # Load existing cache index if available
    cache_index = {}
    if os.path.exists(cache_index_file):
        with open(cache_index_file, 'r') as f:
            try:
                cache_index = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse cache index file {cache_index_file}, creating new index")
    
    # Process each target
    results = {}
    to_download = []
    
    for target_id in target_list:
        # Create a hash for the target ID to use in filename
        target_hash = hashlib.md5(str(target_id).encode()).hexdigest()
        local_path = os.path.join(cache_dir, f"{target_id}_{target_hash}.fits")
        
        # Check if in cache and file exists
        if not force_download and str(target_id) in cache_index and os.path.exists(cache_index[str(target_id)]):
            logger.debug(f"Using cached file for {target_id}: {cache_index[str(target_id)]}")
            results[target_id] = cache_index[str(target_id)]
        else:
            to_download.append(target_id)
            results[target_id] = local_path  # Will be overwritten if download fails
    
    # Download missing files
    if to_download:
        logger.info(f"Downloading {len(to_download)} files that aren't in cache")
        
        # This would be replaced with actual download code for Kepler/TESS
        downloaded = download_astronomical_data(to_download, data_type, cache_dir)
        
        # Update results and cache index with successful downloads
        for target_id, file_path in downloaded.items():
            if file_path and os.path.exists(file_path):
                results[target_id] = file_path
                cache_index[str(target_id)] = file_path
            else:
                logger.warning(f"Failed to download {target_id}")
                if target_id in results:
                    del results[target_id]
        
        # Save updated cache index
        with open(cache_index_file, 'w') as f:
            json.dump(cache_index, f, indent=2)
    
    return results


def download_astronomical_data(target_list, data_type="lightcurve", output_dir=None):
    """
    Download astronomical data for a list of targets.
    
    Args:
        target_list: List of target IDs to download
        data_type: Type of data to download ('lightcurve', 'target', etc.)
        output_dir: Directory to save downloaded files
    
    Returns:
        dict: Mapping of target IDs to local file paths
    """
    output_dir = output_dir or os.path.join(config.DATA_DIR, f"{data_type}_data")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for target_id in target_list:
        try:
            # Handle different data types
            if data_type == "lightcurve":
                # For Kepler/K2 targets
                if str(target_id).startswith(('K', 'k')) or str(target_id).isdigit():
                    file_path = download_kepler_lightcurve(target_id, output_dir)
                # For TESS targets
                elif str(target_id).startswith(('T', 't')):
                    file_path = download_tess_lightcurve(target_id, output_dir)
                else:
                    logger.warning(f"Unknown target ID format: {target_id}")
                    file_path = None
            else:
                logger.warning(f"Unsupported data type: {data_type}")
                file_path = None
            
            results[target_id] = file_path
        except Exception as e:
            logger.error(f"Error downloading data for {target_id}: {e}")
            results[target_id] = None
    
    return results


def download_kepler_lightcurve(kepler_id, output_dir):
    """
    Download Kepler light curve data for a specific target.
    
    Args:
        kepler_id: Kepler ID (KIC ID)
        output_dir: Directory to save the light curve file
    
    Returns:
        str: Path to downloaded file or None if download failed
    """
    try:
        # Convert ID format if needed
        query_id = str(kepler_id)
        if query_id.startswith(('KIC', 'kic')):
            query_id = query_id.replace('KIC', '').replace('kic', '').strip()
        
        # Query MAST for the target
        obs_table = Observations.query_criteria(
            target_name=f"KIC {query_id}",
            obs_collection="Kepler",
            dataproduct_type="timeseries"
        )
        
        if len(obs_table) == 0:
            logger.warning(f"No Kepler observations found for KIC {query_id}")
            return None
        
        # Get products for the first observation
        products = Observations.get_product_list(obs_table[0])
        
        # Filter for light curve files
        lc_products = [p for p in products if 'LIGHTCURVE' in p['dataURI']]
        
        if not lc_products:
            logger.warning(f"No light curve products found for KIC {query_id}")
            return None
        
        # Download the first light curve
        target_hash = hashlib.md5(str(kepler_id).encode()).hexdigest()
        local_path = os.path.join(output_dir, f"kplr{query_id}_{target_hash}_lc.fits")
        
        # Download using MAST download_file function
        download_path = Observations.download_file(lc_products[0]['dataURI'])
        
        # Move to our destination
        import shutil
        shutil.move(download_path, local_path)
        
        logger.info(f"Downloaded Kepler light curve for KIC {query_id} to {local_path}")
        return local_path
    
    except Exception as e:
        logger.error(f"Error downloading Kepler data for {kepler_id}: {e}")
        return None


def download_tess_lightcurve(tic_id, output_dir):
    """
    Download TESS light curve data for a specific target.
    
    Args:
        tic_id: TESS Input Catalog ID (TIC ID)
        output_dir: Directory to save the light curve file
    
    Returns:
        str: Path to downloaded file or None if download failed
    """
    try:
        # Convert ID format if needed
        query_id = str(tic_id)
        if query_id.startswith(('TIC', 'tic')):
            query_id = query_id.replace('TIC', '').replace('tic', '').strip()
        
        # Query MAST for the target
        obs_table = Observations.query_criteria(
            target_name=f"TIC {query_id}",
            obs_collection="TESS",
            dataproduct_type="timeseries"
        )
        
        if len(obs_table) == 0:
            logger.warning(f"No TESS observations found for TIC {query_id}")
            return None
        
        # Get products for the first observation
        products = Observations.get_product_list(obs_table[0])
        
        # Filter for light curve files
        lc_products = [p for p in products if ('LC' in p['dataURI'] or 'lc' in p['dataURI'])]
        
        if not lc_products:
            logger.warning(f"No light curve products found for TIC {query_id}")
            return None
        
        # Download the first light curve
        target_hash = hashlib.md5(str(tic_id).encode()).hexdigest()
        local_path = os.path.join(output_dir, f"tic{query_id}_{target_hash}_lc.fits")
        
        # Download using MAST download_file function
        download_path = Observations.download_file(lc_products[0]['dataURI'])
        
        # Move to our destination
        import shutil
        shutil.move(download_path, local_path)
        
        logger.info(f"Downloaded TESS light curve for TIC {query_id} to {local_path}")
        return local_path
    
    except Exception as e:
        logger.error(f"Error downloading TESS data for {tic_id}: {e}")
        return None


def get_random_target_sample(sample_size=1000, mission="Kepler"):
    """
    Get a random sample of target IDs from a specific mission.
    
    Args:
        sample_size: Number of targets to sample
        mission: Mission name ('Kepler' or 'TESS')
    
    Returns:
        list: List of target IDs
    """
    cache_file = os.path.join(config.METADATA_DIR, f"{mission.lower()}_targets.pkl")
    
    # Check for cached target list
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            all_targets = pickle.load(f)
        logger.info(f"Loaded {len(all_targets)} {mission} targets from cache")
    else:
        # Query for targets
        if mission.upper() == "KEPLER":
            try:
                # Try to use the KIC catalog instead
                logger.info("Querying Kepler Input Catalog (KIC)")
                kic_query = Catalogs.query_object("Kepler", radius=1.0, catalog="KIC")
                all_targets = [f"KIC {kid}" for kid in kic_query['ID']]
                
                # If that fails or returns no results, use a hardcoded list
                if len(all_targets) == 0:
                    raise ValueError("No KIC targets found")
                    
            except Exception as e:
                logger.warning(f"Error querying KIC catalog: {e}")
                logger.info("Using hardcoded list of Kepler targets")
                
                # Generate some synthetic Kepler IDs for testing
                # In a real scenario, you'd want to replace this with actual Kepler IDs
                all_targets = [f"KIC {10000000 + i}" for i in range(10000)]
                
        elif mission.upper() == "TESS":
            try:
                # Query TIC catalog
                logger.info("Querying TESS Input Catalog (TIC)")
                tic_query = Catalogs.query_criteria(catalog="TIC", objType="STAR", rows=sample_size*2)
                all_targets = [f"TIC {tid}" for tid in tic_query['ID']]
            except Exception as e:
                logger.warning(f"Error querying TIC catalog: {e}")
                logger.info("Using hardcoded list of TESS targets")
                
                # Generate some synthetic TESS IDs for testing
                all_targets = [f"TIC {200000000 + i}" for i in range(10000)]
        else:
            raise ValueError(f"Unknown mission: {mission}")
        
        # Cache the target list
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_targets, f)
        
        logger.info(f"Cached {len(all_targets)} {mission} targets")
    
    # Sample targets
    if sample_size >= len(all_targets):
        logger.warning(f"Requested sample size ({sample_size}) is larger than available targets ({len(all_targets)})")
        return all_targets
    
    sampled_targets = np.random.choice(all_targets, size=sample_size, replace=False).tolist()
    logger.info(f"Randomly sampled {len(sampled_targets)} {mission} targets")
    
    return sampled_targets


def load_rv_catalog(cache=True):
    """
    Load a catalog of RV-confirmed exoplanets for cross-validation.
    
    Args:
        cache: Whether to use cached catalog if available
    
    Returns:
        pd.DataFrame: Catalog of RV-confirmed planets
    """
    cache_file = os.path.join(config.METADATA_DIR, "rv_planets_catalog.csv")
    
    if cache and os.path.exists(cache_file):
        logger.info(f"Loading RV planet catalog from cache")
        return pd.read_csv(cache_file)
    
    logger.info("Fetching RV-confirmed planets from NASA Exoplanet Archive")
    
    # Query the NASA Exoplanet Archive for confirmed planets with RV data
    try:
        # Get all confirmed planets
        planets = NasaExoplanetArchive.query_criteria(
            table="ps",  # Planetary Systems table
            select="pl_name,hostname,ra,dec,pl_orbper,pl_masse,pl_rade,discoverymethod"
        )
        
        # Filter for RV-discovered planets or those with RV measurements
        rv_planets = planets[(planets['discoverymethod'] == 'Radial Velocity') | 
                           (planets['pl_masse'].notna())]
        
        # Create a simplified catalog with key properties
        rv_catalog = pd.DataFrame({
            'planet_name': rv_planets['pl_name'],
            'star_name': rv_planets['hostname'],
            'ra': rv_planets['ra'],
            'dec': rv_planets['dec'],
            'orbital_period': rv_planets['pl_orbper'],
            'planet_mass': rv_planets['pl_masse'],
            'planet_radius': rv_planets['pl_rade'],
            'discovery_method': rv_planets['discoverymethod']
        })
        
        # Cache the catalog
        rv_catalog.to_csv(cache_file, index=False)
        
        logger.info(f"Fetched and cached {len(rv_catalog)} RV-confirmed planets")
        return rv_catalog
        
    except Exception as e:
        logger.error(f"Error fetching RV planet catalog: {e}")
        
        # Return empty DataFrame if fetch fails
        return pd.DataFrame(columns=[
            'planet_name', 'star_name', 'ra', 'dec', 'orbital_period', 
            'planet_mass', 'planet_radius', 'discovery_method'
        ])


def extract_star_name(file_path):
    """
    Extract star name/ID from a file path.
    
    Args:
        file_path: Path to light curve file
    
    Returns:
        str: Star name or ID
    """
    file_name = os.path.basename(file_path)
    
    # Handle Kepler/K2 files
    if file_name.startswith('kplr'):
        # Format: kplr012345678-2013131215648_llc.fits
        # or:    kplr012345678_lc.fits
        kic_id = file_name.split('-')[0].replace('kplr', '')
        return f"KIC {kic_id}"
    
    # Handle TESS files
    elif file_name.startswith('tic'):
        # Format: tic123456789_lc.fits
        tic_id = file_name.split('_')[0].replace('tic', '')
        return f"TIC {tic_id}"
    
    # Generic fallback
    else:
        # Just return the filename without extension
        return os.path.splitext(file_name)[0]


def process_in_batches(file_list, batch_size=100, max_workers=8, processor_func=None):
    """
    Process light curves in batches with parallel execution.
    
    Args:
        file_list: List of light curve file paths
        batch_size: Number of files to process in each batch
        max_workers: Maximum number of parallel workers
        processor_func: Function to process each file (defaults to process_light_curve)
    
    Returns:
        list: Combined results from all batches
    """
    import math
    from concurrent.futures import ProcessPoolExecutor
    import pickle
    
    if processor_func is None:
        # Import here to avoid circular imports
        from pipeline.pipeline_runner import process_light_curve
        processor_func = process_light_curve
    
    total_batches = math.ceil(len(file_list) / batch_size)
    all_results = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(file_list))
        batch_files = file_list[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(batch_files)} files")
        
        batch_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(processor_func, file_path) for file_path in batch_files]
            
            # Process results as they complete
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result and result.get('success', False):
                        batch_results.append(result)
                    
                    # Log progress periodically
                    if (i+1) % 10 == 0 or i+1 == len(futures):
                        logger.info(f"  Processed {i+1}/{len(futures)} files in current batch")
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
        
        # Save batch results to avoid losing work
        batch_save_path = os.path.join(config.RESULTS_DIR, f"batch_results_{batch_num}.pkl")
        with open(batch_save_path, 'wb') as f:
            pickle.dump(batch_results, f)
        
        all_results.extend(batch_results)
        logger.info(f"Completed batch {batch_num+1}/{total_batches}, found {len(batch_results)} potential transits")
    
    return all_results
