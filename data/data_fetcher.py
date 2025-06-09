"""
Data fetching and generation utilities for the exoplanet detection pipeline.
This includes fetching data from astronomical archives (Kepler, TESS, Exoplanet Archive)
and generating synthetic light curve data for testing and training.
"""

import os
import logging
import hashlib
import pickle
import astroquery
from pathlib import Path
import time # For simple retry logic

import numpy as np
from astropy.io import fits
from astropy.table import Table
import pandas as pd

# Attempt to import astroquery, handle if not installed
try:
    from astroquery.mast import Observations
    from astroquery.ipac.nexsci.tap import TapPlus
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    Observations = None
    TapPlus = None
    logging.warning(
        "Astroquery library not found. Real data fetching capabilities will be limited. "
        "Please install astroquery (`pip install astroquery`)."
    )

# Attempt to import config from parent directory
try:
    from .. import config # Assumes data_fetcher.py is in data/ and config.py is in project root
except ImportError:
    # Fallback if running script directly from data/ or if structure is different
    # This might fail if config.py is not in Python's path.
    # A better approach is for the main script to pass config values or resolved paths.
    try:
        import config
    except ImportError:
        # Create a dummy config if it truly cannot be found, to avoid crashing,
        # but log a severe warning. Functions relying on it will likely fail or use hardcoded defaults.
        class DummyConfig:
            LIGHT_CURVE_DIR = Path("./data_files/light_curves")
            METADATA_DIR = Path("./data_files/metadata")
            EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"
            # Add other essential config variables with defaults if necessary
        config = DummyConfig()
        logging.error(
            "Could not import config.py. Using dummy config with default paths. "
            "Please ensure config.py is accessible."
        )


logger = logging.getLogger(__name__)

# --- Synthetic Data Generation ---
def generate_sample_light_curves(num_samples=10, output_dir=None, transit_prob=0.5, 
                                 time_steps=2000, observation_duration_days=100):
    """
    Generates synthetic light curve FITS files for testing and returns their paths and labels.

    Args:
        num_samples (int): Number of synthetic light curves to generate.
        output_dir (str or Path, optional): Directory to save the FITS files.
                                            Defaults to config.LIGHT_CURVE_DIR / "synthetic".
        transit_prob (float): Probability (0 to 1) that a generated light curve will contain a transit.
        time_steps (int): Number of data points in the light curve.
        observation_duration_days (float): Duration of the observation in days.

    Returns:
        list: A list of tuples, where each tuple is (filepath_str, label).
              Label is 1 if a transit was injected, 0 otherwise.
    """
    if output_dir is None:
        if hasattr(config, 'LIGHT_CURVE_DIR'):
            output_base_dir = Path(config.LIGHT_CURVE_DIR)
        else: # Fallback if config.LIGHT_CURVE_DIR is missing
            output_base_dir = Path(".") / "data_files" / "light_curves"
            logger.warning(f"config.LIGHT_CURVE_DIR not found, using default: {output_base_dir}")
    else:
        output_base_dir = Path(output_dir)

    synthetic_output_dir = output_base_dir / "synthetic"
    try:
        synthetic_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create synthetic output directory {synthetic_output_dir}: {e}")
        return [] # Return empty if directory cannot be made

    logger.info(f"Generating {num_samples} synthetic light curves in {synthetic_output_dir}")
    
    generated_files_with_labels = []

    for i in range(num_samples):
        kic_id_num = 100000000 + i # Ensure unique IDs
        kic_id_str = f"kplr{kic_id_num:09d}" 
        file_name = f"{kic_id_str}_lc.fits"
        file_path_obj = synthetic_output_dir / file_name
        file_path_str = str(file_path_obj)

        time_array = np.linspace(0, observation_duration_days, time_steps)
        flux_array = np.ones(time_steps)
        
        # Add Gaussian noise
        noise_level = np.random.uniform(0.0005, 0.002)
        flux_array += np.random.normal(0, noise_level, time_steps)

        # Add stellar variability (sine wave)
        if np.random.rand() < 0.7: # 70% chance of variability
            var_period = np.random.uniform(observation_duration_days / 10, observation_duration_days / 2)
            var_amplitude = np.random.uniform(0.001, 0.005)
            flux_array += var_amplitude * np.sin(2 * np.pi * time_array / var_period + np.random.rand() * 2 * np.pi)

        current_label = 0 # Default to no transit
        if np.random.rand() < transit_prob:
            current_label = 1
            # Add a transit signal
            # Ensure period is less than observation duration to see at least one transit
            transit_period_days = np.random.uniform(1, observation_duration_days / 2) 
            transit_duration_hours = np.random.uniform(1, 5) # hours
            transit_duration_days_val = transit_duration_hours / 24.0
            transit_depth_val = np.random.uniform(0.0005, 0.01) # relative depth
            
            t0_epoch = np.random.uniform(0, transit_period_days) # First transit epoch
            
            for n_transit in range(int(np.ceil(observation_duration_days / transit_period_days)) + 1):
                transit_mid_time = t0_epoch + n_transit * transit_period_days
                start_transit = transit_mid_time - (transit_duration_days_val / 2)
                end_transit = transit_mid_time + (transit_duration_days_val / 2)
                
                if start_transit > observation_duration_days or end_transit < 0:
                    continue # Transit outside observation window

                transit_mask = (time_array >= start_transit) & (time_array <= end_transit)
                flux_array[transit_mask] -= transit_depth_val # Simple box transit

        # Create FITS file structure
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TELESCOP'] = ('SYNTHETIC', 'Simulated data')
        primary_hdu.header['KEPLERID'] = (kic_id_num, 'Simulated Kepler ID') # Store numeric part
        primary_hdu.header['OBJECT'] = (kic_id_str, 'Object name')
        primary_hdu.header['LABEL'] = (current_label, '0=no_transit/noise, 1=planet_transit')
        primary_hdu.header['ORIGIN'] = ('ExoPipelineSim', 'File Origin')

        col_time = fits.Column(name='TIME', format='D', array=time_array, unit='BJD - 2454833')
        col_pdcflux = fits.Column(name='PDCSAP_FLUX', format='D', array=flux_array)
        col_sapflux = fits.Column(name='SAP_FLUX', format='D', array=flux_array.copy()) # Example
        
        # Add an error column (e.g., based on noise level)
        flux_err_array = np.full_like(flux_array, noise_level) 
        col_pdcflux_err = fits.Column(name='PDCSAP_FLUX_ERR', format='D', array=flux_err_array)

        cols = fits.ColDefs([col_time, col_pdcflux, col_sapflux, col_pdcflux_err])
        
        try:
            data_hdu = fits.BinTableHDU.from_columns(cols)
            hdul = fits.HDUList([primary_hdu, data_hdu])
            hdul.writeto(file_path_str, overwrite=True)
            logger.info(f"Generated synthetic light curve: {file_path_str} with label: {current_label}")
            generated_files_with_labels.append((file_path_str, current_label))
        except Exception as e:
            logger.error(f"Failed to write synthetic FITS file {file_path_str}: {e}", exc_info=True)

    return generated_files_with_labels


# --- Caching Utilities ---
def _get_cache_path(cache_dir, query_params_dict, filename_prefix="cache"):
    """Generates a hash-based cache file path."""
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a stable string from parameters for hashing
    param_str = "".join(f"{k}:{v}" for k, v in sorted(query_params_dict.items()))
    query_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    cache_filename = f"{filename_prefix}_{query_hash}.pkl"
    return cache_dir_path / cache_filename

def _load_from_cache(cache_filepath):
    """Loads data from a pickle cache file."""
    if cache_filepath.exists():
        try:
            with open(cache_filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded data from cache: {cache_filepath}")
            return data
        except Exception as e:
            logger.warning(f"Could not load from cache file {cache_filepath}: {e}. Will re-fetch.")
    return None

def _save_to_cache(data, cache_filepath):
    """Saves data to a pickle cache file."""
    try:
        with open(cache_filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved data to cache: {cache_filepath}")
    except Exception as e:
        logger.error(f"Could not save to cache file {cache_filepath}: {e}")


# --- Real Data Fetching Functions ---

def fetch_kepler_data(target_name=None, mission="Kepler", max_records=50, use_cache=True,
                      dataproduct_type="timeseries", obs_collection=None):
    """
    Fetches observation data (metadata, not files) from MAST for Kepler/TESS.

    Args:
        target_name (str, optional): Specific target name (e.g., "Kepler-62").
        mission (str or list, optional): Mission(s) to query (e.g., "Kepler", "TESS", ["Kepler", "TESS"]).
        max_records (int): Maximum number of observation records to return.
        use_cache (bool): Whether to use local caching for query results.
        dataproduct_type (str): Type of data product (e.g., "timeseries", "image").
        obs_collection (str or list, optional): Specific observation collection(s).
                                                Defaults to mission name if None.

    Returns:
        astropy.table.Table or None: Table of observation records, or None if error/no data.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not installed. Cannot fetch Kepler/TESS data.")
        return None

    query_params = {
        "target_name": target_name, "mission": mission, "max_records": max_records,
        "dataproduct_type": dataproduct_type, "obs_collection": obs_collection or mission
    }
    cache_file = _get_cache_path(config.METADATA_DIR, query_params, "mast_obs_table")

    if use_cache:
        cached_data = _load_from_cache(cache_file)
        if cached_data is not None:
            return cached_data
    
    logger.info(f"Querying MAST for {mission} observations. Target: {target_name or 'any'}, Max: {max_records}")
    try:
        criteria = {"obs_collection": obs_collection or mission, "dataproduct_type": dataproduct_type}
        if target_name:
            criteria["target_name"] = target_name
        
        obs_table = Observations.query_criteria(**criteria)
        
        if not obs_table:
            logger.warning(f"No observations found for criteria: {criteria}")
            _save_to_cache(None, cache_file) # Cache empty result
            return None
        
        # Astroquery might return more than max_records before slicing,
        # so slice after getting the table.
        obs_table_sliced = obs_table[:max_records]
        
        if use_cache:
            _save_to_cache(obs_table_sliced, cache_file)
        return obs_table_sliced
    except Exception as e:
        logger.error(f"Error querying MAST: {e}", exc_info=True)
        return None


def download_light_curves(obs_table, output_dir=None, use_cache=True, max_workers=4, max_downloads=None):
    """
    Downloads light curve files for given observations from MAST.

    Args:
        obs_table (astropy.table.Table): Table of observations (from fetch_kepler_data).
        output_dir (str or Path, optional): Directory to save downloaded files.
                                           Defaults to config.LIGHT_CURVE_DIR.
        use_cache (bool): If True, skips download if file already exists locally.
        max_workers (int): Number of parallel downloads (currently serial, placeholder).
        max_downloads (int, optional): Maximum number of files to download.

    Returns:
        list: List of local file paths to the downloaded FITS files.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not installed. Cannot download light curves.")
        return []
    if obs_table is None or len(obs_table) == 0:
        logger.warning("No observation table provided to download_light_curves.")
        return []

    if output_dir is None:
        if hasattr(config, 'LIGHT_CURVE_DIR'):
            dl_dir = Path(config.LIGHT_CURVE_DIR)
        else:
            dl_dir = Path(".") / "data_files" / "light_curves"
            logger.warning(f"config.LIGHT_CURVE_DIR not found, using default: {dl_dir}")
    else:
        dl_dir = Path(output_dir)
    
    dl_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    download_count = 0

    for obs_row in obs_table:
        if max_downloads is not None and download_count >= max_downloads:
            logger.info(f"Reached maximum download limit of {max_downloads}.")
            break
        try:
            # Construct a unique local filename
            # Example: KIC12345678_lc.fits or tess2019..._lc.fits
            target_name = obs_row.get('target_name', obs_row.get('obsid', f"unknown_{download_count}")).replace(" ", "_")
            filename_base = f"{target_name}_mast_lc" 
            
            # Get product list for the observation
            products = Observations.get_product_urls(obs_row['obsid'], productType="TIMESERIES")
            
            if not products:
                logger.warning(f"No timeseries products found for obsid: {obs_row['obsid']} (target: {target_name})")
                continue

            # Prioritize FITS light curves, prefer _lc.fits if available
            fits_products = [p for p in products if p.lower().endswith(('.fits', '.fits.gz'))]
            if not fits_products:
                logger.warning(f"No FITS timeseries products found for obsid: {obs_row['obsid']}")
                continue
            
            # Try to find a product that looks like a primary light curve
            # This logic can be quite mission-specific.
            # For Kepler, 'slc.fits' (short cadence) or 'llc.fits' (long cadence)
            # For TESS, often contains 'lc.fits'
            chosen_product_url = None
            preferred_suffixes = ["_lc.fits", "slc.fits", "llc.fits", ".fits"] # Order of preference
            
            for suffix in preferred_suffixes:
                for p_url in fits_products:
                    if p_url.lower().endswith(suffix):
                        chosen_product_url = p_url
                        # Try to make filename more specific from URL
                        url_basename = chosen_product_url.split('/')[-1]
                        # Remove query params if any
                        url_basename = url_basename.split('?')[0]
                        filename_base = Path(url_basename).stem 
                        break
                if chosen_product_url:
                    break
            
            if not chosen_product_url: # Fallback to first FITS product
                chosen_product_url = fits_products[0]
                filename_base = Path(chosen_product_url.split('/')[-1]).stem


            local_filepath = dl_dir / f"{filename_base}.fits" # Ensure .fits extension

            if use_cache and local_filepath.exists():
                logger.info(f"Using cached file: {local_filepath}")
                downloaded_files.append(str(local_filepath))
                download_count += 1
                continue

            logger.info(f"Downloading product for obsid {obs_row['obsid']} (target: {target_name}) from {chosen_product_url} to {local_filepath}")
            
            # Astroquery's download_file handles the actual download and saving.
            # It returns the path to the downloaded file.
            # We might want to rename it or move it if its default location/name isn't what we want.
            temp_download_path_str = Observations.download_file(chosen_product_url, local_path=str(local_filepath))
            
            if temp_download_path_str and Path(temp_download_path_str).exists():
                # If download_file saved it directly to local_filepath, great.
                # If it saved elsewhere (e.g. cache), we'd need to move it.
                # Assuming download_file with local_path argument saves it where specified.
                final_path = Path(temp_download_path_str)
                if final_path.resolve() != local_filepath.resolve():
                    # This case should ideally not happen if local_path is respected by download_file
                    # but as a fallback, attempt to move.
                    logger.warning(f"Downloaded file to {final_path}, moving to {local_filepath}")
                    try:
                        final_path.replace(local_filepath) # Move and overwrite if exists
                    except Exception as move_e:
                        logger.error(f"Failed to move {final_path} to {local_filepath}: {move_e}")
                        continue # Skip this file if move fails

                downloaded_files.append(str(local_filepath))
                download_count += 1
                time.sleep(0.1) # Small delay to be polite to server
            else:
                logger.error(f"Download failed or file not found for {chosen_product_url}")

        except Exception as e:
            logger.error(f"Error processing download for obsid {obs_row.get('obsid','N/A')}: {e}", exc_info=True)
            
    logger.info(f"Downloaded {len(downloaded_files)} light curve files to {dl_dir}.")
    return downloaded_files


def fetch_exoplanet_labels(use_cache=True, service_url=None):
    """
    Fetches confirmed exoplanet data (Planetary Systems table) from NASA Exoplanet Archive TAP service.

    Args:
        use_cache (bool): Whether to use local caching for query results.
        service_url (str, optional): URL for the TAP service. Defaults to config.EXOPLANET_ARCHIVE_TAP_URL.

    Returns:
        pandas.DataFrame or None: DataFrame of exoplanet labels, or None if error.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not installed. Cannot fetch exoplanet labels.")
        return None

    if service_url is None:
        if hasattr(config, 'EXOPLANET_ARCHIVE_TAP_URL'):
            service_url = config.EXOPLANET_ARCHIVE_TAP_URL
        else:
            logger.error("Exoplanet Archive TAP URL not found in config. Cannot fetch labels.")
            return None
            
    query_params = {"service": "ExoplanetArchivePlanetarySystems"} # Simple key for this query
    cache_file = _get_cache_path(config.METADATA_DIR, query_params, "exoplanet_labels_ps")

    if use_cache:
        cached_data_df = _load_from_cache(cache_file)
        if cached_data_df is not None and isinstance(cached_data_df, pd.DataFrame):
            return cached_data_df
    
    logger.info("Fetching exoplanet data from NASA Exoplanet Archive TAP service (Planetary Systems Table)")
    try:
        tap = TapPlus(url=service_url)
        # Query the Planetary Systems (PS) table for comprehensive data.
        # Select columns relevant for identifying hosts and basic planet properties.
        # Prioritize default parameters (pl_def_reflab=1)
        query = """
        SELECT hostname, pl_name, discoverymethod, disc_year, sy_dist,
               pl_orbper, pl_orbsmax, pl_rade, pl_masse, pl_eqt,
               st_rad, st_mass, st_teff, default_flag
        FROM ps
        WHERE default_flag = 1 
        ORDER BY hostname
        """ 
        # Using `default_flag = 1` gets the archive's preferred parameters for each planet.
        # Consider also querying `pscomppars` for composite parameters if needed.
        
        job = tap.launch_job(query)
        exoplanet_data_table = job.get_results() # astropy.table.Table
        exoplanet_data_df = exoplanet_data_table.to_pandas()
        
        if use_cache:
            _save_to_cache(exoplanet_data_df, cache_file)
        return exoplanet_data_df
    except Exception as e:
        logger.error(f"Error fetching exoplanet labels from TAP service: {e}", exc_info=True)
        return None


def get_kepler_koi_targets(use_cache=True, service_url=None):
    """
    Fetches Kepler Objects of Interest (KOI) table and categorizes them.

    Args:
        use_cache (bool): Whether to use local caching.
        service_url (str, optional): URL for the TAP service.

    Returns:
        tuple: (list_of_confirmed_koi_hostnames, list_of_false_positive_koi_hostnames)
               Returns (None, None) on error.
    """
    if not ASTROQUERY_AVAILABLE:
        logger.error("Astroquery is not installed. Cannot fetch KOI targets.")
        return None, None

    if service_url is None:
        if hasattr(config, 'EXOPLANET_ARCHIVE_TAP_URL'):
            service_url = config.EXOPLANET_ARCHIVE_TAP_URL
        else:
            logger.error("Exoplanet Archive TAP URL not found in config. Cannot fetch KOIs.")
            return None, None

    query_params = {"service": "KeplerKOITable"}
    cache_file = _get_cache_path(config.METADATA_DIR, query_params, "koi_table")

    koi_df = None
    if use_cache:
        cached_data_df = _load_from_cache(cache_file)
        if cached_data_df is not None and isinstance(cached_data_df, pd.DataFrame):
            koi_df = cached_data_df
    
    if koi_df is None:
        logger.info("Fetching Kepler Objects of Interest (KOI) table from NASA Exoplanet Archive")
        try:
            tap = TapPlus(url=service_url)
            # Query the KOI Cumulative table (kic_kepname for host, koi_disposition for status)
            query = """
            SELECT kepid, koiname, koi_disposition, koi_period, koi_ror, koi_depth, koi_duration, kic_kepmag, kic_teff, kic_rad
            FROM cumulative
            """
            # koi_disposition can be 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'
            job = tap.launch_job(query)
            koi_table_astro = job.get_results()
            koi_df = koi_table_astro.to_pandas()
            
            if use_cache:
                _save_to_cache(koi_df, cache_file)
        except Exception as e:
            logger.error(f"Error fetching KOI table: {e}", exc_info=True)
            return None, None

    if koi_df is None or koi_df.empty:
        logger.warning("KOI DataFrame is empty or could not be fetched.")
        return [], []

    # Extract KIC IDs (KEPLERIDs) as the primary host identifier from KOI table
    # The 'kepid' column usually stores this.
    confirmed_koi_hosts = koi_df[koi_df['koi_disposition'] == 'CONFIRMED']['kepid'].astype(str).unique().tolist()
    fp_koi_hosts = koi_df[koi_df['koi_disposition'] == 'FALSE POSITIVE']['kepid'].astype(str).unique().tolist()
    
    # Convert KIC IDs to string and prepend "KIC " or "kplr" if needed for matching other data sources
    # This depends on how your other target lists are formatted.
    # For now, returning raw KIC IDs as strings.
    
    logger.info(f"Found {len(confirmed_koi_hosts)} unique hosts for CONFIRMED KOIs.")
    logger.info(f"Found {len(fp_koi_hosts)} unique hosts for FALSE POSITIVE KOIs.")
    
    return confirmed_koi_hosts, fp_koi_hosts


def fetch_kepler_false_positives(max_count=500, use_cache=True, service_url=None):
    """
    Fetches a list of Kepler IDs known to be false positives from the KOI table.
    This is a utility function that directly returns file paths if they can be constructed
    or just the Kepler IDs if file paths are not directly derivable.
    For this example, it will return Kepler IDs that can then be used to download light curves.
    
    Args:
        max_count (int): Maximum number of false positive Kepler IDs to return.
        use_cache (bool): Whether to use cached KOI table data.
        service_url (str, optional): URL for the TAP service.

    Returns:
        list: List of Kepler IDs (strings) identified as false positives.
    """
    _, fp_koi_hosts_kic_ids = get_kepler_koi_targets(use_cache=use_cache, service_url=service_url)
    
    if fp_koi_hosts_kic_ids is None:
        return []
        
    # Convert KIC IDs to a format that might be used for MAST queries, e.g., "KIC 1234567"
    # The `fp_koi_hosts_kic_ids` from get_kepler_koi_targets are already strings of KIC IDs.
    # If you need to format them (e.g. "KIC " + id), do it here.
    # For now, assume they are direct Kepler IDs.
    
    if len(fp_koi_hosts_kic_ids) > max_count:
        # Optionally, you could sort or sample, here just taking the first max_count
        selected_fp_kic_ids = fp_koi_hosts_kic_ids[:max_count]
        logger.info(f"Selected {len(selected_fp_kic_ids)} false positive Kepler IDs (max_count: {max_count}).")
    else:
        selected_fp_kic_ids = fp_koi_hosts_kic_ids
        logger.info(f"Selected {len(selected_fp_kic_ids)} false positive Kepler IDs.")

    # This function now returns KIC IDs. The calling function (e.g., main.py)
    # would then use these IDs to fetch observation tables and download light curves.
    return selected_fp_kic_ids


if __name__ == '__main__':
    # Example usage (primarily for testing this module)
    logging.basicConfig(level=logging.INFO) # Setup basic logging for direct run
    
    # Test synthetic data generation
    print("\n--- Testing Synthetic Data Generation ---")
    # Ensure config.LIGHT_CURVE_DIR is set if generate_sample_light_curves relies on it for default output
    # Or pass an explicit output_dir
    test_synthetic_output_dir = Path("./test_synthetic_output")
    synthetic_files = generate_sample_light_curves(num_samples=5, output_dir=test_synthetic_output_dir, transit_prob=0.6)
    if synthetic_files:
        print(f"Generated {len(synthetic_files)} synthetic files with labels:")
        for fpath, flabel in synthetic_files:
            print(f"  File: {fpath}, Label: {flabel}")
            # Basic check if FITS file is readable
            try:
                with fits.open(fpath) as hdul:
                    print(f"    Primary HDU Header Keywords (first 5): {list(hdul[0].header.keys())[:5]}")
                    if len(hdul)>1 and hasattr(hdul[1],'data') and hdul[1].data is not None:
                         print(f"    Data HDU has {len(hdul[1].data)} rows.")
            except Exception as e_fits:
                print(f"    Error reading generated FITS {fpath}: {e_fits}")
    else:
        print("No synthetic files generated.")

    if ASTROQUERY_AVAILABLE:
        # Test fetching Kepler observation metadata
        print("\n--- Testing Kepler Observation Fetch (Metadata) ---")
        kepler_obs = fetch_kepler_data(target_name="Kepler-62", max_records=2)
        if kepler_obs:
            print(f"Fetched {len(kepler_obs)} observations for Kepler-62:")
            print(kepler_obs['obsid', 'target_name', 't_exptime', 'filters'])
            
            # Test downloading light curves for these observations
            print("\n--- Testing Light Curve Download ---")
            # Create a temporary download directory for this test
            test_lc_download_dir = Path("./test_lc_downloads")
            downloaded_lcs = download_light_curves(kepler_obs, output_dir=test_lc_download_dir, max_downloads=1)
            if downloaded_lcs:
                print(f"Downloaded {len(downloaded_lcs)} light curve(s):")
                for lc_path in downloaded_lcs:
                    print(f"  {lc_path}")
            else:
                print("No light curves downloaded in test.")
        else:
            print("Could not fetch Kepler observations for test.")

        # Test fetching exoplanet labels
        print("\n--- Testing Exoplanet Label Fetch ---")
        labels_df = fetch_exoplanet_labels(use_cache=False) # Force fetch for test
        if labels_df is not None and not labels_df.empty:
            print(f"Fetched {len(labels_df)} exoplanet catalog entries. Columns: {labels_df.columns.tolist()}")
            print("Sample (first 3 confirmed planets):")
            print(labels_df[labels_df['default_flag'] == 1].head(3))
        else:
            print("Could not fetch exoplanet labels.")

        # Test fetching KOI targets
        print("\n--- Testing KOI Target Fetch ---")
        confirmed_kois, fp_kois = get_kepler_koi_targets(use_cache=False) # Force fetch
        if confirmed_kois is not None:
            print(f"Found {len(confirmed_kois)} unique hosts for CONFIRMED KOIs (sample: {confirmed_kois[:3]})")
        if fp_kois is not None:
            print(f"Found {len(fp_kois)} unique hosts for FALSE POSITIVE KOIs (sample: {fp_kois[:3]})")
            
        # Test fetching Kepler False Positives (KIC IDs)
        print("\n--- Testing Kepler False Positive ID Fetch ---")
        fp_kic_ids = fetch_kepler_false_positives(max_count=5, use_cache=False)
        if fp_kic_ids:
            print(f"Fetched {len(fp_kic_ids)} Kepler False Positive KIC IDs: {fp_kic_ids}")
        else:
            print("Could not fetch Kepler False Positive KIC IDs.")
    else:
        print("\nSkipping real data fetching tests as astroquery is not available.")
