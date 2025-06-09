"""
Functions for fetching verified transit and non-transit data from astronomical databases.
"""

import os
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import time

import config
from data.light_curve_processor import preprocess_light_curve

logger = logging.getLogger(__name__)


def fetch_verified_transit_data(num_transits=1000, use_cache=True):
    """
    Fetch real confirmed transit light curves from astronomical databases.
    
    Args:
        num_transits: Number of verified transit light curves to fetch
        use_cache: Whether to use cached data if available
    
    Returns:
        list: Paths to downloaded light curve files
    """
    cache_dir = os.path.join(config.DATA_DIR, "verified_transits")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(config.METADATA_DIR, "verified_transits.pkl")
    
    # Check if we have cached file paths
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached verified transit file paths")
        import pickle
        with open(cache_file, 'rb') as f:
            transit_files = pickle.load(f)
            
        # Verify files still exist
        existing_files = [f for f in transit_files if os.path.exists(f)]
        if len(existing_files) >= num_transits:
            logger.info(f"Found {len(existing_files)} cached verified transit files")
            return existing_files[:num_transits]
        else:
            logger.info(f"Found only {len(existing_files)} valid files from cache, need {num_transits}")
    
    # If we don't have enough cached files, fetch from the NASA Exoplanet Archive
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        from astroquery.mast import Observations, Catalogs

        logger.info(f"Querying NASA Exoplanet Archive for confirmed transiting planets")
        
        # Get confirmed transiting planets with Kepler IDs
        confirmed = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name,hostname,ra,dec,pl_orbper,pl_rade,pl_trandep,st_kepmag,st_ticid",
            where="discoverymethod='Transit' and st_kepmag is not null"
        )
        
        logger.info(f"Found {len(confirmed)} confirmed transiting planets with Kepler magnitudes")
        
        # Filter to planets with valid periods and depths
        confirmed = confirmed[confirmed['pl_orbper'].mask == False]
        confirmed = confirmed[confirmed['pl_trandep'].mask == False]
        
        logger.info(f"Filtered to {len(confirmed)} planets with valid periods and transit depths")
        
        # Initialize list for downloaded files
        transit_files = []
        
        # Loop through planets and download light curves
        count = 0
        for i, row in enumerate(confirmed):
            if count >= num_transits:
                break
                
            try:
                hostname = row['hostname']
                logger.info(f"Fetching light curve for {hostname} ({i+1}/{len(confirmed)})")
                
                # Try to get by Kepler ID first
                if not pd.isna(row['st_kepmag']):
                    obj_name = hostname.replace(' ', '')
                    
                    # Query MAST for light curves
                    obs = Observations.query_criteria(
                        target_name=obj_name,
                        obs_collection='Kepler',
                        dataproduct_type="timeseries"
                    )
                    
                    if len(obs) == 0:
                        # Try with KIC prefix
                        if 'KIC' not in obj_name and 'KIC' not in hostname:
                            kic_id = hostname.replace('Kepler-', '').replace('kepler', '').strip()
                            obs = Observations.query_criteria(
                                target_name=f"KIC {kic_id}",
                                obs_collection='Kepler',
                                dataproduct_type="timeseries"
                            )
                    
                    if len(obs) > 0:
                        products = Observations.get_product_list(obs[0])
                        lc_products = [p for p in products if 'LIGHTCURVE' in p['dataURI']]
                        
                        if lc_products:
                            # Create a filename for consistency
                            target_id = hostname.replace(' ', '_')
                            output_file = os.path.join(cache_dir, f"{target_id}_transit.fits")
                            
                            # Download if not already exists
                            if not os.path.exists(output_file):
                                download_path = Observations.download_file(lc_products[0]['dataURI'])
                                
                                # Move to our designated location
                                os.rename(download_path, output_file)
                                
                            transit_files.append(output_file)
                            count += 1
                            logger.info(f"Downloaded light curve for {hostname} - {count}/{num_transits}")
                            
                            # Add a small delay to avoid overwhelming the server
                            time.sleep(1)
                
                # If we couldn't get Kepler data, try TESS
                elif not pd.isna(row['st_ticid']):
                    tic_id = row['st_ticid']
                    
                    # Query MAST for TESS light curves
                    obs = Observations.query_criteria(
                        target_name=f"TIC {int(tic_id)}",
                        obs_collection='TESS',
                        dataproduct_type="timeseries"
                    )
                    
                    if len(obs) > 0:
                        products = Observations.get_product_list(obs[0])
                        lc_products = [p for p in products if 'LC' in p['dataproduct_subtype']]
                        
                        if lc_products:
                            # Create a filename for consistency
                            target_id = f"TIC_{int(tic_id)}"
                            output_file = os.path.join(cache_dir, f"{target_id}_transit.fits")
                            
                            # Download if not already exists
                            if not os.path.exists(output_file):
                                download_path = Observations.download_file(lc_products[0]['dataURI'])
                                
                                # Move to our designated location
                                os.rename(download_path, output_file)
                                
                            transit_files.append(output_file)
                            count += 1
                            logger.info(f"Downloaded light curve for {hostname} - {count}/{num_transits}")
                            
                            # Add a small delay to avoid overwhelming the server
                            time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching light curve for {hostname}: {e}")
                # Continue to next planet
        
        # Cache the downloaded file paths
        if transit_files:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(transit_files, f)
            
        logger.info(f"Successfully downloaded {len(transit_files)} verified transit light curves")
        return transit_files[:num_transits]
        
    except ImportError:
        logger.error("Required modules not available: astroquery")
        # Fallback to any existing files
        transit_files = [f for f in os.listdir(cache_dir) if f.endswith('.fits')]
        transit_files = [os.path.join(cache_dir, f) for f in transit_files]
        
        if not transit_files:
            logger.error("No verified transit files available and couldn't fetch from database")
            return []
            
        return transit_files[:num_transits]
    except Exception as e:
        logger.error(f"Error fetching verified transit data: {e}")
        # Fallback to any existing files
        transit_files = [f for f in os.listdir(cache_dir) if f.endswith('.fits')]
        transit_files = [os.path.join(cache_dir, f) for f in transit_files]
        
        if not transit_files:
            logger.error("No verified transit files available and couldn't fetch from database")
            return []
            
        return transit_files[:num_transits]


def fetch_verified_non_transit_data(num_non_transits=1000, use_cache=True):
    """
    Fetch verified non-transit light curves from astronomical databases.
    
    Args:
        num_non_transits: Number of verified non-transit light curves to fetch
        use_cache: Whether to use cached data if available
    
    Returns:
        list: Paths to downloaded light curve files
    """
    cache_dir = os.path.join(config.DATA_DIR, "verified_non_transits")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(config.METADATA_DIR, "verified_non_transits.pkl")
    
    # Check if we have cached file paths
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached verified non-transit file paths")
        import pickle
        with open(cache_file, 'rb') as f:
            non_transit_files = pickle.load(f)
            
        # Verify files still exist
        existing_files = [f for f in non_transit_files if os.path.exists(f)]
        if len(existing_files) >= num_non_transits:
            logger.info(f"Found {len(existing_files)} cached verified non-transit files")
            return existing_files[:num_non_transits]
        else:
            logger.info(f"Found only {len(existing_files)} valid files from cache, need {num_non_transits}")
    
    # If we don't have enough cached files, fetch from astronomical databases
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        from astroquery.mast import Observations, Catalogs
        
        logger.info(f"Querying Kepler Objects of Interest for false positives")
        
        # Get KOIs marked as FALSE POSITIVE
        false_positives = NasaExoplanetArchive.query_criteria(
            table="koi",
            select="kepid,kepoi_name,koi_disposition,koi_period,koi_depth",
            where="koi_disposition='FALSE POSITIVE'"
        )
        
        logger.info(f"Found {len(false_positives)} Kepler objects marked as false positives")
        
        # Also get random Kepler stars that have no known planets
        logger.info(f"Querying Kepler stars with no known planets")
        
        # First get all known hosts from exoplanet database
        known_hosts = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="hostname",
            where="1=1"  # Get all hosts
        )
        
        # Convert to a set of lowercase names
        known_host_set = {h.lower() for h in known_hosts['hostname']}
        
        # Query random Kepler stars
        try:
            kepler_stars = Catalogs.query_criteria(catalog="KIC", kp=('10.0..13.0'), rows=num_non_transits*5)
            logger.info(f"Found {len(kepler_stars)} Kepler stars")
        except Exception as e:
            logger.error(f"Error querying Kepler stars: {e}")
            kepler_stars = []
        
        # Initialize list for downloaded files
        non_transit_files = []
        
        # First try to get false positives
        count = 0
        for i, row in enumerate(false_positives):
            if count >= num_non_transits:
                break
                
            try:
                kep_id = row['kepid']
                logger.info(f"Fetching light curve for KIC {kep_id} (false positive, {i+1}/{len(false_positives)})")
                
                # Query MAST for light curves
                obs = Observations.query_criteria(
                    target_name=f"KIC {kep_id}",
                    obs_collection='Kepler',
                    dataproduct_type="timeseries"
                )
                
                if len(obs) > 0:
                    products = Observations.get_product_list(obs[0])
                    lc_products = [p for p in products if 'LIGHTCURVE' in p['dataURI']]
                    
                    if lc_products:
                        # Create a filename for consistency
                        output_file = os.path.join(cache_dir, f"KIC_{kep_id}_false_positive.fits")
                        
                        # Download if not already exists
                        if not os.path.exists(output_file):
                            download_path = Observations.download_file(lc_products[0]['dataURI'])
                            
                            # Move to our designated location
                            os.rename(download_path, output_file)
                            
                        non_transit_files.append(output_file)
                        count += 1
                        logger.info(f"Downloaded light curve for KIC {kep_id} - {count}/{num_non_transits}")
                        
                        # Add a small delay to avoid overwhelming the server
                        time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error fetching light curve for KIC {kep_id}: {e}")
                # Continue to next star
        
        # If we don't have enough false positives, get random Kepler stars
        if len(non_transit_files) < num_non_transits and kepler_stars:
            logger.info(f"Fetching random Kepler stars with no known planets")
            
            for i, row in enumerate(kepler_stars):
                if count >= num_non_transits:
                    break
                    
                try:
                    kep_id = row['ID']
                    
                    # Check if this is a known exoplanet host
                    star_name = f"KIC {kep_id}"
                    if star_name.lower() in known_host_set:
                        continue
                    
                    logger.info(f"Fetching light curve for {star_name} (random star, {i+1}/{len(kepler_stars)})")
                    
                    # Query MAST for light curves
                    obs = Observations.query_criteria(
                        target_name=star_name,
                        obs_collection='Kepler',
                        dataproduct_type="timeseries"
                    )
                    
                    if len(obs) > 0:
                        products = Observations.get_product_list(obs[0])
                        lc_products = [p for p in products if 'LIGHTCURVE' in p['dataURI']]
                        
                        if lc_products:
                            # Create a filename for consistency
                            output_file = os.path.join(cache_dir, f"KIC_{kep_id}_random.fits")
                            
                            # Download if not already exists
                            if not os.path.exists(output_file):
                                download_path = Observations.download_file(lc_products[0]['dataURI'])
                                
                                # Move to our designated location
                                os.rename(download_path, output_file)
                                
                            non_transit_files.append(output_file)
                            count += 1
                            logger.info(f"Downloaded light curve for {star_name} - {count}/{num_non_transits}")
                            
                            # Add a small delay to avoid overwhelming the server
                            time.sleep(1)
                
                except Exception as e:
                    logger.error(f"Error fetching light curve for {star_name}: {e}")
                    # Continue to next star
        
        # Cache the downloaded file paths
        if non_transit_files:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(non_transit_files, f)
            
        logger.info(f"Successfully downloaded {len(non_transit_files)} verified non-transit light curves")
        return non_transit_files[:num_non_transits]
        
    except ImportError:
        logger.error("Required modules not available: astroquery")
        # Fallback to any existing files
        non_transit_files = [f for f in os.listdir(cache_dir) if f.endswith('.fits')]
        non_transit_files = [os.path.join(cache_dir, f) for f in non_transit_files]
        
        if not non_transit_files:
            logger.error("No verified non-transit files available and couldn't fetch from database")
            return []
            
        return non_transit_files[:num_non_transits]
    except Exception as e:
        logger.error(f"Error fetching verified non-transit data: {e}")
        # Fallback to any existing files
        non_transit_files = [f for f in os.listdir(cache_dir) if f.endswith('.fits')]
        non_transit_files = [os.path.join(cache_dir, f) for f in non_transit_files]
        
        if not non_transit_files:
            logger.error("No verified non-transit files available and couldn't fetch from database")
            return []
            
        return non_transit_files[:num_non_transits]


def fetch_verified_training_data(num_transits=1000, num_non_transits=1000, split_ratio=0.8, use_cache=True):
    """
    Fetch verified exoplanet transit and non-transit light curves for training/validation.
    
    Args:
        num_transits: Number of verified transit light curves to fetch
        num_non_transits: Number of verified non-transit light curves to fetch
        split_ratio: Train/validation split ratio (e.g., 0.8 = 80% training, 20% validation)
        use_cache: Whether to use cached data if available
    
    Returns:
        dict: Dictionary containing training and validation datasets
    """
    logger.info(f"Fetching verified training data: {num_transits} transits, {num_non_transits} non-transits")
    
    # Fetch transit and non-transit files
    transit_files = fetch_verified_transit_data(num_transits, use_cache)
    non_transit_files = fetch_verified_non_transit_data(num_non_transits, use_cache)
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    
    train_transit, val_transit = train_test_split(
        transit_files, train_size=split_ratio, random_state=42
    ) if transit_files else ([], [])
    
    train_non_transit, val_non_transit = train_test_split(
        non_transit_files, train_size=split_ratio, random_state=42
    ) if non_transit_files else ([], [])
    
    # Create dataset structure
    dataset = {
        'train': {
            'transit': train_transit,
            'non_transit': train_non_transit
        },
        'val': {
            'transit': val_transit,
            'non_transit': val_non_transit
        }
    }
    
    logger.info(f"Generated verified datasets with {len(train_transit)} training transits, " 
               f"{len(val_transit)} validation transits, {len(train_non_transit)} training non-transits, "
               f"and {len(val_non_transit)} validation non-transits")
    
    return dataset


def load_verified_catalog(use_cache=True):
    """
    Load a catalog of verified transit/non-transit light curves.
    
    Args:
        use_cache: Whether to use cached catalog
    
    Returns:
        dict: Dictionary with transit and non-transit file lists
    """
    cache_file = os.path.join(config.METADATA_DIR, "verified_catalog.csv")
    
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading verified catalog from cache")
        try:
            catalog_df = pd.read_csv(cache_file)
            
            # Convert to dictionary
            transit_files = catalog_df[catalog_df['has_transit'] == True]['file_path'].tolist()
            non_transit_files = catalog_df[catalog_df['has_transit'] == False]['file_path'].tolist()
            
            # Check that files exist
            transit_files = [f for f in transit_files if os.path.exists(f)]
            non_transit_files = [f for f in non_transit_files if os.path.exists(f)]
            
            if transit_files and non_transit_files:
                return {
                    'transit': transit_files,
                    'non_transit': non_transit_files
                }
            else:
                logger.warning("Some cached files not found on disk, fetching fresh data")
        except Exception as e:
            logger.error(f"Error loading cached catalog: {e}")
    
    # Fetch verified data
    dataset = fetch_verified_training_data(
        num_transits=1000, 
        num_non_transits=1000,
        use_cache=use_cache
    )
    
    # Combine training and validation sets
    transit_files = dataset['train']['transit'] + dataset['val']['transit']
    non_transit_files = dataset['train']['non_transit'] + dataset['val']['non_transit']
    
    # Create and save catalog
    catalog_data = []
    for file_path in transit_files:
        catalog_data.append({
            'file_path': file_path,
            'has_transit': True,
            'source': 'real_data'
        })
    
    for file_path in non_transit_files:
        catalog_data.append({
            'file_path': file_path,
            'has_transit': False,
            'source': 'real_data'
        })
    
    catalog_df = pd.DataFrame(catalog_data)
    
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        catalog_df.to_csv(cache_file, index=False)
    except Exception as e:
        logger.error(f"Error saving verified catalog: {e}")
    
    return {
        'transit': transit_files,
        'non_transit': non_transit_files
    }


def prepare_verified_datasets():
    """
    Prepare image and time series datasets from verified light curves.
    
    Returns:
        dict: Dictionary containing preprocessed datasets
    """
    from data.light_curve_processor import extract_transit_features, create_image_representations
    
    # Get verified light curve files
    catalog = load_verified_catalog()
    transit_files = catalog['transit']
    non_transit_files = catalog['non_transit']
    
    # Process transit files
    transit_images = []
    transit_timeseries = []
    
    for file_path in transit_files:
        try:
            # Preprocess light curve
            time, flux, _ = preprocess_light_curve(file_path)
            
            if time is None or flux is None:
                continue
            
            # For real transit data, we'll use transit detection
            from pipeline.pipeline_runner import detect_transits
            transit_info = detect_transits(time, flux)
            
            # If no transit detected, try a more sensitive detection
            if transit_info is None:
                transit_info = detect_transits(time, flux, sensitivity=1.5)
            
            # If still nothing, create synthetic info for the middle of the curve
            if transit_info is None:
                mid_idx = len(time) // 2
                transit_info = {
                    'peak_indices': [mid_idx],
                    'times': [time[mid_idx]],
                    'depths': [flux[mid_idx]],
                    'widths': [20],  # Reasonable default width
                    'prominences': [0.1]
                }
            
            # Extract transit features
            segments = extract_transit_features(time, flux, transit_info)
            if segments is None:
                continue
            
            # Create image representations
            images = create_image_representations(segments)
            if images is None:
                continue
            
            # Add to datasets
            for img, seg in zip(images, segments):
                transit_images.append(img)
                transit_timeseries.append(seg)
        
        except Exception as e:
            logger.error(f"Error processing transit file {file_path}: {e}")
    
    # Process non-transit files
    non_transit_images = []
    non_transit_timeseries = []
    
    for file_path in non_transit_files:
        try:
            # Preprocess light curve
            time, flux, _ = preprocess_light_curve(file_path)
            
            if time is None or flux is None:
                continue
            
            # For non-transits, extract random segments
            window_size = config.WINDOW_SIZE
            max_start = len(flux) - window_size
            
            if max_start <= 0:
                continue
            
            # Extract 3 random segments from each non-transit light curve
            for _ in range(3):
                start_idx = np.random.randint(0, max_start)
                segment = flux[start_idx:start_idx + window_size]
                
                # Create image representation from single segment
                image = create_image_representations(np.array([segment]))
                
                if image is not None:
                    non_transit_images.append(image[0])
                    non_transit_timeseries.append(segment)
        
        except Exception as e:
            logger.error(f"Error processing non-transit file {file_path}: {e}")
    
    # Convert to arrays
    transit_images = np.array(transit_images) if transit_images else np.array([])
    transit_timeseries = np.array(transit_timeseries) if transit_timeseries else np.array([])
    non_transit_images = np.array(non_transit_images) if non_transit_images else np.array([])
    non_transit_timeseries = np.array(non_transit_timeseries) if non_transit_timeseries else np.array([])
    
    # Create labels
    transit_labels = np.ones(len(transit_images))
    non_transit_labels = np.zeros(len(non_transit_images))
    
    # Combine datasets
    X_image = np.vstack([transit_images, non_transit_images]) if len(transit_images) > 0 and len(non_transit_images) > 0 else None
    X_timeseries = np.vstack([transit_timeseries, non_transit_timeseries]) if len(transit_timeseries) > 0 and len(non_transit_timeseries) > 0 else None
    y = np.concatenate([transit_labels, non_transit_labels]) if len(transit_labels) > 0 and len(non_transit_labels) > 0 else None
    
    logger.info(f"Prepared verified datasets with {len(transit_labels)} transit examples and {len(non_transit_labels)} non-transit examples")
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    
    if X_image is not None and X_timeseries is not None and y is not None:
        X_image_train, X_image_val, X_ts_train, X_ts_val, y_train, y_val = train_test_split(
            X_image, X_timeseries, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return {
            'train': {
                'X_image': X_image_train,
                'X_timeseries': X_ts_train,
                'y': y_train
            },
            'val': {
                'X_image': X_image_val,
                'X_timeseries': X_ts_val,
                'y': y_val
            },
            'files': {
                'transit': transit_files,
                'non_transit': non_transit_files
            }
        }
    else:
        return None


def integrate_verified_data(X_image, X_timeseries, y, verified_ratio=0.5):
    """
    Integrate verified transit/non-transit data with existing datasets.
    
    Args:
        X_image: Existing image features
        X_timeseries: Existing time series features 
        y: Existing labels
        verified_ratio: Ratio of verified to original data
    
    Returns:
        tuple: (X_image_enhanced, X_timeseries_enhanced, y_enhanced)
    """
    # Get verified datasets
    verified_datasets = prepare_verified_datasets()
    
    if verified_datasets is None:
        logger.warning("No verified datasets available for integration")
        return X_image, X_timeseries, y
    
    # Calculate number of verified samples to include
    original_count = len(y)
    verified_count = int(original_count * verified_ratio)
    
    # Get verified training samples
    v_X_image = verified_datasets['train']['X_image']
    v_X_timeseries = verified_datasets['train']['X_timeseries']
    v_y = verified_datasets['train']['y']
    
    # Limit to calculated count
    if len(v_y) > verified_count:
        from sklearn.model_selection import train_test_split
        _, v_X_image, _, v_X_timeseries, _, v_y = train_test_split(
            v_X_image, v_X_timeseries, v_y, 
            train_size=verified_count,
            stratify=v_y,
            random_state=42
        )
    
    # Combine datasets
    X_image_enhanced = np.vstack([X_image, v_X_image])
    X_timeseries_enhanced = np.vstack([X_timeseries, v_X_timeseries])
    y_enhanced = np.concatenate([y, v_y])
    
    logger.info(f"Enhanced dataset size: {len(y_enhanced)} examples ({np.sum(y_enhanced == 1)} positive, {np.sum(y_enhanced == 0)} negative)")
    
    return X_image_enhanced, X_timeseries_enhanced, y_enhanced
