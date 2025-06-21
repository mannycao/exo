# FILE: pipeline/pipeline_core.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config
from data.light_curve_processor import preprocess_light_curve, detect_transits, extract_transit_features, create_image_representations
from detection.transit_detector import apply_transit_modeling

logger = logging.getLogger(__name__)

def setup_logging(log_level_to_set=logging.INFO, log_file_path_override=None):
    """Configures the root logger for the application."""
    log_file = log_file_path_override or "pipeline.log"
    logging.basicConfig(
        level=log_level_to_set,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging configured from pipeline_core. Level: {logging.getLevelName(log_level_to_set)}. Log file: {log_file}")

def process_light_curve(file_path, output_dir, file_type):
    """
    Processes a single light curve file.
    """
    logger.debug(f"Starting process_light_curve for {file_path} (type: {file_type})")
    try:
        time, flux, metadata = preprocess_light_curve(file_path)
        if time is None or flux is None:
            logger.warning(f"Preprocessing returned invalid time or flux for {file_path}. Returning failure.")
            return {'file_path': file_path, 'success': False, 'reason': 'Preprocessing failed'}

        transit_info = detect_transits(time, flux)
        if not transit_info.get('found_transits'):
            return {'file_path': file_path, 'success': True, 'transit_count': 0}

        transit_segments = extract_transit_features(time, flux, transit_info['transit_indices'])
        image_representations = create_image_representations(transit_segments)

        # Apply physical modeling
        model_info = apply_transit_modeling(time, flux, transit_info)
        transit_info.update(model_info)

        return {
            'file_path': file_path,
            'success': True,
            'transit_count': transit_info.get('transit_count'),
            'period_days': transit_info.get('period_days'),
            'transit_segments': transit_segments,
            'image_features': image_representations,
            'file_type_source': file_type,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Exception during process_light_curve for {file_path}", exc_info=True)
        return {'file_path': file_path, 'success': False, 'reason': str(e)}

def prepare_datasets(processed_results, exoplanet_labels):
    """Prepares datasets for machine learning from processed light curve results."""
    all_image_features, all_timeseries, all_labels = [], [], []

    for result in processed_results:
        if not result.get('success') or not result.get('image_features'):
            continue
        
        file_type = result['file_type_source']
        is_planet = 1 if 'planet' in file_type else 0
        
        for i, image in enumerate(result['image_features']):
            all_image_features.append(image)
            all_timeseries.append(result['transit_segments'][i])
            all_labels.append(is_planet)

    if not all_image_features:
        logger.warning("No valid features collected. Cannot create dataset.")
        return None, None, None

    return np.array(all_image_features), np.array(all_timeseries), np.array(all_labels)

def prepare_train_test_split(features_list, labels, test_size=0.2, val_size=0.2, stratify_labels=None):
    """Splits data into training, validation, and test sets."""
    if not features_list or labels is None or len(labels) == 0:
        return None, None, None, None, None, None

    # First split: separate out the test set
    interim_indices = np.arange(len(labels))
    train_val_indices, test_indices = train_test_split(
        interim_indices, test_size=test_size, random_state=42,
        stratify=stratify_labels
    )

    X_test_list = [feat[test_indices] for feat in features_list]
    y_test = labels[test_indices]

    # Second split: separate training and validation sets
    if stratify_labels is not None:
        stratify_train_val = stratify_labels[train_val_indices]
    else:
        stratify_train_val = None

    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size / (1.0 - test_size), random_state=42,
        stratify=stratify_train_val
    )
    
    X_train_list = [feat[train_indices] for feat in features_list]
    y_train = labels[train_indices]
    X_val_list = [feat[val_indices] for feat in features_list]
    y_val = labels[val_indices]

    return X_train_list, X_val_list, X_test_list, y_train, y_val, y_test
