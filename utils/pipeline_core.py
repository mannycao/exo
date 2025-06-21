# FILE: pipeline/pipeline_core.py (New, Corrected Version)

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config
from data.light_curve_processor import preprocess_light_curve, extract_transit_features, create_image_representations
from detection.transit_detector import detect_transits, apply_transit_modeling

logger = logging.getLogger(__name__)

def setup_logging(log_level_to_set=logging.INFO, log_file_path_override=None):
    """Configures the root logger for the application."""
    log_file = log_file_path_override or "pipeline.log"
    logging.basicConfig(
        level=log_level_to_set, format=config.LOG_FORMAT,
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
        force=True
    )

def process_light_curve(file_path, output_dir, file_type):
    """Processes a single light curve file from start to finish."""
    logger.debug(f"Processing file: {Path(file_path).name} (Type: {file_type})")
    try:
        time, flux, metadata = preprocess_light_curve(file_path)
        if time is None:
            logger.warning(f"Preprocessing failed for {file_path}.")
            return {'file_path': file_path, 'success': False}

        transit_info = detect_transits(time, flux)
        
        if not transit_info.get('found_transits'):
            # This is a successful processing run, but with no interesting events.
            return {'file_path': file_path, 'success': True, 'transit_count': 0}

        transit_segments = extract_transit_features(time, flux, transit_info['transit_indices'])
        image_reps = create_image_representations(transit_segments)
        model_info = apply_transit_modeling(time, flux, transit_info)
        transit_info.update(model_info)

        return {
            'file_path': file_path, 'success': True,
            'transit_count': transit_info.get('transit_count'),
            'transit_segments': transit_segments, 'image_features': image_reps,
            'file_type_source': file_type, **transit_info
        }
    except Exception as e:
        logger.error(f"Unhandled exception in process_light_curve for {file_path}", exc_info=True)
        return {'file_path': file_path, 'success': False, 'reason': str(e)}

def prepare_datasets(processed_results, exoplanet_labels):
    """Prepares datasets for machine learning."""
    features, segments, labels = [], [], []
    for res in processed_results:
        if res.get('success') and res.get('image_features') is not None:
            is_planet = 1 if 'planet' in res['file_type_source'] else 0
            num_features = len(res['image_features'])
            features.extend(res['image_features'])
            segments.extend(res['transit_segments'])
            labels.extend([is_planet] * num_features)
    
    if not features:
        logger.warning("No valid features were extracted from any files.")
        return None, None, None

    return np.array(features), np.array(segments), np.array(labels)

def prepare_train_test_split(features_list, labels, **kwargs):
    """Splits data into training, validation, and test sets."""
    return train_test_split(*features_list, labels, test_size=0.2, random_state=42, stratify=labels)

