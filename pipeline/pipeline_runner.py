"""
Main pipeline execution for exoplanet detection.
Includes core light curve processing and dataset preparation.
This version has enhanced error handling and logging within process_light_curve
to capture the full traceback of any exception.
"""

import os
import logging
import sys 
import numpy as np
import pandas as pd 
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor 
from pathlib import Path 
from collections import Counter
import traceback # Import the traceback module

import config 
from data.data_fetcher import download_light_curves, fetch_exoplanet_labels 
from data.light_curve_processor import (
    preprocess_light_curve as lc_preprocess, 
    detect_transits as lc_detect_transits, 
    extract_transit_features as lc_extract_features,
    create_image_representations as lc_create_images,
    reshape_data_for_cnn 
)
from detection.periodicity_analyzer import analyze_periodicity, calculate_folded_lightcurve, bin_folded_lightcurve
from detection.transit_detector import apply_transit_modeling, estimate_planet_properties

from utils.visualization import (
    visualize_transit, visualize_folded_transit, visualize_periodogram,
    visualize_transit_model, visualize_detection_results
)

logger = logging.getLogger(__name__) 

def setup_logging(log_level_to_set=None, log_file_path_override=None): 
    log_level = log_level_to_set if log_level_to_set is not None else getattr(config, 'LOG_LEVEL', logging.INFO)
    
    effective_log_file_path = None
    if log_file_path_override:
        effective_log_file_path = Path(log_file_path_override)
    elif hasattr(config, 'LOG_FILE'):
        effective_log_file_path = Path(config.LOG_FILE)
    else: 
        effective_log_file_path = Path.cwd() / "results" / "pipeline_default.log"
        if not logging.getLogger().hasHandlers(): 
            logging.basicConfig(level=logging.WARNING)
        logging.warning(f"No log file path specified, defaulting to {effective_log_file_path}")

    effective_log_file_path.parent.mkdir(parents=True, exist_ok=True) 

    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT if hasattr(config, 'LOG_FORMAT') else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(effective_log_file_path, mode='w'), 
            logging.StreamHandler(sys.stdout) 
        ],
        force=True 
    )
    
    logging.getLogger("matplotlib").setLevel(logging.WARNING) 
    logging.getLogger("astropy").setLevel(logging.WARNING) 

    logger.info(f"Logging configured from pipeline_runner. Level: {logging.getLevelName(log_level)}. Log file: {effective_log_file_path}")


def time_array_valid(time_array):
    return time_array is not None and isinstance(time_array, np.ndarray) and time_array.ndim == 1 and len(time_array) > 0

def flux_array_valid(flux_array):
    return flux_array is not None and isinstance(flux_array, np.ndarray) and flux_array.ndim == 1 and len(flux_array) > 0


def process_light_curve(file_path, result_dir_str=None, file_type=config.FILE_TYPE_UNKNOWN):
    """
    Process a single light curve file.
    Includes file_type for context in subsequent processing.
    Enhanced with detailed DEBUG logging and robust exception handling.
    """
    if result_dir_str:
        result_dir = Path(result_dir_str)
    else:
        default_results_base = Path(config.RESULTS_DIR if hasattr(config, 'RESULTS_DIR') else "./results") / "processed_file_artifacts_direct_call"
        result_dir = default_results_base / Path(file_path).stem
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    processing_metadata = {'file_path': file_path, 'file_type_source': file_type}
    logger.debug(f"Starting process_light_curve for {file_path} (type: {file_type})")

    try:
        # STEP 1: Preprocessing
        time, flux, metadata = lc_preprocess(file_path) 
        processing_metadata.update(metadata if metadata else {})
        if not time_array_valid(time) or not flux_array_valid(flux):
            logger.warning(f"Preprocessing returned invalid time or flux for {file_path}. Returning failure.")
            return {**processing_metadata, 'success': False, 'error': "Preprocessing failed", 'result_dir': str(result_dir), 'has_transit_images': False}
        
        # STEP 2: Transit Detection
        transit_info = lc_detect_transits(time, flux, sensitivity=config.TRANSIT_SENSITIVITY, 
                                          min_duration_cadences=3, max_duration_cadences=30) # Using defaults for simplicity, can be configured
        processing_metadata['transit_info_raw'] = transit_info
        if transit_info is None or not transit_info.get('peak_indices'):
            logger.info(f"No transits detected in {file_path}.")
            return {**processing_metadata, 'transit_count': 0, 'success': True, 'result_dir': str(result_dir), 'has_transit_images': False }

        # STEP 3: Periodicity and Modeling
        periodicity_data = analyze_periodicity(time, flux, transit_info)
        processing_metadata['periodicity_data'] = periodicity_data

        if periodicity_data and periodicity_data.get('median_period'):
            processing_metadata['transit_model_data'] = apply_transit_modeling(time, flux, transit_info, periodicity_data)
            processing_metadata['planet_properties_estimated'] = estimate_planet_properties(transit_info, periodicity_data, stellar_properties=metadata)
        
        # STEP 4: Feature Extraction
        transit_segments_data = lc_extract_features(time, flux, transit_info)
        transit_images_data = lc_create_images(transit_segments_data) if transit_segments_data else []
        
        current_has_transit_images = transit_images_data is not None and len(transit_images_data) > 0

        # STEP 5: Return successful result
        return {
            **processing_metadata,
            'transit_count': len(transit_info['peak_indices']),
            'periodicity': periodicity_data.get('median_period') if periodicity_data else None,
            'has_transit_images': current_has_transit_images,
            'transit_images_features': transit_images_data, 
            'transit_segments_features': transit_segments_data, 
            'result_dir': str(result_dir),
            'success': True
        }

    except Exception as e:
        # VVVVVVVVVVVVVVVVVVVVVV ENHANCED ERROR CATCHING VVVVVVVVVVVVVVVVVVVVVV
        error_type_name = type(e).__name__
        error_message = str(e)
        full_traceback_str = traceback.format_exc()
        
        # Log the full traceback for debugging
        logger.error(
            f"Exception during process_light_curve for {file_path} (type: {file_type}):\n"
            f"{full_traceback_str}"
        )
        
        # Return a more informative error message that will be logged by the main process
        return {
            **processing_metadata, 
            'success': False, 
            'error': f"{error_type_name}: {error_message}", # Example: "KeyError: 'widths'"
            'traceback': full_traceback_str, # Also return the full traceback
            'result_dir': str(result_dir), 
            'has_transit_images': False
        }
        # ^^^^^^^^^^^^^^^^^^^^ ENHANCED ERROR CATCHING ^^^^^^^^^^^^^^^^^^^^

# ... (the rest of the file: prepare_datasets, prepare_train_test_split) remains the same as Canvas "pipeline_runner_py_log_fix"
def prepare_datasets(results_with_file_types, exoplanet_labels_df=None):
    logger.info(f"Preparing datasets from {len(results_with_file_types)} processed results.")
    all_transit_images, all_transit_segments, all_labels = [], [], []
    features_per_source_type_counter = Counter()

    for result_item in results_with_file_types:
        if not result_item.get('success', False):
            logger.debug(f"Skipping result for {result_item.get('file_path', 'N/A')} because 'success' was False. Error: {result_item.get('error')}")
            # Log the full traceback if it was captured
            if result_item.get('traceback'):
                logger.debug(f"  Full Traceback for failed item:\n{result_item.get('traceback')}")
            continue
        if not result_item.get('has_transit_images', False):
            logger.debug(f"Skipping result for {result_item.get('file_path', 'N/A')} because 'has_transit_images' was False.")
            continue

        file_path, file_type = result_item['file_path'], result_item.get('file_type_source', config.FILE_TYPE_UNKNOWN)
        images, segments = result_item.get('transit_images_features'), result_item.get('transit_segments_features')

        if not images or not segments or len(images) != len(segments):
            logger.warning(f"Feature mismatch or empty features for {file_path}. Skipping.")
            continue

        features_per_source_type_counter[file_type] += len(images)
        for img, seg in zip(images, segments):
            label = 1 if file_type in [config.FILE_TYPE_CONFIRMED_PLANET, config.FILE_TYPE_SYNTHETIC_PLANET] else 0
            all_transit_images.append(img)
            all_transit_segments.append(seg)
            all_labels.append(label)

    logger.info(f"DEBUG: Total features extracted and labeled: {len(all_labels)}")
    if all_labels:
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        logger.info(f"DEBUG: Label distribution (before balancing): Labels={unique_labels}, Counts={counts}")
    logger.info(f"DEBUG: Features per source type (before balancing): {dict(features_per_source_type_counter)}")

    if not all_transit_images:
        logger.warning("No valid features collected. Cannot create dataset.")
        return None, None, None

    return np.array(all_transit_images), np.array(all_transit_segments), np.array(all_labels)

def prepare_train_test_split(X_data, y_data, test_size=0.2, val_size_from_train=0.25, random_state=42, stratify_labels=None):
    """
    Prepares train, validation, and test splits of the data.
    Robustly handles single-modal (np.ndarray) and multi-modal (list of np.ndarray) data
    by splitting indices first.
    """
    from sklearn.model_selection import train_test_split as sklearn_tts

    if y_data is None or len(y_data) < 2:
        logger.error(f"Cannot split data: y_data is None or has fewer than 2 samples ({len(y_data) if y_data is not None else 'None'}).")
        # Return empty structures that match expected output format
        if isinstance(X_data, list):
            return ([np.array([]) for _ in X_data] if X_data else [np.array([])]), ([np.array([]) for _ in X_data] if X_data else [np.array([])]), ([np.array([]) for _ in X_data] if X_data else [np.array([])]), np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
    stratify_array = y_data if stratify_labels is None else stratify_labels
    can_stratify = len(np.unique(stratify_array)) > 1 if stratify_array is not None else False
    
    # Split indices instead of data directly
    indices = np.arange(len(y_data))
    
    # First split: separate out the test set
    try:
        train_val_idx, test_idx = sklearn_tts(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_array if can_stratify else None
        )
    except ValueError as e:
        logger.warning(f"Stratified train-test split failed: {e}. Splitting without stratification.")
        train_val_idx, test_idx = sklearn_tts(indices, test_size=test_size, random_state=random_state)

    # Second split: separate train and validation from the train_val set
    y_train_val_for_stratify = y_data[train_val_idx]
    can_stratify_val = len(np.unique(y_train_val_for_stratify)) > 1 if len(y_train_val_for_stratify) > 0 else False
    
    if len(train_val_idx) < 2: # Cannot split train_val further
        train_idx, val_idx = train_val_idx, np.array([], dtype=int)
    else:
        try:
            train_idx, val_idx = sklearn_tts(
                train_val_idx,
                test_size=val_size_from_train,
                random_state=random_state,
                stratify=y_train_val_for_stratify if can_stratify_val else None
            )
        except ValueError as e:
            logger.warning(f"Stratified train-validation split failed: {e}. Splitting without stratification.")
            train_idx, val_idx = sklearn_tts(train_val_idx, test_size=val_size_from_train, random_state=random_state)
            
    # Use indices to slice all datasets
    y_train, y_val, y_test = y_data[train_idx], y_data[val_idx], y_data[test_idx]

    if isinstance(X_data, list):
        # Handle multimodal case
        X_train = [x[train_idx] for x in X_data]
        X_val = [x[val_idx] for x in X_data]
        X_test = [x[test_idx] for x in X_data]
    else:
        # Handle single-modal case
        X_train = X_data[train_idx]
        X_val = X_data[val_idx]
        X_test = X_data[test_idx]
        
    logger.info(f"Data split result: Train ({len(y_train)}), Validation ({len(y_val)}), Test ({len(y_test)})")
    if len(y_train) > 0: logger.info(f"  Train labels distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    if len(y_val) > 0: logger.info(f"  Validation labels distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    if len(y_test) > 0: logger.info(f"  Test labels distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_val, X_test, y_train, y_val, y_test
