import os
import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import tensorflow as tf
import glob
import importlib.util

# Ensure the project root is in the sys.path for importing config and other modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

sys.path.append(str(project_root / "lite"))

import config
from paper2_experiments import config as paper2_config

# Dynamic import for KerasTransformerEncoder
CACL_KERAS_MODULE_PATH = project_root / "convert_cacl_to_keras.py"

if not CACL_KERAS_MODULE_PATH.exists():
    logger.error(f"Required module not found: {CACL_KERAS_MODULE_PATH}. Exiting.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("convert_cacl_to_keras", CACL_KERAS_MODULE_PATH)
cacl_keras_module = importlib.util.module_from_spec(spec)
sys.modules["convert_cacl_to_keras"] = cacl_keras_module
spec.loader.exec_module(cacl_keras_module)
KerasTransformerEncoder = cacl_keras_module.KerasTransformerEncoder

from data.dataset_generator import create_dataset

LOG_FILE = project_root / "paper2_experiments" / "long_period_gap_fill.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(config.LOG_LEVEL)
logger.addHandler(stream_handler)

OUTPUT_CSV_PATH = project_root / "paper2_experiments" / "survey_results_long_period.csv"
CHECKPOINT_INTERVAL = 100

def main():
    logger.info("Starting Long-Period Gap Fill Survey for Kepler targets.")
    
    survey_results = []

    # 1. Load Metadata
    FULL_METADATA_CSV = project_root / "data_files" / "full_metadata_v2.csv"

    if FULL_METADATA_CSV.exists():
        combined_metadata_df = pd.read_csv(FULL_METADATA_CSV)
        logger.info(f"Loaded combined metadata from: {FULL_METADATA_CSV}")
    else:
        logger.error(f"Combined metadata CSV not found at {FULL_METADATA_CSV}. Aborting.")
        sys.exit(1)

    # Filter for Long-Period Kepler Targets
    long_period_kepler_df = combined_metadata_df[
        (combined_metadata_df['period'] > 100) & 
        (combined_metadata_df['source'] == 'KEPLER')
    ].copy()
    long_period_kepler_df.set_index('target_id', inplace=True)

    logger.info(f"Found {len(long_period_kepler_df)} Long-Period Kepler Targets to process.")

    if long_period_kepler_df.empty:
        logger.warning("No Long-Period Kepler Targets found after filtering. Exiting.")
        sys.exit(0)

    # 2. Load the Model (same logic as run_kepler_survey.py)
    model_path_option1 = project_root / "results" / "run_20260125-144401" / "exo_multimodal_model_best.h5"
    model_path_option2 = project_root / "models" / "exo_multimodal_model_best.h5"

    model_file_to_load = None
    if model_path_option1.exists():
        model_file_to_load = model_path_option1
    elif model_path_option2.exists():
        model_file_to_load = model_path_option2
    else:
        logger.error(f"Model file not found at either {model_path_option1} or {model_path_option2}. Exiting.")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_file_to_load, custom_objects={'KerasTransformerEncoder': KerasTransformerEncoder}, compile=False)
        logger.info(f"Model loaded from {model_file_to_load}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_file_to_load}: {e}. Exiting.")
        sys.exit(1)

    # 3. Prioritized Loop: Iterate only through targets for which FITS files are found
    # Define multiple search paths for Kepler files
    search_paths = [
        Path("/Users/emmanuel/proj/exo/kepler_local_data/"),
        project_root / "data_files" / "light_curves"
    ]
    
    # Create a map of available Kepler IDs to their full file paths
    available_id_to_path_map = {}
    for path in search_paths:
        for file_type_glob in ['kplr*_lc.fits', 'kplr*_llc.fits', 'kplr*_slc.fits']:
            for file_path in path.rglob(file_type_glob):
                match = re.search(r'kplr(0*)([0-9]+)', file_path.name)
                if match:
                    kepler_id = int(match.group(2))
                    available_id_to_path_map[kepler_id] = str(file_path)
    
    # Filter long_period_kepler_df to only include those IDs found in FITS files
    long_period_kepler_df_filtered = long_period_kepler_df[
        long_period_kepler_df.index.isin(available_id_to_path_map.keys())
    ]
    
    logger.info(f"Found {len(long_period_kepler_df_filtered)} Long-Period Kepler Targets with available FITS files to process.")

    if long_period_kepler_df_filtered.empty:
        logger.warning("No Long-Period Kepler Targets with available FITS files found after filtering. Exiting.")
        sys.exit(0)

    for i, target_id in enumerate(tqdm(long_period_kepler_df_filtered.index, desc="Processing Long-Period Kepler Targets")):
        file_path_str = available_id_to_path_map.get(target_id)
        
        if not file_path_str:
            # This should truly not happen now due to pre-filtering
            logger.error(f"FATAL: File path not found in map for Kepler ID: {target_id}, but it was expected to exist. Skipping.")
            continue

        source = 'KEPLER' # Explicitly Kepler for this loop

        try:
            # Retrieve true_label and period from metadata (already filtered dataframe)
            metadata_entry = long_period_kepler_df.loc[target_id]
            
            # Handle cases where .loc[target_id] might return a Series (for unique index)
            # or a DataFrame (if target_id is not unique in the original metadata_df, though it should be here)
            if isinstance(metadata_entry, pd.Series):
                true_label = str(metadata_entry.get('true_label', 'UNKNOWN'))
                period_from_metadata = metadata_entry.get('period', np.nan)
            elif isinstance(metadata_entry, pd.DataFrame):
                # If it's a DataFrame, it means there are duplicate target_ids in the original metadata
                # For this context, we'll take the first matching entry.
                logger.warning(f"Duplicate target_id '{target_id}' found in metadata. Using the first entry for processing.")
                true_label = str(metadata_entry.iloc[0].get('true_label', 'UNKNOWN'))
                period_from_metadata = metadata_entry.iloc[0].get('period', np.nan)
            else:
                true_label = 'UNKNOWN'
                period_from_metadata = np.nan
                logger.error(f"Unexpected type for metadata_entry for target_id {target_id}. Type: {type(metadata_entry)}")


            # Atomic Inference (same logic as run_kepler_survey.py)
            create_dataset_input = [{'file_path': file_path_str, 'type': true_label}] 
            X_ts, X_img, X_features, _, all_pipeline_results, _ = create_dataset(
                file_paths=create_dataset_input,
                output_dir=None,
                image_size=config.IMAGE_SIZE
            )

            period_used_for_folding = all_pipeline_results[0].get('periodicity', np.nan)
            output_period = period_from_metadata if pd.notna(period_from_metadata) else period_used_for_folding

            if (X_ts is None or X_img is None or X_features is None or
                    len(X_ts) == 0 or np.all(X_img == 0)):
                logger.warning(f"Skipping {target_id} ({file_path_str}): create_dataset returned empty or all-zero data views. Check X_ts, X_img, or X_features.")
                continue

            x_1d = np.squeeze(X_ts)
            x_1d = np.expand_dims(x_1d, axis=0)

            x_2d = np.squeeze(X_img)
            if x_2d.ndim == 2:
                x_2d = np.expand_dims(x_2d, axis=(0, -1))
            elif x_2d.ndim == 3:
                x_2d = np.expand_dims(x_2d, axis=0)
            
            x_features = np.squeeze(X_features)
            x_features = np.expand_dims(x_features, axis=0) if x_features.ndim == 1 else x_features
            
            zeros_1d = np.zeros_like(x_1d)
            zeros_2d = np.zeros_like(x_2d)
            zeros_features = np.zeros_like(x_features)
            
            p_joint = model.predict([x_2d, x_1d, x_features], verbose=0)[0][0]
            p_1d_raw = model.predict([zeros_2d, x_1d, zeros_features], verbose=0)[0][0]
            p_2d_raw = model.predict([x_2d, zeros_1d, zeros_features], verbose=0)[0][0]

            disagreement = abs(p_1d_raw - p_2d_raw)

            prob_1d = float(np.clip(p_joint + disagreement / 2, 0.0, 1.0))
            prob_2d = float(np.clip(p_joint - disagreement / 2, 0.0, 1.0))

            survey_results.append({
                'target_id': target_id,
                'period': output_period,
                'true_label': true_label,
                'prob_joint': p_joint,
                'prob_1d': prob_1d,
                'prob_2d': prob_2d,
                'disagreement': disagreement
            })

        except Exception as e:
            logger.error(f"Failed to process {file_path_str} (ID: {target_id}): {e}")
            pass

        if (i + 1) % CHECKPOINT_INTERVAL == 0 and len(survey_results) > 0:
            pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({len(survey_results)} items processed).")

    if survey_results:
        final_df = pd.DataFrame(survey_results)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Final survey results saved to: {OUTPUT_CSV_PATH}")
    else:
        logger.warning("No survey results to save.")

    logger.info(f"Long-Period Gap Fill Survey complete: Processed {len(long_period_kepler_df)} targets (actual results: {len(survey_results)}).")
    
    logging.shutdown()

if __name__ == "__main__":
    main()
