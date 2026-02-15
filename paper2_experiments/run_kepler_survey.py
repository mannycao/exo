import os
import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import tensorflow as tf # Explicitly import tensorflow
import glob # Required for file scanning
import importlib.util # For dynamic module loading

# Ensure the project root is in the sys.path for importing config and other modules
project_root = Path(__file__).resolve().parents[1] # Go up one level from paper2_experiments/ to phd/
sys.path.insert(0, str(project_root))

# Add the 'lite' directory to sys.path if convert_cacl_to_keras.py is inside it
# Based on the provided folder structure, convert_cacl_to_keras.py is at the root.
# So, we append the root itself for module resolution of sub-dependencies within 'lite' if any.
sys.path.append(str(project_root / "lite"))


# Import the main config.py for general settings and paper2_experiments.config for specific paths
import config # Main project config
from paper2_experiments import config as paper2_config # Specific config for paper2 experiments

# Dynamic import for KerasTransformerEncoder
# This addresses the ModuleNotFoundError and ensures the class is available for custom_objects.
CACL_KERAS_MODULE_PATH = project_root / "convert_cacl_to_keras.py"

if not CACL_KERAS_MODULE_PATH.exists():
    logger.error(f"Required module not found: {CACL_KERAS_MODULE_PATH}. Exiting.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("convert_cacl_to_keras", CACL_KERAS_MODULE_PATH)
cacl_keras_module = importlib.util.module_from_spec(spec)
sys.modules["convert_cacl_to_keras"] = cacl_keras_module
spec.loader.exec_module(cacl_keras_module)
KerasTransformerEncoder = cacl_keras_module.KerasTransformerEncoder

# Import necessary modules from the project
from data.dataset_generator import create_dataset
# CACL Explainer and LabelEncoder are not explicitly requested for this simplified "Naked Inference" run.
# Ablation logic is explicit for p1 and p2.
LOG_FILE = project_root / "paper2_experiments" / "kepler_survey_ablation.log"
# Ensure the directory for the log file exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure the root logger with a basic level, but no handlers yet.
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Explicitly set logger level to DEBUG
logger.propagate = False # Prevent messages from being passed to the root logger's handlers

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG) # All messages to file
logger.addHandler(file_handler)

# Stream handler (console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(config.LOG_LEVEL) # Only show messages at config.LOG_LEVEL or above to console
logger.addHandler(stream_handler)

# --- Paths ---
# Output CSV path as specified by the user
OUTPUT_CSV_PATH = project_root / "paper2_experiments" / "survey_results_kepler.csv"

# Checkpoint interval as specified by the user
CHECKPOINT_INTERVAL = 100 # Changed to 100 as per request

def main():
    logger.info("Starting Kepler survey (Naked Inference with Ablation) with zero external dependencies in error handler.")
    
    survey_results = []

    # 1. Load Metadata for all relevant sources (Kepler and TESS)
    FULL_METADATA_CSV = project_root / "data_files" / "full_metadata_v2.csv"

    if FULL_METADATA_CSV.exists():
        combined_metadata_df = pd.read_csv(FULL_METADATA_CSV)
        # We assume 'source' column in full_metadata_v2.csv correctly identifies KEPLER vs TESS
        # And that target_id is unique within each source.
        
        logger.info(f"Loaded combined metadata from: {FULL_METADATA_CSV}")
        logger.info(f"Found {len(combined_metadata_df[combined_metadata_df['source'] == 'KEPLER'])} Kepler entries and {len(combined_metadata_df[combined_metadata_df['source'] == 'TESS'])} TESS entries.")
    else:
        logger.error(f"Combined metadata CSV not found at {FULL_METADATA_CSV}. Aborting.")
        sys.exit(1)

    # Index metadata by a combination of 'target_id' and 'source' for fast and unique lookup
    combined_metadata_df.set_index(['target_id', 'source'], inplace=True)
    logger.info("Combined metadata indexed by (target_id, source).")

    # 2. Load the Model
    # Search for the model file
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
        # Load the model with custom_objects to ensure KerasTransformerEncoder is found.
        model = tf.keras.models.load_model(model_file_to_load, custom_objects={'KerasTransformerEncoder': KerasTransformerEncoder}, compile=False)
        logger.info(f"Model loaded from {model_file_to_load}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_file_to_load}: {e}. Exiting.")
        sys.exit(1)

    # 3. Iterate through light curve files (Kepler and TESS)
    # Use glob to find kplr*_llc.fits files in Kepler directories
    kepler_files = []
    kepler_files.extend(glob.glob(paper2_config.CONFIRMED_PLANETS_DIR + "/**/kplr*_llc.fits", recursive=True))
    kepler_files.extend(glob.glob(paper2_config.FALSE_POSITIVES_DIR + "/**/kplr*_llc.fits", recursive=True))

    # Use glob to find tess*_lc.fits files in TESS directories
    tess_files = []
    tess_files.extend(glob.glob(paper2_config.CONFIRMED_PLANETS_DIR + "/**/tess*_lc.fits", recursive=True))
    tess_files.extend(glob.glob(paper2_config.FALSE_POSITIVES_DIR + "/**/tess*_lc.fits", recursive=True))

    all_light_curve_files = []
    all_light_curve_files.extend(kepler_files)
    all_light_curve_files.extend(tess_files)

    initial_total_files = len(all_light_curve_files) # Store the initial length

    if not all_light_curve_files:
        logger.error("No Kepler or TESS light curve files found using glob patterns. Aborting.")
        sys.exit(1)
    
    logger.info(f"Found {initial_total_files} light curve files (Kepler and TESS) for processing.")

    # Loop through each file with robust error handling
    for i, file_path_str in enumerate(tqdm(all_light_curve_files, desc="Running Ablation Survey")):
        target_id = np.nan # Default for logging if extraction fails early
        
        try:
            # Extract target_id and source (Kepler/TESS) from filename
            file_name = Path(file_path_str).name
            
            # Try Kepler naming convention first
            match_kplr = re.search(r'kplr(\d+)-\d+_(s|l)lc\.fits', file_name)
            if match_kplr:
                target_id = int(match_kplr.group(1))
                source = 'KEPLER'
            else:
                # Try TESS naming convention
                match_tess = re.search(r'tess\d{13}-s\d{4}-(\d{8,16})-\d{4}-s_lc\.fits', file_name)
                if match_tess:
                    target_id = int(match_tess.group(1))
                    source = 'TESS'
                else:
                    logger.warning(f"Could not extract target_id from filename: {file_name}. Skipping.")
                    continue


            # Retrieve true_label and period from combined metadata using (target_id, source)
            try:
                metadata_entry = combined_metadata_df.loc[(target_id, source)]
                true_label_raw = metadata_entry.at['true_label']
                period_from_metadata_raw = metadata_entry.at['period']
            except KeyError:
                logger.warning(f"Metadata entry not found for (ID: {target_id}, Source: {source}). Assuming CANDIDATE and NaN period. File: {file_name}")
                true_label_raw = 'CANDIDATE'
                period_from_metadata_raw = np.nan
            
            # Robustly ensure true_label is a scalar string
            if isinstance(true_label_raw, pd.Series):
                true_label = str(true_label_raw.iloc[0]) if not true_label_raw.empty else 'UNKNOWN'
            else:
                true_label = str(true_label_raw)
            
            # Robustly ensure period_from_metadata is a scalar
            if isinstance(period_from_metadata_raw, pd.Series):
                period_from_metadata = period_from_metadata_raw.iloc[0] if not period_from_metadata_raw.empty else np.nan
            else:
                period_from_metadata = period_from_metadata_raw

            # Step 1: create_dataset call (expects list of dicts, not just filename)
            # The 'type' argument is crucial for create_dataset's internal logic.
            create_dataset_input = [{'file_path': file_path_str, 'type': true_label}] 
            X_ts, X_img, X_features, _, all_pipeline_results, _ = create_dataset(
                file_paths=create_dataset_input,
                output_dir=None, # Don't save individual processed data during survey
                image_size=config.IMAGE_SIZE # Pass image_size as it's a required arg
            )

            # Get the period that create_dataset used/detected for phase folding
            period_used_for_folding = all_pipeline_results[0].get('periodicity', np.nan)
            output_period = period_from_metadata if pd.notna(period_from_metadata) else period_used_for_folding

            if (X_ts is None or X_img is None or X_features is None or
                    len(X_ts) == 0 or np.all(X_img == 0)): # Add X_features check for safety
                logger.warning(f"Skipping {target_id} ({file_path_str}): create_dataset returned empty or all-zero data views. Check X_ts, X_img, or X_features.")
                # Atomic error handling: skip without appending partial results
                continue

            # Ensure data shapes are compatible for prediction: (batch_size, ...)
            # For x_1d (time series): target shape (1, 2048)
            x_1d = np.squeeze(X_ts)
            x_1d = np.expand_dims(x_1d, axis=0) # Ensure it's (1, 2048)

            # For x_2d (image): target shape (1, 64, 64, 1)
            x_2d = np.squeeze(X_img)
            if x_2d.ndim == 2: # If it's (64, 64) after squeeze
                x_2d = np.expand_dims(x_2d, axis=(0, -1)) # Add batch and channel -> (1, 64, 64, 1)
            elif x_2d.ndim == 3: # If it's (64, 64, 1) after squeeze (already has channel)
                x_2d = np.expand_dims(x_2d, axis=0) # Add batch -> (1, 64, 64, 1)
            
            # For x_features: target shape (1, features_dim)
            x_features = np.squeeze(X_features)
            x_features = np.expand_dims(x_features, axis=0) if x_features.ndim == 1 else x_features # Ensure it's (1, features_dim)
            
            # Step 2: Inputs for model prediction (already assigned correctly above)

            # Create zeroed-out versions for ablation
            zeros_1d = np.zeros_like(x_1d)
            zeros_2d = np.zeros_like(x_2d)
            zeros_features = np.zeros_like(x_features)
            
            # Step 3 (Ablation):
            # The model expects 3 inputs. Adjust ablation for all three modalities.
            
            # For p_joint: predict with all three inputs (Image, Time Series, Features)
            p_joint = model.predict([x_2d, x_1d, x_features], verbose=0)[0][0]
            
            # For p_1d: predict with x_1d and zeroed x_2d and zeroed x_features
            # (effectively ablating image and features)
            p_1d_raw = model.predict([zeros_2d, x_1d, zeros_features], verbose=0)[0][0]
            
            # For p_2d: predict with zeroed x_1d and x_2d and zeroed x_features
            # (effectively ablating time series and features)
            p_2d_raw = model.predict([x_2d, zeros_1d, zeros_features], verbose=0)[0][0]

            # Step 4: Disagreement
            disagreement = abs(p_1d_raw - p_2d_raw)

            # Derive final p_1d and p_2d (Naked Inference logic)
            prob_1d = float(np.clip(p_joint + disagreement / 2, 0.0, 1.0))
            prob_2d = float(np.clip(p_joint - disagreement / 2, 0.0, 1.0))

            # Step 5: Save to list
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
            # Atomic error handling: Just skip bad files, keep moving.
            # Do NOT reference any other variables or append partial results.
            logger.error(f"Failed to process {file_path_str} (ID: {target_id if not np.isnan(target_id) else 'UNKNOWN'}): {e}")
            pass # Keep moving

        # Output saving every 100 rows
        if (i + 1) % CHECKPOINT_INTERVAL == 0 and len(survey_results) > 0:
            pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({len(survey_results)} items processed).")

    # Final Output
    if survey_results: # Only save if there are results
        final_df = pd.DataFrame(survey_results)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Final survey results saved to: {OUTPUT_CSV_PATH}")
    else:
        logger.warning("No survey results to save.")
    logger.info(f"Ablation survey complete: Processed {initial_total_files} files (actual results: {len(survey_results)}).")
    
    logging.shutdown()

if __name__ == "__main__":
    main()
