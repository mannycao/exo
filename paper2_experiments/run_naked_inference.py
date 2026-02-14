import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from pathlib import Path
import re # Needed for re.search in scan_data_directories

# Ensure the project root is at the beginning of the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary functions from the project
from data.dataset_generator import process_single_file
import config
from models.multimodal_model import build_multimodal_fusion_model

# --- Configuration ---
# Use config.MODEL_PATH from the project's config
MODEL_PATH = config.MODEL_PATH

SURVEY_RESULTS_DIR = config.SURVEY_RESULTS_DIR
os.makedirs(SURVEY_RESULTS_DIR, exist_ok=True)
RESULTS_CSV_PATH = SURVEY_RESULTS_DIR / "survey_results_real.csv"

# Get model input shapes from config
IMAGE_SHAPE = config.IMAGE_SIZE + (1,) # Assuming grayscale image input (height, width, channels)
TIMESERIES_SHAPE = (config.FIXED_LENGTH, 1) # Assuming 1 feature per timestep (length, channels)
FEATURE_SHAPE = (config.FEATURE_VECTOR_LENGTH,)

# --- Helper Functions (Adapted from run_full_scale_survey.py) ---
def scan_data_directories(confirmed_dir, false_positives_dir):
    """
    Scans specified directories for FITS/npz/tbl files and extracts target_ids and labels.
    Returns a list of dictionaries with {'target_id', 'file_path', 'type'}.
    """
    target_data = []

    def process_dir(directory, file_type):
        logger.info(f"Scanning {directory} for {file_type} files...")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.fits', '.npz', '.tbl')):
                    file_path = Path(root) / file
                    # Extract target_id from filename (e.g., kplr010000941-...)
                    # Using regex for more robust extraction of KIC/TIC ID, as filenames can vary
                    # Handles kplr<ID> and tessYYYYMMDDHHHHH-sSSSS-<ID>-...
                    match_kepid = re.search(r'(?:kplr|tess\d{15}-s\d{4}-)(\d+)', file)
                    if match_kepid:
                        target_id = int(match_kepid.group(1))
                    else:
                        logger.warning(f"Could not extract KIC/TIC ID from filename: {file}. Skipping.")
                        continue
                    
                    target_data.append({
                        'target_id': target_id,
                        'file_path': file_path.as_posix(),
                        'type': file_type # 'confirmed_planet' or 'false_positive'
                    })
        logger.info(f"Found {len(target_data)} files in {directory}.")

    # Use the File Type Constants from config.py
    process_dir(confirmed_dir, config.FILE_TYPE_CONFIRMED_PLANET)
    process_dir(false_positives_dir, config.FILE_TYPE_FALSE_POSITIVE)
    
    # Filter for unique target_ids (a target might have multiple files), keeping first encountered
    unique_targets = {}
    for item in target_data:
        if item['target_id'] not in unique_targets:
            unique_targets[item['target_id']] = item
    
    return list(unique_targets.values())


# --- Main Script ---
def run_naked_inference():
    logger.info("Starting naked inference using real Keras model and modality ablation.")

    # 1. Build Model Architecture and Load Weights
    try:
        real_model = build_multimodal_fusion_model(
            image_shape=IMAGE_SHAPE,
            timeseries_shape=TIMESERIES_SHAPE,
            feature_shape=FEATURE_SHAPE
        )
        real_model.load_weights(MODEL_PATH)
        logger.info(f"Successfully built model architecture and loaded weights from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error building model or loading weights from {MODEL_PATH}: {e}", exc_info=True)
        return

    # 2. Scan Data Directories for all target files
    # Use the specific directories where the full dataset is expected to be
    all_targets_info = scan_data_directories(config.CONFIRMED_PLANETS_DIR, config.FALSE_POSITIVES_DIR)
    if not all_targets_info:
        logger.error("No target files found in specified directories. Aborting.")
        return
    logger.info(f"Found {len(all_targets_info)} unique target files to process.")

    results = []
    # Initialize CSV with header if it doesn't exist or is empty
    if not os.path.exists(RESULTS_CSV_PATH) or os.stat(RESULTS_CSV_PATH).st_size == 0:
        pd.DataFrame(columns=['target_id', 'period', 'true_label', 'prob_joint', 'prob_1d', 'prob_2d', 'disagreement']).to_csv(RESULTS_CSV_PATH, index=False)

    # 3. Inference Loop
    for target_info in tqdm(all_targets_info, desc="Performing Inference"):
        target_id = target_info['target_id']
        file_path = target_info['file_path']
        true_label_str = target_info['type']
        
        # Convert true_label to 1 or 0 for consistency
        true_label = 1 if true_label_str == config.FILE_TYPE_CONFIRMED_PLANET else 0

        # file_info_dict is for process_single_file function
        file_info_dict = {'file_path': file_path, 'type': true_label_str}

        try:
            # Process data to get x_ts (1D), x_img (2D), x_features
            # process_single_file returns (segment, img_norm, feature_vector, label, transit_info, periodicity_data, ...)
            processed_data = process_single_file(file_info_dict, config.IMAGE_SIZE, config.FIXED_LENGTH)
            if processed_data is None:
                logger.warning(f"Skipping target {target_id}: Failed to process light curve file. File: {file_path}")
                continue
            
            x_ts_raw, x_img_raw, x_features_raw, _, _, periodicity_data, _, _, _ = processed_data
            
            # Extract period from periodicity_data
            period = periodicity_data.get('median_period', np.nan)

            # Ensure correct dimensions for model input (add batch dim, add channel dim for 1D/2D)
            x_ts = np.expand_dims(x_ts_raw, axis=0) # Shape (1, time_steps)
            x_ts = np.expand_dims(x_ts, axis=-1) # Shape (1, time_steps, 1)

            x_img = np.expand_dims(x_img_raw, axis=0) # Shape (1, height, width)
            x_img = np.expand_dims(x_img, axis=-1) # Shape (1, height, width, 1)

            x_features = np.expand_dims(x_features_raw, axis=0) # Shape (1, num_features)

            # Create zeros for ablation - ensure shapes match original inputs
            zeros_ts = np.zeros_like(x_ts)
            zeros_img = np.zeros_like(x_img)
            zeros_features = np.zeros_like(x_features)

            # Perform predictions with ablation
            # Model expects inputs in order: [image_input, ts_input, feature_input]
            p_joint = real_model.predict([x_img, x_ts, x_features], verbose=0)[0][0]
            
            # For p_1d (temporal-only), zero out image and features
            p_1d = real_model.predict([zeros_img, x_ts, zeros_features], verbose=0)[0][0]
            
            # For p_2d (image-only), zero out temporal and features
            p_2d = real_model.predict([x_img, zeros_ts, zeros_features], verbose=0)[0][0]

            disagreement = abs(p_1d - p_2d)

            results.append({
                'target_id': target_id,
                'period': period,
                'true_label': true_label,
                'prob_joint': p_joint,
                'prob_1d': p_1d,
                'prob_2d': p_2d,
                'disagreement': disagreement
            })

            # Save periodically
            if len(results) % config.CHECKPOINT_INTERVAL == 0:
                pd.DataFrame(results).to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
                results = [] # Clear results after saving
                logger.info(f"Saved checkpoint to {RESULTS_CSV_PATH}")

        except Exception as e:
            logger.error(f"Error processing target {target_id} from {file_path}: {e}", exc_info=True)
            continue

    # Save any remaining results
    if results:
        # Check if the file is empty to write header, otherwise append
        write_header = not os.path.exists(RESULTS_CSV_PATH) or os.stat(RESULTS_CSV_PATH).st_size == 0
        pd.DataFrame(results).to_csv(RESULTS_CSV_PATH, mode='a', header=write_header, index=False)
        logger.info(f"Saved final results to {RESULTS_CSV_PATH}")

    logger.info("Naked inference completed.")

if __name__ == "__main__":
    run_naked_inference()
