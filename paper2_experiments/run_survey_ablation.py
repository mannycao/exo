import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf

# Project specific imports
import config as root_config # Refers to the root config.py
from discovery_stack.ingest import IngestionEngine
from models.multimodal_model import build_multimodal_fusion_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def run_survey_ablation():
    print("Starting the 45k Survey using Modality Ablation.")

    # --- Configuration ---
    TIMESERIES_SHAPE = (root_config.FIXED_LENGTH,)
    IMAGE_SHAPE = root_config.IMAGE_SIZE + (1,) 
    FEATURE_SHAPE = (1024,) # As inferred from multimodal_model.py
    MODEL_PATH = root_config.MODEL_PATH
    LIGHT_CURVES_DIR = root_config.LIGHT_CURVE_DIR
    OUTPUT_CSV_PATH = root_config.SURVEY_RESULTS_DIR / 'survey_results_ablation.csv'
    CHECKPOINT_INTERVAL = root_config.CHECKPOINT_INTERVAL

    # --- 1. Load Metadata ---
    metadata_path = root_config.DATA_DIR / 'full_metadata.csv'
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Please ensure the Kepler metadata file is available.")
        return

    metadata_df = pd.read_csv(metadata_path, index_col='target_id')
    print(f"Loaded Kepler metadata from {metadata_path}. Shape: {metadata_df.shape}")

    # --- 2. Load and Prepare Model ---
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Please ensure the trained model exists.")
        return

    print(f"Loading full multimodal model from {MODEL_PATH}...")
    full_model = build_multimodal_fusion_model(IMAGE_SHAPE, TIMESERIES_SHAPE, FEATURE_SHAPE)
    full_model.load_weights(MODEL_PATH)
    print("Full multimodal model loaded successfully.")

    # --- Prepare "Blinding" Masks ---
    zeros_1d_input = np.zeros(TIMESERIES_SHAPE, dtype=np.float32)
    zeros_2d_input = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    zeros_feature_input = np.zeros(FEATURE_SHAPE, dtype=np.float32)

    # --- 3. Scan Files ---
    if not LIGHT_CURVES_DIR.exists():
        print(f"Error: Light curves directory '{LIGHT_CURVES_DIR}' not found.")
        print("Please ensure Kepler light curve FITS files are downloaded into this directory.")
        return
    
    all_files_in_dir = os.listdir(LIGHT_CURVES_DIR)
    kepler_fits_files = [f for f in all_files_in_dir if re.match(r'kplr([0-9]+)[_|-][^.]*\.fits$', f)]
    fits_file_paths = [LIGHT_CURVES_DIR / f for f in kepler_fits_files]
    print(f"Found {len(fits_file_paths)} Kepler FITS files in {LIGHT_CURVES_DIR}.")

    current_run_results = []
    ingestion_engine = IngestionEngine()

    root_config.SURVEY_RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Check if a partial file exists to resume, otherwise start fresh
    if OUTPUT_CSV_PATH.exists():
        print(f"Resuming from existing results file: {OUTPUT_CSV_PATH}")
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        processed_target_ids = set(existing_df['target_id'].unique())
        print(f"Found {len(processed_target_ids)} already processed targets.")
    else:
        processed_target_ids = set()

    # --- 4. Processing Loop ---
    debug_predict_count = 0
    DEBUG_PREDICT_LIMIT = 5 # Print debug info for the first 5 processed files

    for i, fits_file_path in enumerate(tqdm(fits_file_paths, desc="Processing Kepler FITS files")):
        match_kepler = re.search(r'kplr([0-9]+)[_|-]', fits_file_path.name)
        if not match_kepler:
            print(f"Warning: Could not extract KIC ID from {fits_file_path.name}. Skipping.")
            continue
        
        target_id = int(match_kepler.group(1))

        if target_id in processed_target_ids:
            continue

        if target_id not in metadata_df.index:
            continue
        
        target_metadata = metadata_df.loc[target_id]
        period = target_metadata.get('period', 0.0)
        true_label = target_metadata.get('true_label', 'UNKNOWN')

        x_1d, x_2d = ingestion_engine.extract_features(str(fits_file_path))

        if x_1d is None or x_2d is None:
            prob_joint = 0.0
            prob_1d = 0.0
            prob_2d = 0.0
            disagreement = 0.0
            print(f"Warning: Feature extraction failed for {fits_file_path.name}. Recording 0.0 probabilities.")
        else:
            x_1d_batch = np.expand_dims(x_1d, axis=0)
            x_2d_batch = np.expand_dims(x_2d, axis=0)
            zeros_1d_batch = np.expand_dims(zeros_1d_input, axis=0)
            zeros_2d_batch = np.expand_dims(zeros_2d_input, axis=0)
            zeros_feature_batch = np.expand_dims(zeros_feature_input, axis=0)
            
            if debug_predict_count < DEBUG_PREDICT_LIMIT:
                print(f"\n--- DEBUG: {fits_file_path.name} Input Batch Stats ---")
                print(f"x_1d_batch stats: min={np.min(x_1d_batch):.4f}, max={np.max(x_1d_batch):.4f}, mean={np.mean(x_1d_batch):.4f}, std={np.std(x_1d_batch):.4f}")
                print(f"x_2d_batch stats: min={np.min(x_2d_batch):.4f}, max={np.max(x_2d_batch):.4f}, mean={np.mean(x_2d_batch):.4f}, std={np.std(x_2d_batch):.4f}")
                print(f"zeros_feature_batch stats: min={np.min(zeros_feature_batch):.4f}, max={np.max(zeros_feature_batch):.4f}, mean={np.mean(zeros_feature_batch):.4f}, std={np.std(zeros_feature_batch):.4f}")
                print(f"zeros_1d_batch stats: min={np.min(zeros_1d_batch):.4f}, max={np.max(zeros_1d_batch):.4f}, mean={np.mean(zeros_1d_batch):.4f}, std={np.std(zeros_1d_batch):.4f}")
                print(f"zeros_2d_batch stats: min={np.min(zeros_2d_batch):.4f}, max={np.max(zeros_2d_batch):.4f}, mean={np.mean(zeros_2d_batch):.4f}, std={np.std(zeros_2d_batch):.4f}")

                # Also make a dummy prediction with random data to check model functionality
                dummy_x_1d = np.random.rand(1, *TIMESERIES_SHAPE).astype(np.float32)
                dummy_x_2d = np.random.rand(1, *IMAGE_SHAPE).astype(np.float32)
                dummy_x_feature = np.random.rand(1, *FEATURE_SHAPE).astype(np.float32)
                dummy_pred = full_model.predict([dummy_x_2d, dummy_x_1d, dummy_x_feature], verbose=0).flatten()[0]
                print(f"DEBUG: Dummy Random Input Prediction: {dummy_pred}")


            # Predict Joint
            # Model inputs order: [image_input, ts_input, feature_input]
            p_joint = full_model.predict([x_2d_batch, x_1d_batch, zeros_feature_batch], verbose=0).flatten()[0]

            # Predict 1D Only (blind 2D and features)
            p_1d = full_model.predict([zeros_2d_batch, x_1d_batch, zeros_feature_batch], verbose=0).flatten()[0]

            # Predict 2D Only (blind 1D and features)
            p_2d = full_model.predict([x_2d_batch, zeros_1d_batch, zeros_feature_batch], verbose=0).flatten()[0]
            
            prob_joint = float(p_joint)
            prob_1d = float(p_1d)
            prob_2d = float(p_2d)
            disagreement = abs(prob_1d - prob_2d)

            if debug_predict_count < DEBUG_PREDICT_LIMIT:
                print(f"Actual Predictions: Joint={prob_joint}, 1D={prob_1d}, 2D={prob_2d}, Disagreement={disagreement}")
                debug_predict_count += 1

    # After loop, save any remaining results that weren't part of a full checkpoint batch
    if current_run_results:
        temp_df = pd.DataFrame(current_run_results)
        if OUTPUT_CSV_PATH.exists():
            temp_df.to_csv(OUTPUT_CSV_PATH, mode='a', header=False, index=False)
        else:
            temp_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Final batch saved to {OUTPUT_CSV_PATH}.")

    # Finally, re-load the entire CSV to get the complete, correct DataFrame for final message
    final_df_after_run = pd.read_csv(OUTPUT_CSV_PATH)
    total_processed_targets = len(final_df_after_run)
    print(f"Survey complete. Total processed targets in final CSV: {total_processed_targets}. Results saved to {OUTPUT_CSV_PATH}.")

if __name__ == "__main__":
    run_survey_ablation()