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

# Ensure the project root is in the sys.path for importing config and other modules
# Adjusting sys.path to point to the phd directory, which is the project root for module imports
project_root = Path(__file__).resolve().parents[1] # Go up one level from paper2_experiments/ to phd/
sys.path.insert(0, str(project_root))

# Import the main config.py for general settings and paper2_experiments.config for specific paths
import config # Main project config
from paper2_experiments import config as paper2_config # Specific config for paper2 experiments

# Import necessary modules from the project
from models.multimodal_model import build_multimodal_fusion_model
from data.dataset_generator import create_dataset
from cacl_explainer import CACLFeatureExtractor, explain_with_cacl # Assuming cacl_explainer is at project root
from sklearn.preprocessing import LabelEncoder


# --- Logging Setup ---
LOG_FILE = project_root / "paper2_experiments" / "full_survey_production.log"
# Ensure the directory for the log file exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure the root logger with a basic level, but no handlers yet.
# This prevents basicConfig from adding a default StreamHandler if there are no handlers.
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
# Use the specific paths from paper2_experiments/config.py for the full dataset
CONFIRMED_PLANETS_DATA_DIR = Path(paper2_config.CONFIRMED_PLANETS_DIR)
FALSE_POSITIVES_DATA_DIR = Path(paper2_config.FALSE_POSITIVES_DIR)

# Output CSV path as specified by the user
OUTPUT_CSV_PATH = project_root / "paper2_experiments" / "survey_results_final_production.csv"

# Checkpoint interval as specified by the user
CHECKPOINT_INTERVAL = 500

def scan_single_directory(root_dir: Path, assigned_type: str):
    """
    Recursively scans a single directory for FITS/npz/tbl files.
    Returns a list of dictionaries with {'file_path', 'file_type'}.
    """
    if not root_dir.is_dir():
        logger.warning(f"Directory not found: {root_dir}. Skipping scan for {assigned_type}.")
        return []

    logger.info(f"Scanning {assigned_type} files in: {root_dir}")
    files_in_dir = []
    # Use glob for potentially faster and more concise scanning
    for ext in ['*.fits', '*.npz', '*.tbl']:
        for file_path in root_dir.glob(f"**/{ext}"):
            files_in_dir.append({
                'file_path': file_path.as_posix(),
                'file_type': assigned_type
            })
    logger.info(f"Found {len(files_in_dir)} files of type '{assigned_type}' in {root_dir}.")
    return files_in_dir

def get_model_input_shapes():
    """Derives model input shapes from config."""
    image_shape = config.IMAGE_SIZE + (1,)  # Assuming grayscale image input
    timeseries_shape = (config.FIXED_LENGTH, 1)  # Assuming 1 feature per timestep
    feature_shape = (config.FEATURE_VECTOR_LENGTH,)
    return image_shape, timeseries_shape, feature_shape

def main():
    logger.info("Starting full survey with multimodal discovery pipeline (Naked Inference - Production).")
    
    survey_results = []
    processed_count = 0

    # 1. Load Combined Metadata
    # Using full_metadata_v2.csv as specified by the user
    FULL_METADATA_CSV = project_root / "data_files" / "full_metadata_v2.csv"

    if FULL_METADATA_CSV.exists():
        final_metadata_df = pd.read_csv(FULL_METADATA_CSV)
        logger.info(f"Loaded combined metadata from: {FULL_METADATA_CSV}")
        logger.debug(f"DEBUG: Combined metadata shape: {final_metadata_df.shape}")
        logger.debug(f"DEBUG: Combined metadata columns: {final_metadata_df.columns.tolist()}")
    else:
        logger.error(f"Combined metadata CSV not found at {FULL_METADATA_CSV}. Aborting.")
        sys.exit(1)

    metadata_map = {}
    for _, row in final_metadata_df.iterrows():
        # Ensure target_id is an integer for consistent lookup
        try:
            target_id = int(row['target_id'])
            metadata_map[target_id] = {
                'period': row.get('period'),
                'true_label': row.get('true_label')
            }
        except ValueError:
            logger.warning(f"Skipping metadata entry with invalid target_id: {row['target_id']}")
    logger.info(f"Final metadata_map populated with {len(metadata_map)} valid targets.")
    logger.debug(f"DEBUG: Sample from metadata_map (first 5 items): {dict(list(metadata_map.items())[:5])}")

    # Initialize LabelEncoder for consistent label mapping
    label_encoder = LabelEncoder()
    # Fit with unique non-null true_labels from metadata, plus default 'CANDIDATE' and SKIP statuses
    known_labels = final_metadata_df['true_label'].dropna().unique().tolist()
    # Add potential status labels
    potential_status_labels = [
        'CANDIDATE', 'SKIPPED_ID_UNKNOWN', 'SKIPPED_ID_INVALID', 
        'SKIPPED_PERIOD_MISSING', 'SKIPPED_EMPTY_DATA', 
        'SKIPPED_PROB2D_ZERO', 'ERROR_PROCESSING', 
        'FALSE_POSITIVE', 'CONFIRMED_PLANET' # Explicitly add these if they might come from file_type_from_scan
    ]
    for label in potential_status_labels:
        if label not in known_labels:
            known_labels.append(label)
    
    label_encoder.fit(known_labels)
    logger.info(f"LabelEncoder fitted with categories: {label_encoder.classes_.tolist()}")

    # 2. Initialize Model and CACL Explainer
    image_shape, timeseries_shape, feature_shape = get_model_input_shapes()

    multimodal_model = build_multimodal_fusion_model(
        image_shape=image_shape,
        timeseries_shape=timeseries_shape,
        feature_shape=feature_shape
    )
    if config.MODEL_PATH.exists():
        multimodal_model.load_weights(config.MODEL_PATH)
        logger.info(f"Multimodal model loaded from {config.MODEL_PATH}")
    else:
        logger.error(f"Multimodal model weights not found at {config.MODEL_PATH}. Exiting.")
        sys.exit(1)

    # Initialize CACL Explainer
    sample_light_curve_length = config.FIXED_LENGTH
    partition_size = sample_light_curve_length // config.CACL_K_PARTITIONS
    cacl_partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(config.CACL_K_PARTITIONS - 1)]
    cacl_partitions.append([j for j in range((config.CACL_K_PARTITIONS - 1) * partition_size, sample_light_curve_length)])
    cacl_partitions = [p for p in cacl_partitions if p] # Filter out empty partitions
    
    # Check if config.CACL_MODEL_PATH exists and is valid before passing to CACLFeatureExtractor
    if config.CACL_MODEL_PATH.exists():
        cacl_explainer = CACLFeatureExtractor(
            model_path=config.CACL_MODEL_PATH,
            partitions=cacl_partitions,
            input_dim=config.FIXED_LENGTH,
            output_dim=config.CACL_OUTPUT_DIM,
            proj_dim=config.CACL_PROJ_DIM,
            nhead=config.CACL_NHEAD,
            num_layers=config.CACL_NUM_LAYERS
        )
        logger.info(f"CACL Explainer initialized from {config.CACL_MODEL_PATH}")
    else:
        logger.warning(f"CACL model weights not found at {config.CACL_MODEL_PATH}. Initializing dummy CACL Explainer. Disagreement scores will be 0.")
        # Create a dummy explainer that returns 0 disagreement
        class DummyCACLFeatureExtractor:
            def __init__(self, *args, **kwargs): pass
            def get_transformer_features(self, data_sample_ts_batch): return np.zeros((1, config.CACL_OUTPUT_DIM))
            def get_disagreement(self, data_sample, original_pred, feature_masks): return 0.0 # Dummy disagreement
        cacl_explainer = DummyCACLFeatureExtractor()

    # 3. Scan Light Curve Directories (False Positives + Candidates/Confirmed Planets)
    all_light_curve_files = []
    
    # Scan False Positives
    all_light_curve_files.extend(scan_single_directory(FALSE_POSITIVES_DATA_DIR, config.FILE_TYPE_FALSE_POSITIVE))
    # Scan Confirmed Planets (serving as candidates in this context for the full survey)
    all_light_curve_files.extend(scan_single_directory(CONFIRMED_PLANETS_DATA_DIR, config.FILE_TYPE_CONFIRMED_PLANET))

    if not all_light_curve_files:
        logger.error("No light curve files found in specified directories. Aborting.")
        sys.exit(1)

    logger.info(f"Found {len(all_light_curve_files)} light curve files for processing.")
    
    # 4. Robust Inference Loop with Checkpointing
    for i, file_info in enumerate(tqdm(all_light_curve_files, desc="Running Full Survey")):
        file_path = file_info['file_path']
        file_type_from_scan = file_info['file_type'] # This will be 'confirmed_planet' or 'false_positive'
        
        # Extract target_id from file_path (or filename)
        target_id_str = None # Raw string ID from filename
        file_name = Path(file_path).name
        
        # TESS filename parsing (e.g., tess2018206045859-s0001-0000000000-0112-s_lc.fits)
        match_tess = re.search(r'tess\d+-s\d+-(\d+)-\d+-s_lc\.fits', file_name)
        if match_tess:
            target_id_str = match_tess.group(1)
        else:
            # Kepler filename parsing (e.g., kplr008547463-2010265121752_llc.fits)
            match_kplr = re.search(r'kplr(\d+)-\d+_llc\.fits', file_name)
            if match_kplr:
                target_id_str = match_kplr.group(1)
        
        if target_id_str is None:
            logger.warning(f"Could not extract target_id from filename: {file_name}. Skipping file {file_path}.")
            survey_results.append({
                'target_id': 'UNKNOWN',
                'period': np.nan,
                'true_label': 'SKIPPED_ID_UNKNOWN',
                'prob_1d': np.nan,
                'prob_2d': np.nan,
                'disagreement': np.nan
            })
            # Save progress if checkpoint is due, even for skipped files
            processed_count += 1
            if processed_count % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
                pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
                logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
            continue
        
        # Convert target_id to int for metadata lookup
        try:
            target_id = int(target_id_str)
        except ValueError:
            logger.warning(f"Invalid target_id '{target_id_str}' extracted from filename: {file_name}. Skipping file {file_path}.")
            survey_results.append({
                'target_id': target_id_str,
                'period': np.nan,
                'true_label': 'SKIPPED_ID_INVALID',
                'prob_1d': np.nan,
                'prob_2d': np.nan,
                'disagreement': np.nan
            })
            # Save progress if checkpoint is due, even for skipped files
            processed_count += 1
            if processed_count % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
                pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
                logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
            continue

        # Retrieve metadata
        metadata_entry = metadata_map.get(target_id, {})
        period_from_metadata = metadata_entry.get('period')
        true_label_from_metadata = metadata_entry.get('true_label')
        
        # Determine true_label_str. Prioritize metadata, then file type, default to CANDIDATE.
        true_label_str = 'CANDIDATE'
        if true_label_from_metadata:
            true_label_str = true_label_from_metadata
        elif file_type_from_scan == config.FILE_TYPE_CONFIRMED_PLANET:
            true_label_str = 'CONFIRMED_PLANET'
        elif file_type_from_scan == config.FILE_TYPE_FALSE_POSITIVE:
            true_label_str = 'FALSE_POSITIVE'
            
        try:
            label_encoder.transform([true_label_str]) # Check if label is known
        except ValueError:
            logger.warning(f"Unknown true_label_str '{true_label_str}' for {target_id}. Setting to 'CANDIDATE'.")
            true_label_str = 'CANDIDATE' # Fallback if label is truly unknown to encoder

        # No longer critical skip if period is missing/NaN in metadata.
        # create_dataset will now always attempt processing and period detection.
        
        try:
            # create_dataset expects a list of file_info_dicts
            # Pass the file_type_from_scan as the 'type' for create_dataset
            X_ts, X_img, X_features, _, all_pipeline_results, _ = create_dataset(
                file_paths=[{'file_path': file_path, 'type': file_type_from_scan}],
                output_dir=None, # Don't save individual processed data during survey
                image_size=config.IMAGE_SIZE
            )

            # Get the period that create_dataset used/detected for phase folding
            # This is derived from periodicity_data.get('median_period') inside process_single_file
            # which is then stored in all_pipeline_results_raw.
            period_used_for_folding = all_pipeline_results[0].get('periodicity', np.nan)

            # Determine the period to use for output: prioritize metadata, else use detected period
            output_period = period_from_metadata if pd.notna(period_from_metadata) else period_used_for_folding

            # Crucial Fix: Verify inputs are not empty or all zeros
            if (X_ts is None or X_img is None or X_features is None or
                    len(X_ts) == 0 or np.all(X_img == 0)):
                logger.warning(f"Skipping {target_id} ({file_path}): create_dataset returned empty or all-zero data views (Empty Image Warning).")
                survey_results.append({
                    'target_id': target_id,
                    'period': output_period, # Use determined output period
                    'true_label': 'SKIPPED_EMPTY_DATA',
                    'prob_1d': np.nan,
                    'prob_2d': np.nan,
                    'disagreement': np.nan
                })
                # Save progress if checkpoint is due, even for skipped files
                processed_count += 1
                if processed_count % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
                    pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
                    logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
                continue

            # Run multimodal model prediction
            # The model outputs a single probability for the positive class (planet)
            mean_confidence_raw = multimodal_model.predict([X_img, X_ts, X_features])[0][0]

            # Get CACL explanation for max_disagreement
            # Ensure CACL explainer is not the dummy one before calling get_disagreement
            max_disagreement = 0.0
            if not isinstance(cacl_explainer, DummyCACLFeatureExtractor):
                 cacl_explanation = explain_with_cacl(
                    data_sample=X_ts[0], # Assuming first sample if batch size > 1
                    cacl_explainer=cacl_explainer,
                    # dependency_matrix=None, # Not pre-computed for survey
                    # context_avg_embeddings=None, # Not pre-computed for survey
                    model=multimodal_model # Pass the model for CACL to predict
                 )
                 max_disagreement = cacl_explanation.get('max_disagreement', 0.0)

            # Derive prob_1d and prob_2d (Virtual Lobotomy logic)
            prob_1d = float(np.clip(mean_confidence_raw + max_disagreement / 2, 0.0, 1.0))
            prob_2d = float(np.clip(mean_confidence_raw - max_disagreement / 2, 0.0, 1.0))

            # Critical Fix: If prob_2d is exactly 0.0, log a warning and skip
            if prob_2d <= 0.0001 and prob_2d >= 0.0: # Check for near zero to handle float precision issues
                logger.warning(f"Skipping {target_id} ({file_path}): prob_2d is essentially zero ({prob_2d}). Indicates problematic inference.")
                survey_results.append({
                    'target_id': target_id,
                    'period': output_period, # Use determined output period
                    'true_label': 'SKIPPED_PROB2D_ZERO',
                    'prob_1d': prob_1d, # Keep calculated prob_1d for debugging if needed
                    'prob_2d': prob_2d,
                    'disagreement': np.nan # Disagreement is meaningless if prob_2d is 0.0 and skipped
                })
                # Save progress if checkpoint is due, even for skipped files
                processed_count += 1
                if processed_count % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
                    pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
                    logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
                continue

            survey_results.append({
                'target_id': target_id,
                'period': output_period, # Use determined output period
                'true_label': true_label_str,
                'prob_1d': prob_1d,
                'prob_2d': prob_2d,
                'disagreement': abs(prob_1d - prob_2d)
            })
            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing target {target_id} from {file_path}: {e}", exc_info=True)
            survey_results.append({
                'target_id': target_id,
                'period': np.nan, # No period can be determined in case of ERROR_PROCESSING
                'true_label': 'ERROR_PROCESSING',
                'prob_1d': np.nan,
                'prob_2d': np.nan,
                'disagreement': np.nan
            })
            # Save progress if checkpoint is due, even for skipped files
            processed_count += 1
            if processed_count % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
                pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
                logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
            continue

        # Checkpointing
        if (processed_count) % CHECKPOINT_INTERVAL == 0 and processed_count > 0:
            pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
            # Optionally clear survey_results if memory is a major concern, and append to the CSV.
            # For now, we'll accumulate and overwrite the CSV on each checkpoint.


    # 5. Final Output - ensure all results are saved even if checkpointing didn't catch the last few
    if survey_results: # Only save if there are results
        final_df = pd.DataFrame(survey_results)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info(f"Final survey results saved to: {OUTPUT_CSV_PATH}")
    else:
        logger.warning("No survey results to save.")

    logger.info(f"Full survey complete: Processed {processed_count} targets.")
    logger.info(f"Total files scanned: {len(all_light_curve_files)}")
    
    logging.shutdown()

if __name__ == "__main__":
    main()