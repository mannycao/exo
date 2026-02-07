import os
import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from tqdm import tqdm
import argparse

# Ensure the project root is in the sys.path for importing config and other modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from models.multimodal_model import build_multimodal_fusion_model
from data.dataset_generator import create_dataset
from cacl_explainer import CACLFeatureExtractor, explain_with_cacl
from sklearn.preprocessing import LabelEncoder

# --- Logging Setup ---
LOG_FILE = Path(__file__).resolve().parent / "full_survey.log"
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Explicitly set logger level to DEBUG

# --- Paths ---
OUTPUT_CSV_PATH = config.SURVEY_RESULTS_DIR / "survey_results_v3.csv"

def scan_single_directory(root_dir: Path, assigned_type: str):
    """
    Recursively scans a single directory for FITS/npz/tbl files.
    Returns a list of dictionaries with {'file_path', 'file_type'}.
    """
    logger.info(f"Scanning {assigned_type} files in: {root_dir}")
    files_in_dir = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.fits', '.npz', '.tbl')):
                file_path = Path(root) / file
                files_in_dir.append({
                    'file_path': file_path.as_posix(),
                    'file_type': assigned_type # Assign the type directly from the scan source
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
    parser = argparse.ArgumentParser(description="Run a full survey for exoplanet detection using a multimodal model.")
    parser.add_argument('--planets_dir', type=str,
                        default=str(config.CONFIRMED_PLANETS_DIR), # Use config defaults
                        help='Path to the directory containing confirmed planets light curve files.')
    parser.add_argument('--false_positives_dir', type=str,
                        default=str(config.FALSE_POSITIVES_DIR), # Use config defaults
                        help='Path to the directory containing false positives light curve files.')
    args = parser.parse_args()

    logger.info("Starting full survey with multimodal discovery pipeline.")
    
    survey_results = []
    processed_count = 0

    # 1. Load Combined Metadata
    data_files_dir = Path(__file__).resolve().parents[1] / "data_files"
    FULL_METADATA_CSV = data_files_dir / "full_metadata.csv"

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
        metadata_map[row['target_id']] = {
            'period': row.get('period'),
            'true_label': row.get('true_label') # Use 'true_label' from the new CSV
        }
    logger.info(f"Final metadata_map populated with {len(metadata_map)} targets.")
    logger.debug(f"DEBUG: Sample from metadata_map (first 5 items): {dict(list(metadata_map.items())[:5])}")

    # Initialize LabelEncoder for consistent label mapping
    label_encoder = LabelEncoder()
    # Fit with unique non-null true_labels from metadata, plus default 'CANDIDATE' if necessary
    known_labels = final_metadata_df['true_label'].dropna().unique().tolist()
    if 'CANDIDATE' not in known_labels:
        known_labels.append('CANDIDATE')
    if 'SKIPPED_ID_UNKNOWN' not in known_labels:
        known_labels.append('SKIPPED_ID_UNKNOWN')
    if 'SKIPPED_ID_INVALID' not in known_labels:
        known_labels.append('SKIPPED_ID_INVALID')
    if 'SKIPPED_PERIOD_MISSING' not in known_labels:
        known_labels.append('SKIPPED_PERIOD_MISSING')
    if 'SKIPPED_EMPTY_DATA' not in known_labels:
        known_labels.append('SKIPPED_EMPTY_DATA')
    if 'SKIPPED_PROB2D_ZERO' not in known_labels:
        known_labels.append('SKIPPED_PROB2D_ZERO')
    if 'ERROR_PROCESSING' not in known_labels:
        known_labels.append('ERROR_PROCESSING')
    
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
    
    cacl_explainer = CACLFeatureExtractor(
        model_path=config.CACL_MODEL_PATH,
        partitions=cacl_partitions,
        input_dim=config.FIXED_LENGTH,
        output_dim=config.CACL_OUTPUT_DIM,
        proj_dim=config.CACL_PROJ_DIM,
        nhead=config.CACL_NHEAD,
        num_layers=config.CACL_NUM_LAYERS
    )
    if config.CACL_MODEL_PATH.exists():
        logger.info(f"CACL Explainer initialized from {config.CACL_MODEL_PATH}")
    else:
        logger.warning(f"CACL model weights not found at {config.CACL_MODEL_PATH}. Disagreement scores might be inaccurate.")
        # Create a dummy explainer that returns 0 disagreement
        class DummyCACLFeatureExtractor:
            def __init__(self, *args, **kwargs): pass
            def get_transformer_features(self, data_sample_ts_batch): return np.zeros((1, config.CACL_OUTPUT_DIM))
        cacl_explainer = DummyCACLFeatureExtractor()

    # 3. Scan Light Curve Directories
    all_light_curve_files = []
    
    planets_dir_path = Path(args.planets_dir)
    if planets_dir_path.is_dir():
        all_light_curve_files.extend(scan_single_directory(planets_dir_path, config.FILE_TYPE_CONFIRMED_PLANET))
    else:
        logger.warning(f"Confirmed planets directory not found: {planets_dir_path}. Skipping.")

    false_positives_dir_path = Path(args.false_positives_dir)
    if false_positives_dir_path.is_dir():
        all_light_curve_files.extend(scan_single_directory(false_positives_dir_path, config.FILE_TYPE_FALSE_POSITIVE))
    else:
        logger.warning(f"False positives directory not found: {false_positives_dir_path}. Skipping.")

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
        
        # TESS filename parsing
        match_tess = re.search(r'tess[0-9]+-s[0-9]+-([0-9]+)-', file_name)
        if match_tess:
            target_id_str = match_tess.group(1)
        else:
            # Kepler filename parsing
            match_kplr = re.search(r'kplr(\d+)', file_name)
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
            continue

        # Retrieve metadata
        metadata_entry = metadata_map.get(target_id, {})
        period_from_metadata = metadata_entry.get('period')
        true_label_from_metadata = metadata_entry.get('true_label')
        
        # Critical Skip: If period is missing/NaN, skip the target
        if pd.isna(period_from_metadata):
            logger.warning(f"Skipping {target_id} ({file_path}): Period missing/NaN in metadata. Required for Regime Analysis.")
            survey_results.append({
                'target_id': target_id,
                'period': np.nan,
                'true_label': 'SKIPPED_PERIOD_MISSING',
                'prob_1d': np.nan,
                'prob_2d': np.nan,
                'disagreement': np.nan
            })
            continue

        true_label_str = true_label_from_metadata if true_label_from_metadata else (file_type_from_scan.upper() if file_type_from_scan != 'UNKNOWN' else 'CANDIDATE')

        try:
            label_encoder.transform([true_label_str]) # Check if label is known
        except ValueError:
            logger.warning(f"Unknown true_label_str '{true_label_str}' for {target_id}. Setting to 'CANDIDATE'.")
            true_label_str = 'CANDIDATE'
        # true_label_numeric is not directly used in the output CSV, but useful for internal logic if needed
        # true_label_numeric = label_encoder.transform([true_label_str])[0] 
        
        # period_from_metadata is already retrieved above


        try:
            # create_dataset expects a list of file_info_dicts
            # Pass the file_type_from_scan as the 'type' for create_dataset
            X_ts, X_img, X_features, _, _, _ = create_dataset(
                file_paths=[{'file_path': file_path, 'type': file_type_from_scan}],
                output_dir=None, # Don't save individual processed data during survey
                image_size=config.IMAGE_SIZE
            )

            # Crucial Fix: Verify inputs are not empty or all zeros
            if (X_ts is None or X_img is None or X_features is None or
                    len(X_ts) == 0 or np.all(X_img == 0)):
                logger.warning(f"Skipping {target_id} ({file_path}): create_dataset returned empty or all-zero data views (Empty Image Warning).")
                survey_results.append({
                    'target_id': target_id,
                    'period': period_from_metadata,
                    'true_label': 'SKIPPED_EMPTY_DATA',
                    'prob_1d': np.nan,
                    'prob_2d': np.nan,
                    'disagreement': np.nan
                })
                continue

            # Run multimodal model prediction
            # The model outputs a single probability for the positive class (planet)
            mean_confidence_raw = multimodal_model.predict([X_img, X_ts, X_features])[0][0]

            # Get CACL explanation for max_disagreement
            cacl_explanation = explain_with_cacl(
                data_sample=X_ts[0], # Assuming first sample if batch size > 1
                cacl_explainer=cacl_explainer,
                dependency_matrix=None, # Not pre-computed for survey
                context_avg_embeddings=None, # Not pre-computed for survey
            )
            max_disagreement = cacl_explanation.get('max_disagreement', 0.0)

            # Derive prob_1d and prob_2d
            prob_1d = float(np.clip(mean_confidence_raw + max_disagreement / 2, 0.0, 1.0))
            prob_2d = float(np.clip(mean_confidence_raw - max_disagreement / 2, 0.0, 1.0))

            # Critical Fix: If prob_2d is exactly 0.0, log a warning and skip
            if prob_2d == 0.0:
                logger.warning(f"Skipping {target_id} ({file_path}): prob_2d is exactly 0.0. Indicates problematic inference.")
                survey_results.append({
                    'target_id': target_id,
                    'period': period_from_metadata,
                    'true_label': 'SKIPPED_PROB2D_ZERO',
                    'prob_1d': prob_1d, # Keep calculated prob_1d for debugging if needed
                    'prob_2d': prob_2d,
                    'disagreement': np.nan # Disagreement is meaningless if prob_2d is 0.0 and skipped
                })
                continue

            survey_results.append({
                'target_id': target_id,
                'period': period_from_metadata,
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
                'period': period_from_metadata,
                'true_label': 'ERROR_PROCESSING',
                'prob_1d': np.nan,
                'prob_2d': np.nan,
                'disagreement': np.nan
            })
            continue

        # Checkpointing
        if (processed_count) % config.CHECKPOINT_INTERVAL == 0 and processed_count > 0:
            pd.DataFrame(survey_results).to_csv(OUTPUT_CSV_PATH, index=False)
            logger.info(f"Checkpoint saved: {OUTPUT_CSV_PATH} ({processed_count} items processed).")
            # Clear results to save memory, if needed for extremely large runs, but for 45k targets, accumulating is fine.
            # survey_results = [] # Uncomment if memory becomes an issue


    # 5. Final Output
    final_df = pd.DataFrame(survey_results)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(f"Final survey results saved to: {OUTPUT_CSV_PATH}")

    logger.info(f"Full survey complete: Processed {processed_count} valid targets.")
    logger.info(f"Total files scanned: {len(all_light_curve_files)}")

if __name__ == "__main__":
    main()