import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import glob

# Ensure the project root is in the sys.path for importing config and other modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # Corrected to parents[1] earlier



import argparse # Added for argument parsing
import config # Added to import the config.LOG_LEVEL
from models.multimodal_model import build_multimodal_fusion_model
from cacl_explainer import CACLFeatureExtractor, explain_with_cacl
from data.dataset_generator import create_dataset
# find_transits_bls is not directly used in this script, but other functions might call it
# from detection.transit_detector import find_transits_bls
from config import MODEL_PATH, CACL_MODEL_PATH, CHECKPOINT_INTERVAL, SURVEY_RESULTS_DIR, FILE_TYPE_CONFIRMED_PLANET, FILE_TYPE_FALSE_POSITIVE


LOG_FILE = Path(__file__).resolve().parent / "full_scale_survey.log"

# --- Logging Setup ---
logging.basicConfig(level=config.LOG_LEVEL, # Use LOG_LEVEL from config
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def scan_data_directories(confirmed_dir, false_positives_dir):
    """
    Scans specified directories for FITS/npz files and extracts target_ids and labels.
    Returns a list of dictionaries with {'target_id', 'file_path', 'type'}.
    """
    target_data = []

    def process_dir(directory, file_type):
        logger.info(f"Scanning {directory} for {file_type} files...")
        found_in_dir = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.fits', '.npz', '.tbl')): # Added .tbl as observed in directory structure
                    file_path = Path(root) / file
                    # Extract target_id from filename (e.g., kplr010000941-...)
                    # Assuming Kepler IDs are at the beginning of the filename before the first hyphen
                    target_id = file.split('-')[0].replace('kplr', '') # Remove 'kplr' prefix
                    found_in_dir.append({
                        'target_id': target_id,
                        'file_path': file_path.as_posix(),
                        'type': file_type # 'confirmed_planet' or 'false_positive'
                    })
        logger.info(f"Found {len(found_in_dir)} files in {directory}.")
        return found_in_dir

    # Use the File Type Constants from config.py
    target_data.extend(process_dir(confirmed_dir, config.FILE_TYPE_CONFIRMED_PLANET))
    target_data.extend(process_dir(false_positives_dir, config.FILE_TYPE_FALSE_POSITIVE))
    
    return target_data

def get_model_input_shapes():
    """
    Derives model input shapes from config.
    """
    image_shape = config.IMAGE_SIZE + (1,) # Assuming grayscale image input
    timeseries_shape = (config.FIXED_LENGTH, 1) # Assuming 1 feature per timestep
    feature_shape = (config.FEATURE_VECTOR_LENGTH,)
    return image_shape, timeseries_shape, feature_shape

def run_full_scale_survey(planets_dir, false_positives_dir):
    logger.info("Starting full-scale survey for disagreement pipeline.")
    logger.info(f"Confirmed Planets Directory: {planets_dir}")
    logger.info(f"False Positives Directory: {false_positives_dir}")

    # 1. Load the Full Catalog
    all_targets = scan_data_directories(planets_dir, false_positives_dir)
    if not all_targets:
        logger.error("No target files found in specified directories. Aborting.")
        return

    logger.info(f"Total files to be processed: {len(all_targets)}")

    # 2. Initialize Model and CACL Explainer
    image_shape, timeseries_shape, feature_shape = get_model_input_shapes()
    
    # Build multimodal model
    multimodal_model = build_multimodal_fusion_model(
        image_shape=image_shape,
        timeseries_shape=timeseries_shape,
        feature_shape=feature_shape
    )
    multimodal_model.summary(print_fn=logger.info)
    multimodal_model.load_weights(MODEL_PATH)
    logger.info(f"Multimodal model loaded from {MODEL_PATH}")

    # Initialize CACL Explainer
    sample_light_curve_length = config.FIXED_LENGTH
    partition_size = sample_light_curve_length // config.CACL_K_PARTITIONS
    cacl_partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(config.CACL_K_PARTITIONS - 1)]
    cacl_partitions.append([j for j in range((config.CACL_K_PARTITIONS - 1) * partition_size, sample_light_curve_length)])
    cacl_partitions = [p for p in cacl_partitions if p] # Filter out empty partitions
    
    cacl_explainer = CACLFeatureExtractor(
        model_path=CACL_MODEL_PATH,
        partitions=cacl_partitions,
        input_dim=config.FIXED_LENGTH,
        output_dim=config.CACL_OUTPUT_DIM,
        proj_dim=config.CACL_PROJ_DIM,
        nhead=config.CACL_NHEAD,
        num_layers=config.CACL_NUM_LAYERS
    )
    logger.info(f"CACL Explainer initialized from {CACL_MODEL_PATH}")

    survey_results = []
    processed_count = 0
    candidate_count = 0
    rejection_count = 0

    # 3. Batch Processing Loop
    for i, target_info in enumerate(tqdm(all_targets, desc="Processing targets")):
        target_id = target_info['target_id']
        file_path_str = target_info['file_path']
        true_label_str = target_info['type']
        
        try:
            # Ingest/Load data for a single target
            # create_dataset returns a tuple of (X_ts, X_img, X_features, y, all_pipeline_results)
            # We need to wrap it in a list as create_dataset expects a list of items
            X_ts, X_img, X_features, y, all_pipeline_results, all_provenance = create_dataset(
                file_paths=[{'file_path': file_path_str, 'type': true_label_str}],
                output_dir=None, # Don't save individual processed data during survey
                image_size=config.IMAGE_SIZE
            )
            
            if X_ts is None or X_img is None or X_features is None or len(X_ts) == 0:
                logger.warning(f"Skipping {target_id}: create_dataset returned empty or None data.")
                continue

            # Run model prediction (multimodal_model expects a list of inputs)
            # The model outputs a single probability for the positive class (planet)
            mean_confidence_raw = multimodal_model.predict([X_img, X_ts, X_features])[0][0]
            
            # Get period from all_pipeline_results
            period = all_pipeline_results[0].get('period', np.nan)
            
            # Get CACL explanation for max_disagreement
            # explain_with_cacl expects a single data_sample (time-series)
            cacl_explanation = explain_with_cacl(
                data_sample=X_ts[0], # Assuming first sample if batch size > 1
                cacl_explainer=cacl_explainer,
                dependency_matrix=None, # Not pre-computed for survey
                context_avg_embeddings=None, # Not pre-computed for survey
            )
            max_disagreement = cacl_explanation.get('max_disagreement', 0.0)

            # Derive prob_1d and prob_2d (conceptual)
            prob_1d = min(1.0, mean_confidence_raw + max_disagreement / 2)
            prob_2d = max(0.0, mean_confidence_raw - max_disagreement / 2)

            survey_results.append({
                'target_id': target_id,
                'true_label_str': true_label_str,
                'period': period,
                'mean_confidence': mean_confidence_raw,
                'max_disagreement': max_disagreement,
                'prob_1d': prob_1d,
                'prob_2d': prob_2d
            })
            processed_count += 1

            # Update candidate/rejection counts (using a simple threshold for summary)
            if mean_confidence_raw > config.DEFAULT_THRESHOLD:
                candidate_count += 1
            else:
                rejection_count += 1

        except Exception as e:
            logger.error(f"Error processing target {target_id} from {file_path_str}: {e}", exc_info=True)
            # Continue to the next target

        # 4. Save Checkpoints
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            partial_df = pd.DataFrame(survey_results)
            checkpoint_file = SURVEY_RESULTS_DIR / f"survey_results_partial_{i+1}.csv"
            partial_df.to_csv(checkpoint_file, index=False)
            logger.info(f"Checkpoint saved: {checkpoint_file} ({len(survey_results)} items processed).")

    # 5. Final Output
    final_df = pd.DataFrame(survey_results)
    final_output_file = SURVEY_RESULTS_DIR / "survey_results_final.csv"
    final_df.to_csv(final_output_file, index=False)
    logger.info(f"Final survey results saved to: {final_output_file}")

    logger.info(f"Survey complete: Processed {processed_count} targets.")
    logger.info(f"Summary: Found {candidate_count} Candidates (mean_confidence > {config.DEFAULT_THRESHOLD}), {rejection_count} Rejections.")
    logger.info(f"Total targets in catalog: {len(all_targets)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a full-scale survey for exoplanet detection using a multimodal model.")
    parser.add_argument('--planets_dir', type=str, 
                        default=str(config.CONFIRMED_PLANETS_DIR),
                        help='Path to the directory containing confirmed planets light curve files.')
    parser.add_argument('--false_positives_dir', type=str, 
                        default=str(config.FALSE_POSITIVES_DIR),
                        help='Path to the directory containing false positives light curve files.')
    args = parser.parse_args()

    run_full_scale_survey(args.planets_dir, args.false_positives_dir)
