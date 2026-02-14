import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
from pathlib import Path
import os
import re
import tensorflow as tf # Import tensorflow

# Ensure the project root is in the sys.path for importing config and other modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.dataset_generator import process_single_file # Corrected import for image generation
# Removed: from models.bayesian_predictor import BayesianPredictor # Not directly used for isolated 2D prob
from models.multimodal_model import build_multimodal_fusion_model

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_tess_image(tic_id: int):
    """
    Diagnoses why a specific TESS image (TIC ID) might be failing the 2D model.
    """
    logger.info(f"Starting diagnosis for TIC ID: {tic_id}")

    # 1. Find the file path for the problem target
    file_path = None
    # We specifically look in confirmed_planets because that's where the TIC ID 48503881 was found.
    search_dirs = [config.CONFIRMED_PLANETS_DIR] # Restrict search to improve performance/accuracy

    for base_dir in search_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                # This regex needs to match how target_id is extracted in run_full_survey.py
                match_tess = re.search(r'tess[0-9]+-s[0-9]+-([0-9]+)-', file)
                if match_tess and int(match_tess.group(1)) == tic_id:
                    file_path = Path(root) / file
                    logger.info(f"Found file for TIC ID {tic_id}: {file_path}")
                    break
            if file_path:
                break
        if file_path:
            break

    if not file_path:
        logger.error(f"Could not find light curve file for TIC ID: {tic_id}. Exiting.")
        return

    # 2. Call process_single_file from data.dataset_generator to get the 2D image
    file_info_dict = {'file_path': file_path, 'type': 'CANDIDATE'} # 'type' is needed but value doesn't matter for image gen

    try:
        # process_single_file returns: segment, img_norm, feature_vector, label, transit_info, periodicity_data,
        #                             planet_properties, plot_path, provenance, period_from_header, time_lc, flux_lc
        processing_results = process_single_file(file_info_dict, config.IMAGE_SIZE, config.FIXED_LENGTH)
        
        if processing_results is None:
            logger.error(f"process_single_file failed for TIC ID: {tic_id}. Exiting.")
            return
        
        # img_norm (the 2D image) is the second element in the returned tuple
        secondary_view = processing_results[1] # This will be (IMAGE_SIZE_H, IMAGE_SIZE_W)
        
        # Ensure it has a channel dimension of 1 for model input if needed later, but squeeze for inspection/plotting
        secondary_view_squeezed = secondary_view.squeeze()

    except Exception as e:
        logger.error(f"Error during image generation for TIC ID {tic_id}: {e}", exc_info=True)
        return

    # 3. Inspect the Array
    logger.info(f"--- Image Array Inspection for TIC ID: {tic_id} ---")
    logger.info(f"Shape: {secondary_view_squeezed.shape}")
    logger.info(f"Min: {np.min(secondary_view_squeezed)}")
    logger.info(f"Max: {np.max(secondary_view_squeezed)}")
    logger.info(f"Mean: {np.mean(secondary_view_squeezed)}")
    logger.info(f"Standard Deviation: {np.std(secondary_view_squeezed)}")

    if np.max(secondary_view_squeezed) == 0:
        logger.error("FAIL: Image is pure zeros.")
    # Assuming normalization to a range like 0-1 or similar. Max > 100 is a heuristic for unnormalized raw flux.
    if np.max(secondary_view_squeezed) > 100: 
        logger.warning("FAIL: Image might not be normalized (Max value > 100). Check normalization steps in dataset_generator.py.")
    
    # 4. Visualize
    output_image_path = Path(__file__).resolve().parent / f"debug_tess_view_{tic_id}.png"
    plt.figure(figsize=(config.IMAGE_SIZE[0]/10, config.IMAGE_SIZE[1]/10))
    plt.imshow(secondary_view_squeezed, cmap='viridis', origin='lower')
    plt.title(f"TESS Image for TIC ID: {tic_id}")
    plt.colorbar(label='Normalized Flux')
    plt.xlabel("Phase Bin")
    plt.ylabel("Depth Bin")
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    logger.info(f"Debug image saved to: {output_image_path}")

    # 5. Test Inference
    image_shape = config.IMAGE_SIZE + (1,)
    timeseries_shape = (config.FIXED_LENGTH, 1)
    feature_shape = (config.FEATURE_VECTOR_LENGTH,)

    multimodal_model_full = build_multimodal_fusion_model(
        image_shape=image_shape,
        timeseries_shape=timeseries_shape,
        feature_shape=feature_shape
    )
    if config.MODEL_PATH.exists():
        multimodal_model_full.load_weights(config.MODEL_PATH)
        logger.info(f"Full multimodal model loaded for inference testing from {config.MODEL_PATH}")
    else:
        logger.error(f"Full multimodal model weights not found at {config.MODEL_PATH}. Cannot test inference. Exiting.")
        return

    # Prepare dummy inputs for timeseries and features to isolate the 2D branch's effect
    dummy_ts_input = np.zeros((1,) + timeseries_shape)
    dummy_feature_input = np.zeros((1,) + feature_shape)

    # Reshape secondary_view to be a batch of 1 image for model prediction
    # process_single_file returns (H, W), model expects (batch, H, W, channels)
    image_input_for_model = np.expand_dims(secondary_view_squeezed, axis=0) # (H, W) -> (1, H, W)
    image_input_for_model = np.expand_dims(image_input_for_model, axis=-1) # (1, H, W) -> (1, H, W, 1)

    try:
        # Use the full multimodal model to get the raw confidence for this image, isolating its effect.
        # This will be the `mean_confidence_raw` if other inputs were zeroed out.
        raw_output_from_model = multimodal_model_full.predict(
            [image_input_for_model, dummy_ts_input, dummy_feature_input]
        )
        prob_2d_raw_isolated = raw_output_from_model[0][0] # Assuming single output scalar

        logger.info(f"Raw prob_2d (isolated from 2D input) for TIC ID {tic_id}: {prob_2d_raw_isolated}")

    except Exception as e:
        logger.error(f"Error during 2D isolated inference for TIC ID {tic_id}: {e}", exc_info=True)
        return

if __name__ == "__main__":
    PROBLEM_TIC_ID = 48503881
    diagnose_tess_image(PROBLEM_TIC_ID)
