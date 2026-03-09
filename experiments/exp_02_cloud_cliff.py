# experiments/exp_02_cloud_cliff.py

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import os
from pathlib import Path
import sys

# Ensure the project root is in the sys.path for importing models
# Assuming this script is in experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.eo_sar_model import build_multimodal_fusion_model

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EO_SHAPE = (224, 224, 3) # Optical RGB
SAR_SHAPE = (224, 224, 1) # SAR Grayscale
OPACITY_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_SAMPLES_PER_OPACITY = 10 # Number of dummy samples to average disagreement over

# --- Function to inject cloud noise ---
def inject_cloud_noise(image, opacity):
    """
    Blends white noise into the RGB image to simulate cloud cover.
    
    Args:
        image (np.array): The original RGB image (H, W, 3).
        opacity (float): Blending factor, 0.0 for no noise, 1.0 for full noise.
                         (Effectively, 1-opacity for original, opacity for noise)
                         
    Returns:
        np.array: Image with injected cloud noise.
    """
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("Opacity must be between 0.0 and 1.0")

    # Generate white noise (random values between 0 and 1)
    noise = np.random.rand(*image.shape).astype(np.float32)
    
    # Simple linear blending: (1 - opacity) * original + opacity * noise
    # This models opacity as how much of the noise is visible
    degraded_image = (1.0 - opacity) * image + opacity * noise
    
    # Ensure values remain in [0, 1] range
    degraded_image = np.clip(degraded_image, 0.0, 1.0)
    
    return degraded_image

# --- Main Experiment ---
def main():
    logger.info("Starting Experiment 02: Cloud Cliff Analysis.")

    # 1. Load the model
    logger.info("Building the EO/SAR model to extract branch predictions.")
    model = build_multimodal_fusion_model(EO_SHAPE, SAR_SHAPE, return_branches=True)
    
    # Compile the model (needed for predict method)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    results = []

    # Generate one set of clean dummy data for SAR and EO
    # We will only degrade the EO image
    clean_eo_sample = np.random.rand(1, *EO_SHAPE).astype(np.float32)
    clean_sar_sample = np.random.rand(1, *SAR_SHAPE).astype(np.float32)

    # 2. Run a loop with opacity levels
    for opacity in OPACITY_LEVELS:
        logger.info(f"Processing opacity level: {opacity:.1f}")
        
        disagreements_at_opacity = []

        for _ in range(NUM_SAMPLES_PER_OPACITY):
            # Degrad the EO image
            degraded_eo_sample = inject_cloud_noise(clean_eo_sample[0], opacity)
            
            # The model expects a batch, so expand dimensions
            degraded_eo_batch = np.expand_dims(degraded_eo_sample, axis=0)
            clean_sar_batch = clean_sar_sample # Already (1, H, W, C)

            # Pass the degraded EO image and clean SAR image to the model
            # Model outputs are [main_output, eo_branch_output, sar_branch_output]
            _, prob_eo, prob_sar = model.predict(
                {'eo_input': degraded_eo_batch, 'sar_input': clean_sar_batch},
                verbose=0 # Suppress progress bar
            )
            
            # Extract scalar probabilities
            prob_eo_scalar = prob_eo[0][0]
            prob_sar_scalar = prob_sar[0][0]

            # Calculate disagreement
            disagreement = abs(prob_eo_scalar - prob_sar_scalar)
            disagreements_at_opacity.append(disagreement)
        
        # Calculate average disagreement for this opacity level
        avg_disagreement = np.mean(disagreements_at_opacity)
        results.append({'Opacity': opacity, 'Average Disagreement': avg_disagreement})

    # 3. Print a table of Opacity vs. Disagreement
    results_df = pd.DataFrame(results)
    logger.info("--- Cloud Cliff Experiment Results ---")
    logger.info(results_df.to_string(index=False))
    logger.info("Experiment 02 completed.")

if __name__ == "__main__":
    main()