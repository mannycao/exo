# experiments/exp_03_sar_noise.py

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import sys
from pathlib import Path

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
INTENSITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # Standard deviation for Gaussian noise
NUM_SAMPLES_PER_INTENSITY = 10 # Number of dummy samples to average disagreement over

# --- Function to inject speckle noise ---
def inject_speckle_noise(sar_image, intensity):
    """
    Injects multiplicative Gaussian (speckle) noise into the SAR image.
    Noise is generated with mean 1.0 and standard deviation `intensity`.
    
    Args:
        sar_image (np.array): The original SAR image (H, W, 1).
        intensity (float): Standard deviation of the Gaussian noise. 
                           0.0 for no noise, increasing for more noise.
                         
    Returns:
        np.array: Image with injected speckle noise.
    """
    if not (intensity >= 0.0):
        raise ValueError("Intensity must be non-negative.")

    # Generate multiplicative Gaussian noise. Centered around 1.0 to preserve mean intensity.
    noise = np.random.normal(loc=1.0, scale=intensity, size=sar_image.shape).astype(np.float32)
    
    degraded_image = sar_image * noise
    
    # Ensure values remain in a reasonable range (e.g., 0 to 1 for normalized images)
    # The clip range might need adjustment based on SAR image properties.
    # For dummy data between 0-1, clipping to 0-1 is reasonable.
    degraded_image = np.clip(degraded_image, 0.0, 1.0) 
    
    return degraded_image

# --- Main Experiment ---
def main():
    logger.info("Starting Experiment 03: SAR Noise Analysis.")

    # 1. Load the model
    logger.info("Building the EO/SAR model to extract branch predictions.")
    model = build_multimodal_fusion_model(EO_SHAPE, SAR_SHAPE, return_branches=True)
    
    # Compile the model (needed for predict method)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    results = []

    # Generate one set of clean dummy data for EO and SAR
    # We will only degrade the SAR image
    clean_eo_sample = np.random.rand(1, *EO_SHAPE).astype(np.float32)
    clean_sar_sample = np.random.rand(1, *SAR_SHAPE).astype(np.float32)

    # 2. Run a loop with intensity levels
    for intensity in INTENSITY_LEVELS:
        logger.info(f"Processing intensity level: {intensity:.1f}")
        
        disagreements_at_intensity = []

        for _ in range(NUM_SAMPLES_PER_INTENSITY):
            # Degrad the SAR image
            degraded_sar_sample = inject_speckle_noise(clean_sar_sample[0], intensity)
            
            # The model expects a batch, so expand dimensions if needed
            clean_eo_batch = clean_eo_sample # Already (1, H, W, C)
            degraded_sar_batch = np.expand_dims(degraded_sar_sample, axis=0)

            # Pass the clean EO image and degraded SAR image to the model
            # Model outputs are [main_output, eo_branch_output, sar_branch_output]
            _, prob_eo, prob_sar = model.predict(
                {'eo_input': clean_eo_batch, 'sar_input': degraded_sar_batch},
                verbose=0 # Suppress progress bar
            )
            
            # Extract scalar probabilities
            prob_eo_scalar = prob_eo[0][0]
            prob_sar_scalar = prob_sar[0][0]

            # Calculate disagreement
            disagreement = abs(prob_eo_scalar - prob_sar_scalar)
            disagreements_at_intensity.append(disagreement)
        
        # Calculate average disagreement for this intensity level
        avg_disagreement = np.mean(disagreements_at_intensity)
        results.append({'Intensity': intensity, 'Average Disagreement': avg_disagreement})

    # 3. Print a table of Intensity vs. Disagreement
    results_df = pd.DataFrame(results)
    logger.info("--- SAR Noise Experiment Results ---")
    logger.info(results_df.to_string(index=False))
    logger.info("Experiment 03 completed.")

if __name__ == "__main__":
    main()