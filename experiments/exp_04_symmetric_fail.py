# experiments/exp_04_symmetric_fail.py

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import sys
from pathlib import Path
import cv2 # For Gaussian blur
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the project root is in the sys.path for importing models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.eo_sar_model import build_multimodal_fusion_model

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EO_SHAPE = (224, 224, 3) # Optical RGB
SAR_SHAPE = (224, 224, 1) # SAR Grayscale
K_SIZE_LEVELS = [1, 3, 5, 7, 9, 11, 13, 15] # Odd integers for Gaussian blur kernel size
NUM_SAMPLES_PER_K_SIZE = 5 # Number of dummy samples to average results over

# --- Function to apply symmetric Gaussian blur ---
def symmetric_blur(eo_img, sar_img, k_size):
    """
    Applies Gaussian blur to both EO and SAR images.
    
    Args:
        eo_img (np.array): The original EO image (H, W, 3).
        sar_img (np.array): The original SAR image (H, W, 1).
        k_size (int): Kernel size for Gaussian blur. Must be an odd integer.
                         
    Returns:
        tuple: (blurred_eo_img, blurred_sar_img)
    """
    if k_size % 2 == 0:
        raise ValueError("k_size for Gaussian blur must be an odd integer.")

    # Apply Gaussian blur to EO image
    blurred_eo = cv2.GaussianBlur(eo_img, (k_size, k_size), 0)
    
    # Apply Gaussian blur to SAR image
    # For 1-channel images, OpenCV's GaussianBlur works directly.
    blurred_sar = cv2.GaussianBlur(sar_img, (k_size, k_size), 0)
    
    return blurred_eo, blurred_sar

# --- Main Experiment ---
def main():
    logger.info("Starting Experiment 04: Symmetric Failure Analysis.")

    # 1. Load the model
    logger.info("Building the EO/SAR model to extract fused confidence and branch predictions.")
    model = build_multimodal_fusion_model(EO_SHAPE, SAR_SHAPE, return_branches=True)
    
    # Compile the model (needed for predict method)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    results = []

    # Generate one set of clean dummy data for EO and SAR
    clean_eo_sample = np.random.rand(1, *EO_SHAPE).astype(np.float32)
    clean_sar_sample = np.random.rand(1, *SAR_SHAPE).astype(np.float32)

    # 2. Run a loop with increasing k_size levels
    for k_size in K_SIZE_LEVELS:
        logger.info(f"Processing blur kernel size: {k_size}")
        
        fused_confidences = []
        disagreements = []

        for _ in range(NUM_SAMPLES_PER_K_SIZE):
            # Degrad both EO and SAR images
            # Note: symmetric_blur expects (H,W,C) or (H,W), so pass [0] index
            degraded_eo_sample, degraded_sar_sample = symmetric_blur(clean_eo_sample[0], clean_sar_sample[0], k_size)
            
            # The model expects a batch, so expand dimensions
            degraded_eo_batch = np.expand_dims(degraded_eo_sample, axis=0)
            degraded_sar_batch = np.expand_dims(degraded_sar_sample, axis=0)

            # Pass the degraded images to the model
            # Model outputs are [main_output, eo_branch_output, sar_branch_output]
            fused_output, prob_eo, prob_sar = model.predict(
                {'eo_input': degraded_eo_batch, 'sar_input': degraded_sar_batch},
                verbose=0 # Suppress progress bar
            )
            
            # Extract scalar probabilities
            fused_confidence = fused_output[0][0]
            prob_eo_scalar = prob_eo[0][0]
            prob_sar_scalar = prob_sar[0][0]

            # Calculate disagreement
            disagreement = abs(prob_eo_scalar - prob_sar_scalar)
            
            fused_confidences.append(fused_confidence)
            disagreements.append(disagreement)
        
        # Calculate average for this k_size level
        avg_fused_confidence = np.mean(fused_confidences)
        avg_disagreement = np.mean(disagreements)
        
        results.append({
            'K_Size': k_size, 
            'Average Fused Confidence': avg_fused_confidence, 
            'Average Disagreement': avg_disagreement
        })

    # Print a table of results
    results_df = pd.DataFrame(results)
    logger.info("--- Symmetric Failure Experiment Results ---")
    logger.info(results_df.to_string(index=False))

    # 3. Plot Fused_Confidence vs. Disagreement
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Use seaborn scatterplot with hue for k_size to show progression
    sns.scatterplot(
        x='Average Disagreement', 
        y='Average Fused Confidence', 
        hue='K_Size', 
        size='K_Size', # Vary size by k_size for better visualization
        sizes=(50, 400), # Range of marker sizes
        data=results_df, 
        palette='viridis', 
        legend='full'
    )
    
    plt.title("Symmetric Degradation: Fused Confidence vs. Disagreement", fontsize=14)
    plt.xlabel("Average Disagreement", fontsize=12)
    plt.ylabel("Average Fused Confidence", fontsize=12)
    plt.xlim(0, max(results_df['Average Disagreement'].max() * 1.1, 0.1))
    plt.ylim(0, max(results_df['Average Fused Confidence'].max() * 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    figure_path = Path("paper_figures") / "figure_exp04_symmetric_fail.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    logger.info(f"Plot saved to {figure_path}")
    plt.close()

    logger.info("Experiment 04 completed.")

if __name__ == "__main__":
    main()
