import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
import json

# Ensure the project root is in the sys.path for importing config
# This assumes generate_ece.py is in paper_figures/ or a similar subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # Adjust based on actual depth of generate_ece.py

import config # Import config for constants

import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Paths ---
BASE_DIR = Path(__file__).resolve().parents[1] # Assumes script is in paper_figures/
# Use the results directory from config for the latest run
LATEST_RUN_DIR = config.RESULTS_DIR / "run_20260215-183146" # Specific run ID from memory
PAPER_FIGURES_DIR = BASE_DIR / "paper_figures" # Output directory for figures

# Ensure output directory exists
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- ECE Calculation Function ---
def ece_score(y_true, y_pred_confidences, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    
    Args:
        y_true (np.array): True labels (0 or 1).
        y_pred_confidences (np.array): Predicted probabilities/confidences for the positive class.
        n_bins (int): Number of bins for calibration curve.

    Returns:
        float: The ECE score.
    """
    if len(y_true) == 0:
        return np.nan # Cannot calculate ECE for empty dataset

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Filter samples into current bin
        in_bin = (y_pred_confidences > bin_lower) & (y_pred_confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Calculate accuracy and average confidence for the bin
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_pred_confidences[in_bin] > 0.5))
            avg_confidence_in_bin = np.mean(y_pred_confidences[in_bin])
            
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece

# --- Main Script ---
def main():
    logger.info("Starting ECE calculation and figure generation.")

    # 1. Load Data
    logger.info(f"Loading data from: {LATEST_RUN_DIR}")
    try:
        y_true = np.load(LATEST_RUN_DIR / 'y_val_actual.npy')
        y_pred_raw = np.load(LATEST_RUN_DIR / 'y_pred_val_raw.npy')
        with open(LATEST_RUN_DIR / 'all_pipeline_results_val.json', 'r') as f:
            all_pipeline_results_val = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}. Ensure the pipeline has been run successfully.")
        return

    # Extract periods and true_labels from all_pipeline_results_val and ensure they are aligned
    # We need to reconstruct a DataFrame that combines predictions, true labels, and periods
    # The original indices are crucial here if the order was not preserved exactly
    
    # Create a list of dictionaries with relevant info from all_pipeline_results_val
    data_for_df = []
    # all_pipeline_results_val is a list of dicts for each validated sample
    for i, res in enumerate(all_pipeline_results_val):
        # The 'periodicity' key holds the median_period
        # The true_label is not directly in res, but y_true matches y_pred_raw by index
        data_for_df.append({
            'period': res.get('periodicity', np.nan),
            'true_label': y_true[i], # y_true and y_pred_raw are aligned with this list
            'p_joint': y_pred_raw[i][0]
        })
    
    df_eval = pd.DataFrame(data_for_df)
    
    # Ensure all necessary columns are numeric and handle NaNs
    df_eval['period'] = pd.to_numeric(df_eval['period'], errors='coerce')
    df_eval['p_joint'] = pd.to_numeric(df_eval['p_joint'], errors='coerce')
    df_eval['true_label'] = pd.to_numeric(df_eval['true_label'], errors='coerce') # True labels (0 or 1)
    
    logger.info(f"NaNs in 'period' before dropna: {df_eval['period'].isnull().sum()}")
    logger.info(f"NaNs in 'p_joint' before dropna: {df_eval['p_joint'].isnull().sum()}")
    logger.info(f"NaNs in 'true_label' before dropna: {df_eval['true_label'].isnull().sum()}")

    df_eval.dropna(subset=['period', 'p_joint', 'true_label'], inplace=True)

    if df_eval.empty:
        logger.error("No valid data for ECE calculation after loading and cleaning. Aborting.")
        return

    logger.info(f"Total valid samples for ECE calculation: {len(df_eval)}")

    # 2. Define Orbital Regimes
    def get_regime(p):
        if p < 20: return "Short (<20d)"
        elif 20 <= p <= 100: return "Medium (20-100d)"
        else: return "Long (>100d)"

    df_eval['regime'] = df_eval['period'].apply(get_regime)

    # 3. Calculate ECE for each regime
    ece_results = {}
    for regime in ["Short (<20d)", "Medium (20-100d)", "Long (>100d)"]:
        regime_df = df_eval[df_eval['regime'] == regime]
        if not regime_df.empty:
            ece_val = ece_score(regime_df['true_label'].values, regime_df['p_joint'].values)
            ece_results[regime] = ece_val
            logger.info(f"ECE for {regime}: {ece_val:.4f} ({len(regime_df)} samples)")
        else:
            ece_results[regime] = np.nan
            logger.warning(f"No samples found for {regime}. ECE set to NaN.")

    # 4. Generate Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Define custom color palette
    colors = ["#1f77b4", "#17becf", "#98df8a"] # Navy Blue, Teal, Light Green
    # Using a list of tuples for consistency: (regime, ece_value)
    ece_data = pd.DataFrame(list(ece_results.items()), columns=['Regime', 'ECE'])
    
    # Ensure correct order for plotting
    order = ["Short (<20d)", "Medium (20-100d)", "Long (>100d)"]
    ece_data['Regime'] = pd.Categorical(ece_data['Regime'], categories=order, ordered=True)

    sns.barplot(x='Regime', y='ECE', data=ece_data, palette=colors, ax=plt.gca())

    plt.title("Expected Calibration Error by Orbital Period Regime", fontsize=14)
    plt.ylabel("Expected Calibration Error (ECE)", fontsize=12)
    plt.xlabel("Orbital Period Regime", fontsize=12)
    plt.ylim(0, ece_data['ECE'].max() * 1.2 if not ece_data['ECE'].isnull().all() else 1) # Dynamic y-limit
    plt.tight_layout()
    
    figure_path = PAPER_FIGURES_DIR / "figure5_ece.png"
    plt.savefig(figure_path, dpi=300)
    logger.info(f"Figure 5 saved to {figure_path}")
    plt.close()

    logger.info("ECE calculation and figure generation completed.")

if __name__ == '__main__':
    main()
