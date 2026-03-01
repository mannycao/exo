import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # Corrected parents index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import config # Import config for constants

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration Paths ---
# Assuming the script is run from the project root or paths are handled externally
BASE_DIR = Path(__file__).resolve().parent.parent # /Users/emmanuel/proj/phd
DATA_FILES_DIR = BASE_DIR / "data_files"
PAPER2_EXPERIMENTS_DIR = BASE_DIR / "paper2_experiments"

SURVEY_RESULTS_FILE = PAPER2_EXPERIMENTS_DIR / "survey_results_final.csv"
FULL_METADATA_PATH = DATA_FILES_DIR / "full_metadata_v2.csv"

FIGURE1_PATH = PAPER2_EXPERIMENTS_DIR / "Figure1_TheCliff.png"
FIGURE2_PATH = PAPER2_EXPERIMENTS_DIR / "Figure2_TheVeto.png"

# --- Figure Parameters ---
SHORT_PERIOD_THRESHOLD = 20
MEDIUM_PERIOD_THRESHOLD = 100
P_FUSED_THRESHOLD = 0.5
NUM_FP_FOR_VETO = 21 # Approximately 21 hard false positives

def load_and_clean_data():
    """
    Loads and cleans the necessary data for figure generation.
    Returns a cleaned DataFrame.
    """
    logger.info("Loading and cleaning data...")

    # Load the single combined survey results file
    df = pd.read_csv(SURVEY_RESULTS_FILE)
    df['target_id'] = df['target_id'].astype(str) # Ensure target_id is string

    # Ensure columns are numeric and handle NaNs
    df['disagreement'] = pd.to_numeric(df['disagreement'], errors='coerce')
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    df['p_joint'] = pd.to_numeric(df['p_joint'], errors='coerce') # Ensure p_joint is numeric
    df['true_label'] = df['true_label'].astype(str) # Ensure true_label is string

    # Drop rows where essential data is missing
    df.dropna(subset=['disagreement', 'period', 'true_label', 'p_joint'], inplace=True)

    logger.info(f"Loaded and cleaned data. Total {len(df)} entries.")
    return df

def generate_reliability_cliff_figure(df):
    """
    Generates Figure 1: The Reliability Cliff (Bar Chart).
    Y-Axis: Mean Disagreement, X-Axis: Regime.
    """
    logger.info("Generating Figure 1: The Reliability Cliff...")

    # Define Regimes
    def get_period_regime(period):
        if period < SHORT_PERIOD_THRESHOLD:
            return "Short (<20d)"
        elif SHORT_PERIOD_THRESHOLD <= period <= MEDIUM_PERIOD_THRESHOLD:
            return "Medium (20-100d)"
        else:
            return "Long (>100d)"

    df['regime'] = df['period'].apply(get_period_regime)

    # Ensure order for plotting
    regime_order = ["Short (<20d)", "Medium (20-100d)", "Long (>100d)"]
    df['regime'] = pd.Categorical(df['regime'], categories=regime_order, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='regime', y='disagreement', data=df, errorbar='ci', capsize=0.1, palette='viridis')

    plt.title('Figure 1: Mean Disagreement by Period Regime (The Reliability Cliff)')
    plt.xlabel('Orbital Period Regime')
    plt.ylabel('Mean Disagreement (abs(p_1d - p_2d))')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotation: Median disagreement for Medium regime
    median_disagreement_medium = df[df['regime'] == "Medium (20-100d)"]['disagreement'].median()
    plt.text(1, # X-coordinate (index of Medium bar)
             plt.gca().get_ylim()[1] * 0.9, # Y-coordinate (near top of plot)
             f"Median $\\delta \\approx {median_disagreement_medium:.2f}$", 
             ha='center', va='top', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig(FIGURE1_PATH)
    logger.info(f"Figure 1 saved to {FIGURE1_PATH}")
    plt.close()

def generate_disagreement_veto_figure(df):
    """
    Generates Figure 2: The Disagreement Veto (Histogram).
    Filter for High Confidence Candidates (P_fused > 0.5).
    Plot two overlapping histograms for Verified Candidates and Hard False Positives.
    """
    logger.info("Generating Figure 2: The Disagreement Veto...")

    # Filter for High Confidence Candidates (p_joint > 0.5)
    high_confidence_df = df[df['p_joint'] > P_FUSED_THRESHOLD]
    logger.info(f"High confidence targets (p_joint > {P_FUSED_THRESHOLD}): {len(high_confidence_df)} entries.")

    # Class A: Verified Candidates
    class_a_df = high_confidence_df[high_confidence_df['true_label'] == config.FILE_TYPE_CONFIRMED_PLANET]
    
    # Class B: Verified False Positives (the "Hard" ones)
    # The prompt mentions N approx 21, these would be high confidence FP, so we filter by true_label and p_joint.
    class_b_df = high_confidence_df[high_confidence_df['true_label'] == config.FILE_TYPE_FALSE_POSITIVE]
    
    logger.info(f"Class A (Verified Candidates) count: {len(class_a_df)}")
    logger.info(f"Class B (Verified False Positives) count: {len(class_b_df)}")
    
    # Ensure there's data to plot
    if class_a_df.empty and class_b_df.empty:
        logger.warning("No data for high confidence candidates or false positives to plot for Figure 2. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=class_a_df, 
        x='disagreement', 
        stat='density', 
        kde=True, 
        color='skyblue', 
        label='Verified Candidates', 
        alpha=0.6,
        bins=np.linspace(0, 1, 50) # Disagreement score from 0 to 1
    )
    sns.histplot(
        data=class_b_df, 
        x='disagreement', 
        stat='density', 
        kde=True, 
        color='red', 
        label='Verified False Positives', 
        alpha=0.6,
        bins=np.linspace(0, 1, 50)
    )

    plt.title('Figure 2: Disagreement Score Distribution for High Confidence Targets (The Disagreement Veto)')
    plt.xlabel('Disagreement Score (abs(p_1d - p_2d))')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 1) # Disagreement score is between 0 and 1

    plt.tight_layout()
    plt.savefig(FIGURE2_PATH)
    logger.info(f"Figure 2 saved to {FIGURE2_PATH}")
    plt.close()

def main():
    """Main function to run the figure generation workflow."""
    logger.info("Starting publication figures generation.")
    
    # Ensure output directory exists
    PAPER2_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()

    if df.empty:
        logger.error("No data available after loading and cleaning. Cannot generate figures.")
        return

    generate_reliability_cliff_figure(df.copy()) # Pass a copy to avoid modifying original df for next figure
    generate_disagreement_veto_figure(df.copy())

    logger.info("Publication figures generation completed.")

if __name__ == '__main__':
    main()
