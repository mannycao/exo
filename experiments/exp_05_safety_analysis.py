# experiments/exp_05_safety_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import logging
import sys
from pathlib import Path
import pandas as pd

# Ensure the project root is in the sys.path if this script needs to import from it
# Assuming this script is in experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Plotting (MNRAS-like style for Systems Engineering Journal) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.figsize": (3.5, 3.2), # Single column width
    "lines.linewidth": 2.0,
    "text.usetex": False # Set to True if you have local LaTeX installed
})
sns.set_style("whitegrid") # Using seaborn's whitegrid style

# --- Main Experiment ---
def main():
    logger.info("Starting Experiment 05: Safety Analysis.")

    # 1. Simulate two datasets
    np.random.seed(42) # for reproducibility
    N_SAMPLES = 5000 # Number of samples for each state

    # Nominal data: Low disagreement scores (disagreement should be low for nominal operation)
    # Gaussian centered at 0.05, with a small spread
    nominal_disagreement = np.random.normal(loc=0.05, scale=0.03, size=N_SAMPLES)
    nominal_disagreement = np.clip(nominal_disagreement, 0.0, 0.2) # Ensure values are plausible for low disagreement

    # Compromised data: High disagreement scores (disagreement should be high for compromise)
    # Gaussian centered at 0.6, with some spread, clipped to realistic range
    compromised_disagreement = np.random.normal(loc=0.6, scale=0.15, size=N_SAMPLES)
    compromised_disagreement = np.clip(compromised_disagreement, 0.3, 1.0) # Ensure values are plausible for high disagreement

    # Combine into a single dataset for ROC analysis
    disagreement_scores = np.concatenate([nominal_disagreement, compromised_disagreement])
    # True labels: 0 for Nominal, 1 for Compromised
    true_labels = np.concatenate([np.zeros(N_SAMPLES), np.ones(N_SAMPLES)])

    logger.info(f"Simulated {N_SAMPLES} nominal samples (mean disagreement={np.mean(nominal_disagreement):.2f})")
    logger.info(f"Simulated {N_SAMPLES} compromised samples (mean disagreement={np.mean(compromised_disagreement):.2f})")

    # 2. Use sklearn.metrics to calculate the ROC Curve and AUC
    # Note: For ROC curve, higher scores should correspond to the positive class (compromised).
    # Since higher disagreement implies compromise, we can use disagreement_scores directly as "probabilities".
    fpr, tpr, thresholds = roc_curve(true_labels, disagreement_scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC for detecting System Compromise: {roc_auc:.4f}")

    # 3. Calculate the "False Veto Rate"
    # A "veto" occurs when disagreement > 0.3. A "False Veto" occurs when nominal data
    # triggers a veto (i.e., disagreement > 0.3 for nominal data).
    SAFETY_THRESHOLD = 0.3
    false_vetos = nominal_disagreement[nominal_disagreement > SAFETY_THRESHOLD]
    false_veto_rate = len(false_vetos) / N_SAMPLES
    logger.info(f"Safety threshold set at Disagreement > {SAFETY_THRESHOLD:.1f}")
    logger.info(f"Number of False Vetos (nominal data with disagreement > {SAFETY_THRESHOLD:.1f}): {len(false_vetos)}")
    logger.info(f"False Veto Rate: {false_veto_rate:.4f} ({(false_veto_rate * 100):.2f}%)")

    # 4. Generate a matplotlib figure showing the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for System Compromise Detection')
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    figure_path = Path("paper_figures") / "figure_exp05_safety_analysis.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    logger.info(f"ROC Curve plot saved to {figure_path}")
    plt.close()

    logger.info("Experiment 05 completed.")

if __name__ == "__main__":
    main()
