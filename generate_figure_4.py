
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

# --- 1. Load Data ---
run_dir = '/Users/emmanuel/proj/phd/results/run_20251230-084051'
y_true = np.load(f'{run_dir}/y_val_actual.npy')
y_pred_raw = np.load(f'{run_dir}/y_pred_val_raw.npy').flatten()

with open(f'{run_dir}/cacl_explanations.json', 'r') as f:
    cacl_explanations = json.load(f)

# --- 2. Prepare Data ---
# Baseline: Overconfident probabilities (raw model output)
y_prob_baseline = y_pred_raw

# CAC Set: Well-calibrated probabilities (CACL confidence scores)
# Ensure the order of cacl scores matches the y_true labels
with open(f'{run_dir}/all_pipeline_results_val.json', 'r') as f:
    all_pipeline_results_val = json.load(f)

original_indices = [result['original_index'] for result in all_pipeline_results_val]
cacl_confidence_scores = [cacl_explanations[str(i)]['confidence_score'] for i in original_indices]
y_prob_cac = np.array(cacl_confidence_scores)


# --- 3. Calculate Calibration Curves and ECE ---
# Baseline
fraction_of_positives_base, mean_predicted_value_base = calibration_curve(y_true, y_prob_baseline, n_bins=20)
ece_base = expected_calibration_error(y_true, y_prob_baseline)

# CAC
fraction_of_positives_cac, mean_predicted_value_cac = calibration_curve(y_true, y_prob_cac, n_bins=20)
ece_cac = expected_calibration_error(y_true, y_prob_cac)


# --- 4. Plot Reliability Diagram ---
plt.style.use('default')
fig, ax = plt.subplots(figsize=(8, 8))

# Perfect calibration line
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# Baseline curve
ax.plot(mean_predicted_value_base, fraction_of_positives_base, "s-", label=f'Baseline (ECE = {ece_base:.4f})', color='tab:blue')

# CAC curve
ax.plot(mean_predicted_value_cac, fraction_of_positives_cac, "o-", label=f'CAC (Validated) (ECE = {ece_cac:.4f})', color='tab:orange')

ax.set_xlabel('Mean Predicted Confidence')
ax.set_ylabel('Fraction of Positives')
ax.set_ylim([-0.05, 1.05])
ax.legend(loc="lower right")
ax.set_title('Reliability Diagram for run 20251230-084051')

plt.tight_layout()
plt.savefig('Fig4_Reliability_20251230-084051.png', dpi=300)

print("Figure saved as Fig4_Reliability_20251230-084051.png")
