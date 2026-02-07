import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path

# Add the parent directory to the sys.path to import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# --- Configuration ---
LATEST_RUN_DIR = "/Users/emmanuel/proj/phd/results/run_20260125-144401" # Hardcoded based on previous step
DEFAULT_THRESHOLD = 0.8
DISAGREEMENT_FILTER_THRESHOLD = 1.0

def calculate_metrics(y_true, y_pred_proba, threshold=DEFAULT_THRESHOLD):
    """Calculates evaluation metrics given true labels and predicted probabilities."""
    y_pred_binary = (y_pred_proba > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'FPR': fpr,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }

def run_catalog_reliability_experiment():
    print(f"Running catalog reliability experiment using data from: {LATEST_RUN_DIR}\n")

    # 1. Load Data
    try:
        y_val_actual = np.load(os.path.join(LATEST_RUN_DIR, 'y_val_actual.npy'))
        y_pred_val_raw = np.load(os.path.join(LATEST_RUN_DIR, 'y_pred_val_raw.npy'))
        with open(os.path.join(LATEST_RUN_DIR, 'cacl_explanations.json'), 'r') as f:
            cacl_explanations = json.load(f)
        with open(os.path.join(LATEST_RUN_DIR, 'all_pipeline_results_val.json'), 'r') as f:
            all_pipeline_results_val = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure all required files exist in {LATEST_RUN_DIR}")
        return

    # 2. Data Structuring
    data_list = []
    # cacl_explanations keys are original_index as strings
    
    # Need to correctly map cacl_explanations to y_val_actual and y_pred_val_raw
    # all_pipeline_results_val contains 'original_index' which maps to the keys in cacl_explanations
    
    # Create a mapping from original_index to the index in y_val_actual/y_pred_val_raw
    original_idx_to_val_idx = {str(res['original_index']): i for i, res in enumerate(all_pipeline_results_val)}

    for res in all_pipeline_results_val:
        original_idx = str(res['original_index'])
        val_idx = original_idx_to_val_idx.get(original_idx)

        if val_idx is None:
            print(f"Warning: original_index {original_idx} not found in y_val_actual/y_pred_val_raw mapping. Skipping.")
            continue
        
        # Ensure that y_val_actual and y_pred_val_raw have valid indices
        if val_idx >= len(y_val_actual) or val_idx >= len(y_pred_val_raw):
            print(f"Warning: val_idx {val_idx} out of bounds for y_val_actual/y_pred_val_raw. Skipping.")
            continue

        true_label = y_val_actual[val_idx]
        mean_confidence = y_pred_val_raw[val_idx]

        # Get CACL explanation for this original_index
        explanation = cacl_explanations.get(original_idx)
        if not explanation:
            print(f"Warning: CACL explanation not found for original_index {original_idx}. Skipping.")
            continue
        max_disagreement = explanation.get('max_disagreement', 0.0)

        data_list.append({
            'original_index': original_idx,
            'true_label': int(true_label), # Ensure int
            'mean_confidence': float(mean_confidence), # Ensure float
            'period': res.get('period', np.nan),
            'max_disagreement': float(max_disagreement) # Ensure float
        })

    df = pd.DataFrame(data_list)
    df['true_label_str'] = df['true_label'].map({1: 'CONFIRMED', 0: 'FALSE POSITIVE'})

    # 3. Derive prob_1d and prob_2d (conceptual)
    df['prob_1d'] = df.apply(lambda row: min(1.0, row['mean_confidence'] + row['max_disagreement'] / 2), axis=1)
    df['prob_2d'] = df.apply(lambda row: max(0.0, row['mean_confidence'] - row['max_disagreement'] / 2), axis=1)
    # Ensure consistency: abs(p1 - p2) should equal disagreement after clamping
    df['calculated_disagreement'] = np.abs(df['prob_1d'] - df['prob_2d'])


    # --- Experiment 1 (Baseline) ---
    print("\n--- Experiment 1: Baseline Metrics (mean_confidence > 0.8) ---")
    baseline_metrics = calculate_metrics(df['true_label'], df['mean_confidence'], threshold=DEFAULT_THRESHOLD)
    df_table1 = pd.DataFrame([baseline_metrics]).round(4)
    print(df_table1.to_markdown(index=False))
    df_table1.to_csv(os.path.join("paper2_experiments", "Table1_Baseline.csv"), index=False)


    # --- Experiment 2 (Stratification) ---
    print("\n--- Experiment 2: Disagreement Score Stratification ---")
    true_planets_disagreement = df[df['true_label'] == 1]['max_disagreement']
    true_false_positives_disagreement = df[df['true_label'] == 0]['max_disagreement']

    table2_data = {
        'Category': ['True Planets', 'True False Positives'],
        'Mean Disagreement': [true_planets_disagreement.mean(), true_false_positives_disagreement.mean()],
        'Std Disagreement': [true_planets_disagreement.std(), true_false_positives_disagreement.std()],
        'Min Disagreement': [true_planets_disagreement.min(), true_false_positives_disagreement.min()],
        'Max Disagreement': [true_planets_disagreement.max(), true_false_positives_disagreement.max()],
    }
    df_table2 = pd.DataFrame(table2_data).round(4)
    print(df_table2.to_markdown(index=False))
    df_table2.to_csv(os.path.join("paper2_experiments", "Table2_Stratification.csv"), index=False)


    # --- Experiment 3 (Contamination Reduction - The Money Table) ---
    print("\n--- Experiment 3: Contamination Reduction (Disagreement Filter) ---")
    
    # Metrics before filtering (using the same threshold for consistency)
    initial_metrics = calculate_metrics(df['true_label'], df['mean_confidence'], threshold=DEFAULT_THRESHOLD)
    fp_initial = initial_metrics['FP']
    tp_initial = initial_metrics['TP']
    
    filtered_df = df[df['max_disagreement'] <= DISAGREEMENT_FILTER_THRESHOLD].copy()

    if filtered_df.empty:
        print("No samples remaining after disagreement filter. Skipping Experiment 3 detailed metrics.")
        df_table3 = pd.DataFrame([{
            'Filter Threshold': DISAGREEMENT_FILTER_THRESHOLD,
            'Precision (Filtered)': np.nan,
            'Recall (Filtered)': np.nan,
            'FPR (Filtered)': np.nan,
            'FP Reduction %': np.nan,
            'Planet Loss %': np.nan,
            'Remaining Samples': 0
        }])
    else:
        filtered_metrics = calculate_metrics(filtered_df['true_label'], filtered_df['mean_confidence'], threshold=DEFAULT_THRESHOLD)
        fp_filtered = filtered_metrics['FP']
        tp_filtered = filtered_metrics['TP']

        fp_reduction_percent = (1 - (fp_filtered / fp_initial)) * 100 if fp_initial > 0 else (100 if fp_filtered == 0 else np.nan)
        planet_loss_percent = (1 - (tp_filtered / tp_initial)) * 100 if tp_initial > 0 else (0 if tp_filtered == 0 else np.nan)
        
        table3_data = {
            'Filter Threshold': DISAGREEMENT_FILTER_THRESHOLD,
            'Precision (Filtered)': filtered_metrics['Precision'],
            'Recall (Filtered)': filtered_metrics['Recall'],
            'FPR (Filtered)': filtered_metrics['FPR'],
            'FP Reduction %': fp_reduction_percent,
            'Planet Loss %': planet_loss_percent,
            'Remaining Samples': len(filtered_df)
        }
        df_table3 = pd.DataFrame([table3_data]).round(4)

    print(df_table3.to_markdown(index=False))
    df_table3.to_csv(os.path.join("paper2_experiments", "Table3_Impact.csv"), index=False)

    print("\nExperiment complete. Check 'paper2_experiments/' for CSV results.")

if __name__ == '__main__':
    run_catalog_reliability_experiment()