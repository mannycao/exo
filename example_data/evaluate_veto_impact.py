import pandas as pd
import numpy as np
import os
from statsmodels.stats.proportion import proportion_confint

def load_combined_data():
    base_dir = 'paper2_experiments'
    dataframes = []

    # 1. Kepler Original (primary dataset)
    orig_path = os.path.join(base_dir, 'survey_results_kepler_orig.csv')
    df_orig = pd.read_csv(orig_path).rename(columns={'prob_joint': 'p_joint'})
    df_orig = df_orig[['target_id', 'true_label', 'p_joint', 'disagreement']]
    dataframes.append(df_orig)

    # 2. Final Results (TESS data)
    final_path = os.path.join(base_dir, 'survey_results_final.csv')
    df_final = pd.read_csv(final_path)
    df_final = df_final[['target_id', 'true_label', 'p_joint', 'disagreement']]
    dataframes.append(df_final)

    # 3. Long Keep (fills missing Kepler long periods)
    long_path = os.path.join(base_dir, 'survey_results_long_period_keep.csv')
    df_long = pd.read_csv(long_path).rename(columns={'prob_joint': 'p_joint'})
    df_long = df_long[['target_id', 'true_label', 'p_joint', 'disagreement']]
    dataframes.append(df_long)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['true_label'] = combined_df['true_label'].replace({
        'confirmed_planet': 'confirmed_planet',
        'CANDIDATE': 'confirmed_planet',
        'false_positive': 'false_positive'
    })
    return combined_df

def evaluate_veto():
    df = load_combined_data()

    # Note: Using 0.5 as threshold because p_joint > 0.9 is missing in local CSVs
    THRESHOLD = 0.5 
    high_conf_df = df[df['p_joint'] > THRESHOLD].copy()
    
    print(f"Total Predictions with p_joint > {THRESHOLD}: {len(high_conf_df)}")

    if len(high_conf_df) == 0:
        print(f"Error: No samples found with fused probability > {THRESHOLD}.")
        return

    # 3. Split into subsets
    consensus_df = high_conf_df[high_conf_df['disagreement'] < 0.2]
    disagreement_df = high_conf_df[high_conf_df['disagreement'] > 0.3]

    def calculate_fpr_and_ci(subset_df):
        total = len(subset_df)
        if total == 0: return 0.0, (0.0, 0.0), 0
        num_fps = len(subset_df[subset_df['true_label'] == 'false_positive'])
        fpr = num_fps / total
        ci_low, ci_high = proportion_confint(num_fps, total, alpha=0.05, method='wilson')
        return fpr, (ci_low, ci_high), total

    fpr_con, ci_con, total_con = calculate_fpr_and_ci(consensus_df)
    fpr_dis, ci_dis, total_dis = calculate_fpr_and_ci(disagreement_df)

    print("-" * 60)
    print(f"Consensus Subset (delta < 0.2): FPR = {fpr_con:.1%} (95% CI: {ci_con[0]:.1%} - {ci_con[1]:.1%}) [N={total_con}]")
    print(f"Disagreement Subset (delta > 0.3): FPR = {fpr_dis:.1%} (95% CI: {ci_dis[0]:.1%} - {ci_dis[1]:.1%}) [N={total_dis}]")
    print("-" * 60)
    print("Note: Local dataset does not support the p_joint > 0.9 threshold claimed in paper.")

if __name__ == "__main__":
    evaluate_veto()
