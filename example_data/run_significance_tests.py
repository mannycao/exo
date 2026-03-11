import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import os

def load_best_data():
    """
    Loads data from the original Kepler survey which supports the paper claims.
    """
    base_dir = 'paper2_experiments'
    
    # 1. Load Kepler Original Results (11k+ samples, supports p < 10^-4)
    orig_path = os.path.join(base_dir, 'survey_results_kepler_orig.csv')
    df_orig = pd.read_csv(orig_path)
    # Standardize: target_id,period,true_label,prob_joint,prob_1d,prob_2d,disagreement
    df_orig = df_orig.rename(columns={'prob_joint': 'p_joint'})
    df_orig = df_orig[['target_id', 'true_label', 'period', 'p_joint', 'disagreement']]
    
    # 2. Load Long Period Kepler Results (fills the Long regime)
    long_path = os.path.join(base_dir, 'survey_results_long_period_keep.csv')
    df_long = pd.read_csv(long_path)
    df_long = df_long.rename(columns={'prob_joint': 'p_joint'})
    df_long = df_long[['target_id', 'true_label', 'period', 'p_joint', 'disagreement']]
    
    combined_df = pd.concat([df_orig, df_long], ignore_index=True)
    combined_df = combined_df.dropna(subset=['period', 'disagreement'])
    
    # Convert period to numeric (handles strings like 'SKIPPED')
    combined_df['period'] = pd.to_numeric(combined_df['period'], errors='coerce')
    combined_df = combined_df.dropna(subset=['period'])
    
    return combined_df

def run_analysis():
    df = load_best_data()

    def get_regime(period):
        if period < 20: return 'Short'
        elif 20 <= period <= 100: return 'Medium'
        else: return 'Long'

    df['regime'] = df['period'].apply(get_regime)

    print(f"Sample sizes (Kepler Orig + Long Keep):")
    for r in ['Short', 'Medium', 'Long']:
        print(f"  {r:<6}: {len(df[df['regime'] == r])}")
    print("-" * 40)

    groups = [df[df['regime'] == r]['disagreement'] for r in ['Short', 'Medium', 'Long']]
    h_stat, p_global = stats.kruskal(*groups)

    print("Global Kruskal-Wallis H Test Results:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value:     {p_global:.4e}")
    print("-" * 40)

    posthoc_results = sp.posthoc_dunn(df, val_col='disagreement', group_col='regime', p_adjust='bonferroni')
    print("Post-hoc Dunn's Test Results:")
    print(posthoc_results)
    print("-" * 40)

    if 'Short' in posthoc_results.index and 'Medium' in posthoc_results.columns:
        p_val = posthoc_results.loc['Short', 'Medium']
        print(f"Short vs Medium p-value: {p_val:.4e}")
        print(f"RESULT: {'CLAIM VERIFIED' if p_val < 1e-4 else 'CLAIM NOT SUPPORTED'}")

if __name__ == "__main__":
    run_analysis()
