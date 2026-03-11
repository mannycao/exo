import pandas as pd
import numpy as np
import os

def load_combined_data():
    base_dir = 'paper2_experiments'
    
    # 1. Kepler Original Results
    orig_path = os.path.join(base_dir, 'survey_results_kepler_orig.csv')
    df_orig = pd.read_csv(orig_path)
    df_orig = df_orig[['period', 'disagreement']]
    
    # 2. Long Period Kepler Results
    long_path = os.path.join(base_dir, 'survey_results_long_period_keep.csv')
    df_long = pd.read_csv(long_path)
    df_long = df_long[['period', 'disagreement']]
    
    # 3. Final Results (TESS)
    final_path = os.path.join(base_dir, 'survey_results_final.csv')
    df_final = pd.read_csv(final_path)
    df_final = df_final[['period', 'disagreement']]
    
    combined_df = pd.concat([df_orig, df_long, df_final], ignore_index=True)
    combined_df['period'] = pd.to_numeric(combined_df['period'], errors='coerce')
    combined_df = combined_df.dropna(subset=['period', 'disagreement'])
    
    return combined_df

def bootstrap_mean_ci(data, n_iterations=5000, alpha=0.05):
    if len(data) == 0: return np.nan, np.nan, np.nan
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_iterations)]
    return np.mean(data), np.percentile(boot_means, (alpha/2)*100), np.percentile(boot_means, (1-alpha/2)*100)

def run_bootstrap_analysis():
    df = load_combined_data()

    def get_regime(period):
        if period < 20: return 'Short (<20d)'
        elif 20 <= period <= 100: return 'Medium (20-100d)'
        else: return 'Long (>100d)'

    df['regime'] = df['period'].apply(get_regime)
    regimes = ['Short (<20d)', 'Medium (20-100d)', 'Long (>100d)']
    results = {}

    print(f"Sample Sizes: {[len(df[df['regime']==r]) for r in regimes]}")
    print(f"{'Regime':<20} | {'Mean':<6} | {'95% CI Lower':<12} | {'95% CI Upper':<12}")
    print("-" * 65)

    for regime in regimes:
        regime_data = df[df['regime'] == regime]['disagreement'].values
        mean_val, ci_low, ci_high = bootstrap_mean_ci(regime_data)
        results[regime] = {'mean': mean_val, 'low': ci_low, 'high': ci_high}
        if not np.isnan(mean_val):
            print(f"{regime:<20} | {mean_val:.4f} | {ci_low:.4f}       | {ci_high:.4f}")

    print("-" * 65)

    if not np.isnan(results[regimes[0]]['mean']) and not np.isnan(results[regimes[1]]['mean']):
        short_high = results[regimes[0]]['high']
        medium_low = results[regimes[1]]['low']
        overlap = not (medium_low > short_high or results[regimes[0]]['low'] > results[regimes[1]]['high'])
        print(f"Reliability Cliff Hypothesis (Short vs Medium overlap): {'OVERLAP' if overlap else 'NO OVERLAP (VERIFIED)'}")

if __name__ == "__main__":
    np.random.seed(42)
    run_bootstrap_analysis()
