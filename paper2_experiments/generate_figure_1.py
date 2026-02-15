import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define project root and file paths
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "paper2_experiments" / "survey_results_kepler.csv"
OUTPUT_PLOT_PATH = project_root / "paper2_experiments" / "Figure1_Regime_Analysis.png"

def generate_figure_1():
    """
    Generates "Figure 1: Disagreement vs. Orbital Period" for the MDPI paper.
    """
    print(f"Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please ensure the survey_results_kepler.csv file exists.")
        return

    # Filter out any rows with NaN values in 'period' or 'disagreement'
    df.dropna(subset=['period', 'disagreement'], inplace=True)
    print(f"Loaded {len(df)} samples after dropping NaNs.")

    # Define Regimes
    def get_regime(period):
        if period < 20:
            return "Short (<20d)"
        elif 20 <= period <= 100:
            return "Medium (20-100d)"
        else:
            return "Long (>100d)"

    df['regime'] = df['period'].apply(get_regime)

    # Compute Metrics per Regime
    regime_order = ["Short (<20d)", "Medium (20-100d)", "Long (>100d)"]
    metrics_data = []

    for regime in regime_order:
        subset = df[df['regime'] == regime]
        mean_disagreement = subset['disagreement'].mean()
        high_disagreement_count = (subset['disagreement'] > 0.1).sum()
        total_count = len(subset)
        high_disagreement_rate = (high_disagreement_count / total_count) * 100 if total_count > 0 else 0
        proportion_high_disagreement = high_disagreement_rate / 100 # This will now be defined
        # Standard error of a proportion
        std_error_high_disagreement = np.sqrt(proportion_high_disagreement * (1 - proportion_high_disagreement) / total_count) * 100 if total_count > 0 else 0

        metrics_data.append({
            "Regime": regime,
            "Mean Disagreement": mean_disagreement,
            "High Disagreement Rate (%)": high_disagreement_rate,
            "Std Error Disagreement": subset['disagreement'].sem(), # Keep for boxplot potentially or just reference
            "Std Error High Disagreement": std_error_high_disagreement
        })
    
    metrics_df = pd.DataFrame(metrics_data)

    # --- Plotting ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Two subplots side-by-side

    # Panel A (Boxplot): Distribution of Disagreement Scores for Short, Medium, Long
    sns.boxplot(x='regime', y='disagreement', data=df, order=regime_order, ax=axes[0], palette="viridis", hue='regime', legend=False)
    axes[0].set_title("A. Disagreement Score Distribution by Orbital Period Regime")
    axes[0].set_xlabel("Orbital Period Regime")
    axes[0].set_ylabel(r"Disagreement $\delta = |p_{1D} - p_{2D}|$")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Panel B (Bar Chart): "High Disagreement Rate (%)" by Regime
    y_values = metrics_df['High Disagreement Rate (%)'].values
    yerr_values = metrics_df['Std Error High Disagreement'].values
    
    # Get colors from the 'viridis' palette for consistency
    colors = sns.color_palette("viridis", n_colors=len(regime_order))
    
    axes[1].bar(x=metrics_df['Regime'], height=y_values, yerr=yerr_values, color=colors)
    axes[1].set_title(r"B. Rate of High Disagreement ($\delta > 0.1$) by Orbital Period Regime")
    axes[1].set_xlabel("Orbital Period Regime")
    axes[1].set_ylabel("High Disagreement Rate (%)")
    axes[1].set_ylim(0, metrics_df['High Disagreement Rate (%)'].max() * 1.1) # Add some buffer to y-axis
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    print(f"\nPlot saved to: {OUTPUT_PLOT_PATH}")

    # --- Print Markdown Table of Stats ---
    print("\n### Disagreement Metrics by Orbital Period Regime")
    print(metrics_df.to_markdown(index=False))

if __name__ == "__main__":
    generate_figure_1()