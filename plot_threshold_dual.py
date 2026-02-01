
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_threshold_dual_axis():
    # --- Data Source ---
    results_csv_path = 'threshold_sensitivity.csv'

    if not os.path.exists(results_csv_path):
        print(f"Error: {results_csv_path} not found. Please ensure the threshold sensitivity analysis (threshold_sensitivity.py) has been run to generate it.")
        return

    df = pd.read_csv(results_csv_path)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-Axis (Left): Precision
    color_precision = 'tab:blue'
    ax1.set_xlabel('Disagreement Threshold (tau)', fontsize=12)
    ax1.set_ylabel('Precision', color=color_precision, fontsize=12)
    ax1.plot(df['Threshold'], df['Precision'], color=color_precision, marker='o', label='Precision')
    ax1.tick_params(axis='y', labelcolor=color_precision)
    ax1.set_ylim(0.8, 1.0) # Range: [0.8, 1.0]

    # Secondary Y-Axis (Right): Yield
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_yield = 'tab:orange'
    ax2.set_ylabel('Number of Alerts (Yield)', color=color_yield, fontsize=12)
    ax2.plot(df['Threshold'], df['Yield'], color=color_yield, linestyle='--', marker='x', label='Yield')
    ax2.tick_params(axis='y', labelcolor=color_yield)

    # Aesthetics
    plt.title("Oracle Stability: Precision vs. Alert Volume", fontsize=14)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Add legend - combine handles and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True) # Add grid to primary axis

    # Save plot
    output_filename = 'threshold_dual.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Generated {output_filename}")

if __name__ == '__main__':
    plot_threshold_dual_axis()
