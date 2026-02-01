
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_sensitivity_plot():
    output_csv_path = 'threshold_sensitivity.csv'
    output_plot_path = 'threshold_sensitivity_plot.png'

    if not os.path.exists(output_csv_path):
        print(f"Error: {output_csv_path} not found. Please run threshold_sensitivity.py first to generate the data.")
        return

    # Load the generated results
    results_df = pd.read_csv(output_csv_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Disagreement Threshold (tau)')
    ax1.set_ylabel('Precision', color=color)
    ax1.plot(results_df['Threshold'], results_df['Precision'], color=color, marker='o', label='Precision')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-0.05, 1.05]) # Ensure y-axis covers full probability range

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Yield (True Flags)', color=color)  # we already handled the x-label with ax1
    ax2.plot(results_df['Threshold'], results_df['Yield'], color=color, marker='x', label='Yield')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a horizontal line for 90% precision target
    ax1.axhline(y=0.90, color='gray', linestyle='--', label='90% Precision Target')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Disagreement Threshold Sensitivity Analysis')
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0, 0), fancybox=True, shadow=True)
    plt.grid(True)
    
    plt.savefig(output_plot_path, dpi=300)
    print(f"Sensitivity plot saved to {output_plot_path}")

if __name__ == '__main__':
    generate_sensitivity_plot()
