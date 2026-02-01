
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_orthogonal_sensitivity_errorbars():
    # --- Data Source ---
    results_csv_path = 'stress_test_results.csv'

    if not os.path.exists(results_csv_path):
        print(f"Error: {results_csv_path} not found. Please ensure the stress test (orthogonality_test.py) has been run to generate it.")
        return

    df = pd.read_csv(results_csv_path)

    # Calculate the mean and standard deviation of the specified columns
    mean_drop_1d_noise = df['Drop_1D_Noise'].mean()
    std_drop_1d_noise = df['Drop_1D_Noise'].std()
    mean_drop_2d_noise = df['Drop_2D_Noise'].mean()
    std_drop_2d_noise = df['Drop_2D_Noise'].std()
    mean_drop_1d_mask = df['Drop_1D_Mask'].mean()
    std_drop_1d_mask = df['Drop_1D_Mask'].std()
    mean_drop_2d_mask = df['Drop_2D_Mask'].mean()
    std_drop_2d_mask = df['Drop_2D_Mask'].std()

    # --- Plotting ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data for plotting
    model_1d_sensitivity_means = [mean_drop_1d_noise, mean_drop_1d_mask] 
    model_1d_sensitivity_stds = [std_drop_1d_noise, std_drop_1d_mask]

    model_2d_sensitivity_means = [mean_drop_2d_noise, mean_drop_2d_mask]
    model_2d_sensitivity_stds = [std_drop_2d_noise, std_drop_2d_mask]

    attack_labels = ["Temporal Noise Attack", "Spatial Occlusion Attack"]
    x = np.arange(len(attack_labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, model_1d_sensitivity_means, width, 
                    yerr=model_1d_sensitivity_stds, capsize=5, label='1D Model (Temporal)', color='red', ecolor='gray')
    rects2 = ax.bar(x + width/2, model_2d_sensitivity_means, width, 
                    yerr=model_2d_sensitivity_stds, capsize=5, label='2D Model (Spatial)', color='blue', ecolor='gray')

    # Aesthetics
    ax.set_ylabel("Confidence Drop", fontsize=12)
    ax.set_title("Figure 2: Orthogonal Sensitivity (Mean ± Std Dev)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, fontsize=10)
    ax.set_ylim(0.0, max(max(model_1d_sensitivity_means + model_1d_sensitivity_stds), max(model_2d_sensitivity_means + model_2d_sensitivity_stds)) * 1.1) # Dynamic y-limit
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Grid only on y-axis

    plt.tight_layout()
    output_filename = 'sensitivity_errorbars.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Generated {output_filename}")

if __name__ == '__main__':
    plot_orthogonal_sensitivity_errorbars()
