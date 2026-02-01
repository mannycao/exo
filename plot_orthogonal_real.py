
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_orthogonal_sensitivity():
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

    # --- Plotting Logic ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data for plotting
    model_1d_sensitivity_means = [mean_drop_1d_noise, mean_drop_1d_mask] 
    model_1d_sensitivity_stds = [std_drop_1d_noise, std_drop_1d_mask]

    model_2d_sensitivity_means = [mean_drop_2d_noise, mean_drop_2d_mask]
    model_2d_sensitivity_stds = [std_drop_2d_noise, std_drop_2d_mask]

    attack_labels = ["Attack A: Temporal Noise", "Attack B: Spatial Occlusion"]
    x = np.arange(len(attack_labels))  # the label locations
    width = 0.2  # Adjusted width for closer bars

    rects1 = ax.bar(x - width/2, model_1d_sensitivity_means, width, 
                    yerr=model_1d_sensitivity_stds, capsize=5, label='1D Model (Temporal)', color='red', ecolor='gray')
    rects2 = ax.bar(x + width/2, model_2d_sensitivity_means, width, 
                    yerr=model_2d_sensitivity_stds, capsize=5, label='2D Model (Spatial)', color='blue', ecolor='gray')

    # Function to add labels on bars
    def autolabel(rects, means, stds, color='black'):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            xpos = rect.get_x() + rect.get_width() / 2
            
            # Label for mean value
            ax.text(xpos, height + 0.01, f'{means[i]:.2f}',
                    ha='center', va='bottom', fontsize=9, color=color)
            
            # Label for std dev value (just above the error bar cap)
            if stds[i] is not None and not np.isnan(stds[i]):
                # Find the top of the error bar
                err_cap_y = height + stds[i] + 0.02 # Add a small offset above the capsize
                ax.text(xpos, err_cap_y, f'({stds[i]:.2f})',
                        ha='center', va='bottom', fontsize=8, color=color)

    autolabel(rects1, model_1d_sensitivity_means, model_1d_sensitivity_stds, color='red')
    autolabel(rects2, model_2d_sensitivity_means, model_2d_sensitivity_stds, color='blue')

    # Aesthetics
    ax.set_ylabel("Average Confidence Drop", fontsize=12)
    ax.set_title("Figure 2: Orthogonal Sensitivity of Representations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, fontsize=10)
    ax.set_ylim(0.0, 1.0) # Scale from 0.0 to 1.0, adjusted for text labels
    ax.legend()
    ax.grid(axis='y')

    plt.tight_layout()
    output_filename = 'sensitivity_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Generated {output_filename}")

if __name__ == '__main__':
    plot_orthogonal_sensitivity()
