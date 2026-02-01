
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_disagreement_scatter():
    # --- Setup ---
    disagreement_csv_path = 'representation_disagreement.csv'

    if not os.path.exists(disagreement_csv_path):
        print(f"Error: {disagreement_csv_path} not found. Please ensure Experiment 1 has been run to generate it.")
        return

    df = pd.read_csv(disagreement_csv_path)

    # --- Plotting Logic ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))

    # Calculate Delta for filtering Latent Faults
    df['Delta'] = np.abs(df['Prob_1D'] - df['Prob_2D'])

    # Filter Valid Candidates (Ground_Truth_Label == 1)
    valid_candidates = df[df['Ground_Truth_Label'] == 1]
    # Filter Latent Faults (Ground_Truth_Label == 0 AND Delta > 0.5)
    latent_faults = df[(df['Ground_Truth_Label'] == 0) & (df['Delta'] > 0.5)]
    # Filter other False Positives (Ground_Truth_Label == 0, but Delta <= 0.5 or not specifically called out)
    # This ensures all ground_truth_label == 0 points are plotted, with specific ones highlighted as 'latent_faults'
    other_false_positives = df[(df['Ground_Truth_Label'] == 0) & (df['Delta'] <= 0.5)]

    # Plot Valid Candidates (Gray)
    plt.scatter(valid_candidates['Prob_1D'], valid_candidates['Prob_2D'], 
                color='gray', alpha=0.1, label='Confirmed Planet (Valid Candidate)', s=50) # s is marker size

    # Plot other False Positives (Light Red/Orange to differentiate from highlighted latent faults)
    plt.scatter(other_false_positives['Prob_1D'], other_false_positives['Prob_2D'], 
                color='orange', alpha=0.3, label='False Positive (Low Disagreement)', s=50)


    # Plot Latent Faults (Red)
    plt.scatter(latent_faults['Prob_1D'], latent_faults['Prob_2D'], 
                color='red', alpha=0.8, label='Latent Fault (High Disagreement)', s=50)

    # Add diagonal line for "Perfect Agreement"
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Agreement (y=x)')

    # Annotate the "Latent Fault Cluster"
    # The latent faults are defined as Ground_Truth_Label == 0 AND Delta > 0.5.
    # From the problem description "Annotate the 'Latent Fault Cluster' (High 1D, Low 2D)"
    # This implies points with high Prob_1D and low Prob_2D, and vice-versa.
    # Let's pick a representative point for annotation, e.g., for High 1D, Low 2D.
    # We can find the average position of these points for a more accurate arrow.
    high_1d_low_2d_faults = latent_faults[latent_faults['Prob_1D'] > latent_faults['Prob_2D']]
    if not high_1d_low_2d_faults.empty:
        # Example annotation for high 1D, low 2D cluster
        annotate_x = high_1d_low_2d_faults['Prob_1D'].mean() 
        annotate_y = high_1d_low_2d_faults['Prob_2D'].mean() 
        # Adjust annotation position to be readable, pointing from text to cluster mean
        plt.annotate('Latent Fault Cluster', xy=(annotate_x, annotate_y), xytext=(0.2, 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12, color='black',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

    # Aesthetics
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('1D Model Output Score', fontsize=12)
    plt.ylabel('2D Model Output Score', fontsize=12)
    plt.title('Model Disagreement Scatter Plot', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save plot
    output_filename = 'disagreement_scatter.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved as {output_filename}")

if __name__ == '__main__':
    plot_disagreement_scatter()
