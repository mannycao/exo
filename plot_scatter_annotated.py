
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_disagreement_scatter_annotated():
    # --- Setup ---
    disagreement_csv_path = 'representation_disagreement.csv'

    if not os.path.exists(disagreement_csv_path):
        print(f"Error: {disagreement_csv_path} not found. Please ensure Experiment 1 has been run to generate it.")
        return

    df = pd.read_csv(disagreement_csv_path)

    # --- Plotting ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate Delta for filtering Disagreement
    df['Delta'] = np.abs(df['Prob_1D'] - df['Prob_2D'])

    # Filter for Disagreement points (|p1-p2|>0.5) to be colored Red
    disagreement_points = df[df['Delta'] > 0.5]
    # All other points for Agreement (Gray)
    agreement_points = df[df['Delta'] <= 0.5]
    
    # Plot Agreement points (Gray)
    ax.scatter(agreement_points['Prob_1D'], agreement_points['Prob_2D'], 
               color='gray', alpha=0.1, label='Agreement', s=50)

    # Plot Disagreement points (Red)
    ax.scatter(disagreement_points['Prob_1D'], disagreement_points['Prob_2D'], 
               color='red', alpha=0.8, label='Disagreement', s=50)

    # Add diagonal line (y=x)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Agreement (y=x)')

    # --- Annotations (The Key Upgrade) ---
    # Draw dashed lines at x=0.5 and y=0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.0)

    # Add Text Boxes
    text_box_props_green = dict(boxstyle='round,pad=0.3', fc='white', ec='green', lw=0.5, alpha=0.9)
    text_box_props_red = dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=0.5, alpha=0.9)

    # Top-Right: "Consensus: Likely Planet"
    ax.text(0.75, 0.9, "Consensus: Likely Planet", ha='center', va='center', fontsize=11, 
            color='green', bbox=text_box_props_green)

    # Bottom-Left: "Consensus: Noise"
    ax.text(0.25, 0.1, "Consensus: Noise", ha='center', va='center', fontsize=11, 
            color='green', bbox=text_box_props_green)

    # Bottom-Right: "DISAGREEMENT\n(Latent Faults)"
    ax.text(0.75, 0.1, "DISAGREEMENT\n(Latent Faults)", ha='center', va='center', fontsize=11, 
            color='red', fontweight='bold', bbox=text_box_props_red)

    # Top-Left: "Disagreement"
    ax.text(0.25, 0.9, "Disagreement", ha='center', va='center', fontsize=11, 
            color='red', bbox=text_box_props_red)


    # Aesthetics
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('1D Model Output Score', fontsize=12)
    ax.set_ylabel('2D Model Output Score', fontsize=12)
    ax.set_title('Figure 1: Model Disagreement with Quadrant Analysis', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save plot
    output_filename = 'disagreement_scatter_annotated.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Annotated scatter plot saved as {output_filename}")

if __name__ == '__main__':
    plot_disagreement_scatter_annotated()
