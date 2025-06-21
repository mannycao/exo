"""
Feature Visualization Script for the Exoplanet Detection Pipeline.

This script loads the final prepared dataset (image features and labels)
from a pipeline run directory and displays samples from the positive (planet)
and negative (non-planet) classes.

This is a critical debugging step to visually inspect the data that the
AI model is being trained on.

Usage:
    python visualize_features.py /path/to/your/results/pipeline_run_YYYYMMDD_HHMMSS
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_features(run_directory):
    """
    Loads and visualizes features from a specified pipeline run directory.
    """
    run_path = Path(run_directory)
    dataset_path = run_path / "prepared_dataset_for_ml"

    if not dataset_path.is_dir():
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        print("Please ensure you have run the pipeline with the version of")
        print("enhanced_pipeline_runner.py that saves the final dataset.")
        return

    try:
        print(f"Loading data from: {dataset_path}")
        X_image = np.load(dataset_path / "X_image_features.npy")
        y_labels = np.load(dataset_path / "y_labels.npy")
        print(f"Successfully loaded {len(y_labels)} features.")
        print(f"Label distribution: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
    except FileNotFoundError as e:
        print(f"Error: Could not find required .npy file: {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # Find indices for each class
    positive_indices = np.where(y_labels == 1)[0]
    negative_indices = np.where(y_labels == 0)[0]

    if len(positive_indices) == 0:
        print("Warning: No positive samples (label 1) found in the dataset.")
    if len(negative_indices) == 0:
        print("Warning: No negative samples (label 0) found in the dataset.")

    # Select random samples to display
    num_samples_to_show = min(10, len(positive_indices), len(negative_indices))
    if num_samples_to_show == 0:
        print("Not enough samples from both classes to display.")
        # Optionally display whichever class has samples
        if len(positive_indices) > 0:
             print("Showing positive samples only...")
             num_samples_to_show_pos = min(10, len(positive_indices))
             plot_samples(X_image, positive_indices, num_samples_to_show_pos, "Positive Class (Planet Candidates)")
        if len(negative_indices) > 0:
             print("Showing negative samples only...")
             num_samples_to_show_neg = min(10, len(negative_indices))
             plot_samples(X_image, negative_indices, num_samples_to_show_neg, "Negative Class (Non-Planet/Noise)")
        plt.show()
        return
        
    plot_samples(X_image, positive_indices, num_samples_to_show, "Positive Class (Planet Candidates)")
    plot_samples(X_image, negative_indices, num_samples_to_show, "Negative Class (Non-Planet/Noise)")
    
    plt.show() # Display all generated figures

def plot_samples(X_image, indices, num_to_show, title):
    """Helper function to plot a grid of image samples."""
    
    # Randomly select indices without replacement
    selected_indices = np.random.choice(indices, size=num_to_show, replace=False)
    
    # Determine grid size (e.g., 2 rows for up to 10 samples)
    ncols = 5
    nrows = int(np.ceil(num_to_show / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    axes = axes.flatten() # Flatten to make indexing easier

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        # The image features from create_image_representations are 2D arrays
        # Use a colormap that works well for this kind of data, like 'viridis' or 'gray'
        im = ax.imshow(X_image[idx], cmap='viridis', origin='lower', aspect='auto')
        ax.set_title(f"Sample Index: {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for j in range(num_to_show, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/pipeline_run_directory")
        sys.exit(1)
        
    run_directory = sys.argv[1]
    visualize_features(run_directory)
