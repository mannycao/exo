# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

def plot_confusion_matrix(y_true, y_pred, class_names, file_path):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(file_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, file_path):
    """
    Generates and saves a ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_path)
    plt.close()

def visualize_calibration_plot(y_true, y_pred, n_bins=10, filename="calibration_plot.png", output_dir="results"):
    """
    Visualizes a calibration plot (reliability diagram).
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    # Correcting the line for the 'perfectly calibrated' plot
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / filename)
    plt.close()

def visualize_uncertainty_distribution(uncertainty, y_pred_mean, y_true, filename="uncertainty_distribution.png", output_dir="results"):
    """
    Visualizes the distribution of uncertainty for correct and incorrect predictions.
    """
    # Ensure inputs are numpy arrays
    uncertainty = np.asarray(uncertainty)
    y_pred_mean = np.asarray(y_pred_mean)
    y_true = np.asarray(y_true)

    plt.figure(figsize=(10, 6))
    
    # Determine correctness of predictions by rounding the mean prediction
    correct_mask = np.round(y_pred_mean) == y_true
    correct_uncertainty = uncertainty[correct_mask]
    incorrect_uncertainty = uncertainty[~correct_mask]
    
    # Plot histograms for both correct and incorrect predictions if they exist
    if len(correct_uncertainty) > 0:
        sns.histplot(correct_uncertainty, color="green", label=f"Correct ({len(correct_uncertainty)})", kde=True, stat="density", common_norm=False)
    if len(incorrect_uncertainty) > 0:
        sns.histplot(incorrect_uncertainty, color="red", label=f"Incorrect ({len(incorrect_uncertainty)})", kde=True, stat="density", common_norm=False)
    
    plt.title("Distribution of Prediction Uncertainty")
    plt.xlabel("Uncertainty (Variance)")
    plt.ylabel("Density")
    plt.legend()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / filename)
    plt.close()
