"""
Visualization utilities for the exoplanet detection pipeline.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

logger = logging.getLogger(__name__)


def visualize_transit(time, flux, transit_info=None, title=None, filename=None, output_dir=None):
    """
    Create visualization of a light curve and detected transits.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        transit_info: Dictionary with transit information (optional)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, flux, 'k-', alpha=0.8, label='Normalized Flux')
    
    if transit_info is not None and len(transit_info['peak_indices']) > 0:
        plt.scatter(
            transit_info['times'], 
            transit_info['depths'], 
            color='red', 
            s=50, 
            marker='v', 
            label=f"Detected Transits ({len(transit_info['peak_indices'])})"
        )
        
        # Highlight the transit regions
        for idx, width in zip(transit_info['peak_indices'], transit_info['widths']):
            half_width = int(width / 2)
            left_idx = max(0, idx - half_width)
            right_idx = min(len(flux) - 1, idx + half_width)
            plt.axvspan(time[left_idx], time[right_idx], color='red', alpha=0.2)
    
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Normalized Flux')
    plt.title(title or 'Light Curve with Detected Transits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_folded_transit(phase, flux, binned_phase=None, binned_flux=None, 
                             binned_error=None, period=None, title=None, filename=None, output_dir=None):
    """
    Create visualization of a phase-folded light curve.
    
    Args:
        phase: Phase values (0-1)
        flux: Corresponding flux values
        binned_phase: Binned phase values (optional)
        binned_flux: Binned flux values (optional)
        binned_error: Binned flux error values (optional)
        period: Orbital period in days (optional, for title)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the raw folded data
    plt.scatter(phase, flux, s=2, alpha=0.3, color='gray', label='Folded Data')
    
    # Plot the binned data if provided
    if binned_phase is not None and binned_flux is not None:
        if binned_error is not None:
            plt.errorbar(
                binned_phase, binned_flux, yerr=binned_error,
                fmt='o-', color='blue', ecolor='blue', alpha=0.7,
                label='Binned Data'
            )
        else:
            plt.plot(binned_phase, binned_flux, 'o-', color='blue', alpha=0.7, label='Binned Data')
    
    period_str = f" (Period: {period:.2f} days)" if period else ""
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.title(title or f'Phase-Folded Light Curve{period_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved folded transit visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_periodogram(period, power, peak_periods=None, title=None, filename=None, output_dir=None):
    """
    Create visualization of a periodogram.
    
    Args:
        period: Period values
        power: Corresponding power values
        peak_periods: List of peak periods to highlight (optional)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the periodogram
    plt.semilogx(period, power, 'k-', alpha=0.8)
    
    # Mark peak periods if provided
    if peak_periods is not None and len(peak_periods) > 0:
        for i, p in enumerate(peak_periods[:5]):  # Show top 5 peaks
            idx = np.argmin(np.abs(period - p))
            plt.plot(p, power[idx], 'ro', markersize=8)
            plt.annotate(
                f"{p:.2f} days", 
                xy=(p, power[idx]), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center'
            )
    
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(title or 'Lomb-Scargle Periodogram')
    plt.grid(True, alpha=0.3)
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved periodogram visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_transit_model(time, flux, model_flux, title=None, filename=None, output_dir=None):
    """
    Create visualization of a transit model fit.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        model_flux: Model flux data
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the data and model
    plt.scatter(time, flux, s=2, alpha=0.5, color='gray', label='Data')
    plt.plot(time, model_flux, 'r-', linewidth=2, label='Transit Model')
    
    # Plot the residuals
    residuals = flux - model_flux
    offset = min(flux) - 0.02
    plt.plot(time, residuals + offset, 'k-', alpha=0.5, label='Residuals')
    plt.axhline(offset, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Normalized Flux')
    plt.title(title or 'Transit Model Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transit model visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_detection_results(results, title=None, filename=None, output_dir=None):
    """
    Create visualization summarizing detection results.
    
    Args:
        results: List of detection results
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    # Extract data
    # Safely get transit counts
    transit_counts = [r.get('transit_count', 0) for r in results]
    
    # Safely get periods, only for results that have the 'periodicity' key and it's not None
    periods = [r.get('periodicity') for r in results if r.get('periodicity') is not None]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot transit count histogram
    ax1.hist(transit_counts, bins=np.arange(0, max(transit_counts) + 2) - 0.5, alpha=0.7)
    ax1.set_xlabel('Number of Transits Detected')
    ax1.set_ylabel('Count')
    ax1.set_title('Transit Detection Histogram')
    ax1.grid(True, alpha=0.3)
    
    # Plot period histogram
    if periods:
        ax2.hist(periods, bins=20, alpha=0.7)
        ax2.set_xlabel('Orbital Period (days)')
        ax2.set_ylabel('Count')
        ax2.set_title('Orbital Period Histogram')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No periods detected', ha='center', va='center')
        ax2.set_title('Orbital Period Histogram')
    
    plt.suptitle(title or 'Exoplanet Detection Results')
    plt.tight_layout()
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved detection results visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_model_comparison(model_metrics, title=None, filename=None, output_dir=None):
    """
    Create visualization comparing multiple model metrics.
    
    Args:
        model_metrics: Dictionary mapping model names to metric dictionaries
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    if not model_metrics:
        logger.warning("No model metrics provided for comparison")
        return
    
    # Get common metrics across all models
    common_metrics = set.intersection(*[set(metrics.keys()) for metrics in model_metrics.values()])
    # Filter to typical classification metrics
    important_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'average_precision', 'roc_auc']
    metrics_to_plot = [m for m in important_metrics if m in common_metrics]
    
    if not metrics_to_plot:
        logger.warning("No common metrics found for comparison")
        return
    
    # Create a DataFrame for plotting
    data = []
    for model_name, metrics in model_metrics.items():
        row = {'Model': model_name}
        for metric in metrics_to_plot:
            row[metric] = metrics.get(metric, 0)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    bar_width = 0.15
    x = np.arange(len(df))
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * bar_width
        plt.bar(x + offset, df[metric], width=bar_width, label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(title or 'Model Comparison')
    plt.xticks(x, df['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_learning_curves(history, metrics=None, title=None, filename=None, output_dir=None):
    """
    Visualize learning curves from model training history.
    
    Args:
        history: Keras history object or dictionary
        metrics: List of metrics to plot (optional)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    # Convert to dictionary if it's a Keras history object
    if hasattr(history, 'history'):
        history = history.history
    
    if not history:
        logger.warning("Empty training history provided")
        return
    
    # If metrics not specified, plot all except validation metrics
    if metrics is None:
        metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history[metric], label=f'Training {metric}')
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs. Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title or 'Learning Curves')
    plt.tight_layout()
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_periodicity_analysis(periodogram_data, title=None, filename=None, output_dir=None):
    """
    Create a comprehensive visualization of periodicity analysis results including confidence metrics.
    
    Args:
        periodogram_data: Dictionary containing periodicity analysis results
            Including: period, power, best_period, fap, power_snr, peak_snr, confidence_score
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, 2)

    # Main periodogram plot
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.semilogx(periodogram_data['period'], periodogram_data['power'], 'k-', alpha=0.8)
    
    # Highlight best period
    best_idx = np.argmin(np.abs(np.array(periodogram_data['period']) - periodogram_data['best_period']))
    ax1.plot(periodogram_data['best_period'], periodogram_data['power'][best_idx], 'ro', markersize=10)
    ax1.annotate(
        f"Best Period: {periodogram_data['best_period']:.2f} days",
        xy=(periodogram_data['best_period'], periodogram_data['power'][best_idx]),
        xytext=(0, 10), textcoords='offset points',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    ax1.set_xlabel('Period (days)')
    ax1.set_ylabel('Power')
    ax1.set_title('Lomb-Scargle Periodogram')
    ax1.grid(True, alpha=0.3)

    # Confidence metrics visualization
    ax2 = fig.add_subplot(gs[2, 0])
    metrics = ['FAP', 'Power SNR', 'Peak SNR']
    values = [
        1 - periodogram_data['fap'],  # Convert FAP to confidence
        min(1.0, periodogram_data['power_snr'] / 10),  # Normalize to 0-1
        min(1.0, periodogram_data['peak_snr'] / 5)     # Normalize to 0-1
    ]
    
    bars = ax2.bar(metrics, values)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Confidence Metrics')
    
    # Color bars based on values
    colors = plt.cm.RdYlGn(np.array(values))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{val:.2f}',
            ha='center', va='bottom'
        )

    # Overall confidence gauge
    ax3 = fig.add_subplot(gs[2, 1])
    confidence = periodogram_data['confidence_score']
    
    # Create a gauge-like visualization
    gauge = np.linspace(0, np.pi, 100)
    radius = np.ones_like(gauge)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(gauge)))
    
    ax3.set_aspect('equal')
    
    # Plot the gauge background
    for i in range(len(gauge)-1):
        ax3.fill_between(
            [gauge[i], gauge[i+1]], [0, 0], [radius[i], radius[i+1]],
            color=colors[i]
        )
    
    # Plot the confidence needle
    needle_angle = confidence * np.pi
    ax3.plot([0, needle_angle], [0, 0.8], 'k-', linewidth=3)
    ax3.text(np.pi/2, -0.2, f'Confidence: {confidence:.2%}', ha='center', va='top')
    
    ax3.set_xlim(0, np.pi)
    ax3.set_ylim(-0.5, 1)
    ax3.set_title('Overall Confidence')
    ax3.axis('off')

    plt.suptitle(title or 'Periodicity Analysis Results', y=0.95)
    plt.tight_layout()
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved periodicity analysis visualization to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()


def visualize_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, filename=None, output_dir=None):
    """
    Visualize confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (not probabilities)
        classes: List of class names (optional)
        normalize: Whether to normalize the confusion matrix (optional)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    from sklearn.metrics import confusion_matrix
    import itertools
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title or 'Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    if classes is None:
        classes = [f'Class {i}' for i in range(cm.shape[0])]
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization to {full_path}: {e}")
        plt.close()
    else:
        plt.show()
