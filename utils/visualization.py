# FILE: utils/visualization.py (Final, Complete, and Corrected Version)

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _save_plot(output_dir, filename, figure):
    """Helper function to save a matplotlib figure and close it properly."""
    if not output_dir:
        logger.warning(f"No output directory provided for plot '{filename}'. Skipping save.")
        plt.close(figure)
        return
    try:
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        logger.info(f"Saved visualization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}")
    finally:
        # Explicitly close the figure to release memory and prevent hangs
        plt.close(figure)


def visualize_transit_model(time, flux, model_flux, period, output_dir, filename='transit_model_fit.png'):
    """
    Creates a plot of the light curve with the batman model fit overlaid.
    """
    if time is None or flux is None or model_flux is None:
        logger.warning("Invalid data for transit model visualization. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, flux, '.', label='Detrended Flux', markersize=2, alpha=0.7)
    ax.plot(time, model_flux, 'r-', label='Fitted Batman Model', linewidth=2)
    
    ax.set_title(f'Transit Model Fit (Period = {period:.4f} days)')
    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Normalized Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    _save_plot(output_dir, filename, fig)


def visualize_periodogram(frequency, power, period, output_dir, filename='periodogram.png'):
    """
    Creates and saves a plot of the Lomb-Scargle periodogram.
    """
    if frequency is None or power is None or period is None:
        logger.warning("Invalid data for periodogram. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(1. / frequency, power, 'b-')
    ax.axvline(period, color='r', linestyle='--', label=f'Best Period = {period:.4f} days')
    ax.set_title('Lomb-Scargle Periodogram')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_plot(output_dir, filename, fig)


def visualize_folded_transit(time, flux, period, t0, output_dir, filename='folded_transit.png'):
    """
    Creates and saves a phase-folded light curve plot.
    """
    if period is None or period <= 0:
        logger.warning("Invalid period provided for folding. Skipping folded plot.")
        return

    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    sort_mask = np.argsort(phase)
    phase_sorted, flux_sorted = phase[sort_mask], flux[sort_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phase_sorted, flux_sorted, '.', markersize=2, label='Phase-Folded Flux')
    ax.set_title(f'Light Curve Folded at Period = {period:.4f} days')
    ax.set_xlabel('Phase (days)')
    ax.set_ylabel('Normalized Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    _save_plot(output_dir, filename, fig)


def visualize_transit(time, flux, transit_times, transit_depths, period, output_dir, filename='transit_visualization.png'):
    """
    Creates a plot of the light curve, highlighting detected transits.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, flux, '.', label='Detrended Flux', markersize=2)
    
    if transit_times is not None and len(transit_times) > 0 and transit_depths is not None and len(transit_depths) > 0:
        ax.plot(transit_times, 1 - transit_depths, 'ro', label='Detected Transits', markersize=4)
    
    ax.set_title(f'Detected Transits (Period = {period:.4f} days)')
    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Normalized Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    _save_plot(output_dir, filename, fig)


def visualize_detection_results(processed_results, max_plots=25, output_dir=None, filename='all_detections_summary.png'):
    """
    Creates a gallery of plots for each detected transit event.
    """
    if not processed_results: return

    all_detections = [seg for r in processed_results if r.get('success') and r.get('transit_segments') for seg in r['transit_segments']]
    if not all_detections:
        logger.warning("No transit segments found in any processed results to visualize.")
        return

    if len(all_detections) > max_plots:
        indices = np.random.choice(len(all_detections), max_plots, replace=False)
        all_detections = [all_detections[i] for i in indices]

    num_detections = len(all_detections)
    if num_detections == 0: return

    cols = int(np.ceil(np.sqrt(num_detections)))
    rows = int(np.ceil(num_detections / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes_flat = axes.flatten()

    for i, segment in enumerate(all_detections):
        axes_flat[i].plot(segment, '.-')
        axes_flat[i].set_title(f'Detection #{i+1}')
        axes_flat[i].set_xticks([])
        axes_flat[i].set_yticks([])

    for i in range(num_detections, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    _save_plot(output_dir, filename, fig)


def visualize_learning_curves(history, output_dir, filename='learning_curves.png'):
    """
    Creates plots for model training and validation accuracy and loss.
    """
    if not history: return
    
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'accuracy' in df and 'val_accuracy' in df:
        ax1.plot(df['accuracy'], label='Train Accuracy')
        ax1.plot(df['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend(loc='upper left')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')

    if 'loss' in df and 'val_loss' in df:
        ax2.plot(df['loss'], label='Train Loss')
        ax2.plot(df['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.legend(loc='upper left')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    fig.tight_layout()
    _save_plot(output_dir, filename, fig)

