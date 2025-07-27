# run_bayesian_inference.py

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# --- Workaround for astropy logging issue ---
# This patch addresses a known conflict between astropy's logging system and
# other logging configurations. The error "'Logger' object has no attribute
# '_set_defaults'" occurs when astropy tries to call a custom method on the
# standard logger. This patch defines that method as a no-op to prevent a crash.
try:
    from astropy.utils.logger import AstropyLogger
    def _set_defaults_noop(self, **kwargs):
        pass
    AstropyLogger._set_defaults = _set_defaults_noop
except (ImportError, AttributeError):
    # This will fail gracefully if astropy is not installed or the class structure changes.
    pass
# --- End of patch ---

# --- Add the project root to the system path ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from utils.file_utils import setup_logging
from data.real_data_fetcher import smart_data_fetcher
from legacy_pipeline import prepare_multimodal_data
from models.bayesian_predictor import BayesianPredictor
from models.enhanced_trainer import focal_loss # Import the custom loss function factory
from utils.visualization import visualize_calibration_plot, visualize_uncertainty_distribution
from utils.metrics import get_uncertainty_metrics
from pipeline.report_generator import generate_html_report # We'll adapt this

def run_inference(args):
    """
    Runs Bayesian inference on a trained model and generates an uncertainty report.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = output_dir / 'bayesian_inference.log'
    setup_logging(str(log_file_path))
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Bayesian Inference Pipeline ---")
    logger.info(f"Loading trained model from: {args.model_path}")

    # --- 1. Load Model and Data ---
    try:
        # The key 'focal_loss_fixed' must match the name Keras used when saving the model.
        # The value is the actual loss function object, obtained by calling the factory.
        custom_objects = {'focal_loss_fixed': focal_loss()}
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Fetching and preparing data for inference...")
    light_curve_files = smart_data_fetcher(args.planets_dir, args.false_positives_dir)
    if not light_curve_files:
        logger.error("No data found. Aborting.")
        return
    
    # We use a subset for demonstration purposes if specified
    if args.sample_size:
        light_curve_files = light_curve_files[:args.sample_size]

    labels_df = pd.DataFrame(light_curve_files)
    X_image, X_timeseries, y_true = prepare_multimodal_data(light_curve_files, labels_df)

    if X_image is None or X_timeseries is None:
        logger.error("Failed to prepare data. Aborting.")
        return

    # --- 2. Perform Bayesian Inference ---
    predictor = BayesianPredictor(model, n_samples=args.n_samples)
    y_pred_mean, y_pred_uncertainty = predictor.predict(X_image, X_timeseries)

    # --- 3. Probabilistic Triage and Analysis ---
    logger.info("--- Probabilistic Triage Results ---")
    
    # Define thresholds for triage
    high_conf_thresh = 0.90
    low_uncert_thresh = np.percentile(y_pred_uncertainty, 25) # e.g., lower quartile of uncertainty

    high_confidence_planets = (y_pred_mean > high_conf_thresh) & (y_pred_uncertainty < low_uncert_thresh)
    ambiguous_candidates = (y_pred_mean > 0.5) & (y_pred_uncertainty >= low_uncert_thresh)
    
    logger.info(f"High-Confidence Planet Candidates: {np.sum(high_confidence_planets)}")
    logger.info(f"Ambiguous Candidates for Review: {np.sum(ambiguous_candidates)}")
    logger.info(f"Likely False Positives (<0.5 prob): {np.sum(y_pred_mean <= 0.5)}")

    # --- 4. Evaluate Uncertainty Quality ---
    logger.info("Evaluating uncertainty quality...")
    uncertainty_metrics = get_uncertainty_metrics(y_true, y_pred_mean)
    logger.info(f"Expected Calibration Error (ECE): {uncertainty_metrics['expected_calibration_error']:.4f}")
    logger.info(f"Brier Score: {uncertainty_metrics['brier_score']:.4f}")

    visualize_calibration_plot(y_true, y_pred_mean, filename="calibration_plot.png", output_dir=output_dir)
    # Updated call to include y_pred_mean for more insightful visualization
    visualize_uncertainty_distribution(y_pred_uncertainty, y_pred_mean, y_true, filename="uncertainty_distribution.png", output_dir=output_dir)

    # --- 5. Generate Report ---
    logger.info("Generating HTML report...")
    # Adapt the report generator's input format
    results_for_report = []
    for i, file_info in enumerate(light_curve_files):
        results_for_report.append({
            'file_path': file_info['file_path'],
            'success': True,
            'prediction': y_pred_mean[i],
            'uncertainty': y_pred_uncertainty[i],
            'true_label': y_true[i]
        })
    
    # A simplified call to a new or adapted report function
    # For now, we'll just log the results. A full report would require modifying report_generator.py
    # to accept this new data structure.
    report_df = pd.DataFrame(results_for_report)
    report_df.to_csv(output_dir / "inference_results.csv", index=False)
    logger.info(f"Inference results saved to {output_dir / 'inference_results.csv'}")

    logger.info("--- Bayesian Inference Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Inference with MC Dropout on a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--planets_dir', type=str, required=True, help="Directory for confirmed planet light curves.")
    parser.add_argument('--false_positives_dir', type=str, required=True, help="Directory for false positive light curves.")
    parser.add_argument('--output_dir', type=str, default="results/bayesian_run", help="Directory to save inference results and plots.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of Monte Carlo samples for inference.")
    parser.add_argument('--sample_size', type=int, default=None, help="Limit the number of files to process for a quick test.")
    
    main_args = parser.parse_args()
    run_inference(main_args)
