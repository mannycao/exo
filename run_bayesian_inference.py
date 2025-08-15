# run_bayesian_inference.py (Legacy Demonstration Script)
#
# NOTE: The functionality of this script has been integrated into the main
# pipeline runner. The recommended way to perform this analysis is now:
# `python enhanced_main.py --mode bayesian_cross_val`
#
# This script is preserved here as a reference for the original, standalone
# inference and triage process.

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# --- Workaround for astropy logging issue ---
try:
    from astropy.utils.logger import AstropyLogger
    def _set_defaults_noop(self, **kwargs):
        pass
    AstropyLogger._set_defaults = _set_defaults_noop
except (ImportError, AttributeError):
    pass
# --- End of patch ---

# --- Add the project root to the system path ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.file_utils import setup_logging
from data.real_data_fetcher import smart_data_fetcher
from legacy_pipeline import prepare_multimodal_data
from models.bayesian_predictor import BayesianPredictor
from models.enhanced_trainer import focal_loss
from utils.visualization import visualize_calibration_plot, visualize_uncertainty_distribution
from utils.metrics import get_uncertainty_metrics

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
        custom_objects = {'focal_loss_fixed': focal_loss()}
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Fetching and preparing data for inference...")
    # Use the paths from the arguments, as this script doesn't use the config file
    light_curve_files = smart_data_fetcher(args.planets_dir, args.false_positives_dir)
    if not light_curve_files:
        logger.error("No data found. Aborting.")
        return
    
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
    low_uncert_thresh = np.percentile(y_pred_uncertainty, 25)
    high_confidence_planets = (y_pred_mean > 0.90) & (y_pred_uncertainty < low_uncert_thresh)
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
    visualize_uncertainty_distribution(y_pred_uncertainty, y_pred_mean, y_true, filename="uncertainty_distribution.png", output_dir=output_dir)

    # --- 5. Generate Report ---
    logger.info("Generating report...")
    results_for_report = []
    for i, file_info in enumerate(light_curve_files):
        results_for_report.append({
            'file_path': file_info['file_path'],
            'success': True,
            'prediction': y_pred_mean[i],
            'uncertainty': y_pred_uncertainty[i],
            'true_label': y_true[i]
        })
    
    report_df = pd.DataFrame(results_for_report)
    report_df.to_csv(output_dir / "inference_results.csv", index=False)
    logger.info(f"Inference results saved to {output_dir / 'inference_results.csv'}")

    logger.info("--- Bayesian Inference Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Inference with MC Dropout on a trained model (Legacy).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--planets_dir', type=str, required=True, help="Directory for confirmed planet light curves.")
    parser.add_argument('--false_positives_dir', type=str, required=True, help="Directory for false positive light curves.")
    parser.add_argument('--output_dir', type=str, default="results/bayesian_run", help="Directory to save inference results and plots.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of Monte Carlo samples for inference.")
    parser.add_argument('--sample_size', type=int, default=None, help="Limit the number of files to process for a quick test.")
    
    main_args = parser.parse_args()
    run_inference(main_args)
