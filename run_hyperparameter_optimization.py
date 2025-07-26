# run_hyperparameter_optimization.py

import logging
import argparse
import pandas as pd
import numpy as np

from pipeline import prepare_multimodal_data, optimize_hyperparameters
from data.data_fetcher import fetch_kepler_data, fetch_exoplanet_labels
from data.data_fetcher import download_light_curves

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for the exoplanet detection model.')
    parser.add_argument('--max_records', type=int, default=200, help='Maximum number of light curves to use for optimization.')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter settings that are sampled.')
    parser.add_argument('--cv', type=int, default=3, help='Number of cross-validation folds.')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting hyperparameter optimization process...")

    # --- Step 1: Fetch and Prepare Data ---
    logger.info("Fetching observation data and labels...")
    obs_table = fetch_kepler_data(max_records=args.max_records, use_cache=True)
    exoplanet_labels = fetch_exoplanet_labels(use_cache=True)
    
    logger.info("Downloading light curve files...")
    light_curve_files = download_light_curves(obs_table, use_cache=True)

    if not light_curve_files:
        logger.error("No light curve files were downloaded. Cannot proceed with optimization.")
        return

    logger.info(f"Preparing multimodal dataset from {len(light_curve_files)} files...")
    X_image, X_timeseries, y = prepare_multimodal_data(light_curve_files, exoplanet_labels)

    if X_image is None or X_timeseries is None or y is None:
        logger.error("Failed to create a dataset. Aborting optimization.")
        return

    # Ensure data is in the correct format for the model
    X_image = np.expand_dims(X_image, axis=-1)

    # --- Step 2: Run Optimization ---
    logger.info("Starting hyperparameter search...")
    best_params = optimize_hyperparameters(
        X_image,
        X_timeseries,
        y,
        n_iter=args.n_iter,
        cv=args.cv
    )

    # --- Step 3: Display Results ---
    logger.info("="*50)
    logger.info("Hyperparameter Optimization Finished")
    logger.info("="*50)
    if best_params:
        logger.info("Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
    else:
        logger.warning("Optimization did not return any parameters.")

if __name__ == "__main__":
    main()