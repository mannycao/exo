# run_real_data.py

import logging
import argparse
from datetime import datetime
from pathlib import Path

import config
# --- THIS IS THE FIX ---
# Import the logging setup from its correct, modern location
from utils.file_utils import setup_logging
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
from pipeline.real_data_pipeline import run_real_data_pipeline

def main():
    """
    Main entry point for running the exoplanet detection pipeline on real, verified data.
    """
    parser = argparse.ArgumentParser(description='Run the Exoplanet Detection Pipeline on real data.')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Total number of targets to fetch (half confirmed, half false positive).')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of light curves to process in each batch.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers for processing. Defaults to CPU count.')

    args = parser.parse_args()

    # Create a timestamped directory for this run's results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.RESULTS_DIR) / f"real_data_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file within the new run-specific directory
    log_file_path = output_dir / 'real_data_pipeline.log'
    setup_logging(log_file_path=log_file_path)
    logger = logging.getLogger(__name__)

    logger.info(f"Real data pipeline run initiated. Output directory: {output_dir}")
    logger.info(f"Command line arguments: {args}")

    # Launch the main pipeline
    run_real_data_pipeline(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        output_dir=output_dir,
        max_workers=args.workers
    )

    logger.info("Real data pipeline execution completed.")

if __name__ == "__main__":
    main()