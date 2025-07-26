#!/usr/bin/env python3
"""
Main entry point for the enhanced exoplanet detection pipeline.
Now supports typed local data inputs for improved labeling.
"""
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import config
# Import setup_logging from its correct location
from utils.file_utils import setup_logging
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
# Import the data fetcher that works with the enhanced pipeline
from data.real_data_fetcher import smart_data_fetcher

def main():
    """Main function to run the enhanced exoplanet detection pipeline."""
    parser = argparse.ArgumentParser(description='Exoplanet Detection Pipeline')
    parser.add_argument('--planets_dir', type=str, required=True,
                        help="Directory containing light curves for confirmed exoplanets.")
    parser.add_argument('--false_positives_dir', type=str, required=True,
                        help="Directory containing light curves for known false positives.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.RESULTS_DIR) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = output_dir / 'pipeline.log'
    setup_logging(log_file_path=str(log_file_path))
    logger = logging.getLogger(__name__)
    
    logger.info(f"Pipeline run initiated. Output directory: {output_dir}")
    logger.info(f"Arguments: {args}")

    logger.info("Fetching local data files...")
    typed_light_curve_files = smart_data_fetcher(
        confirmed_planet_dir=args.planets_dir,
        false_positive_dir=args.false_positives_dir
    )

    if not typed_light_curve_files:
        logger.warning("No light curve files found. Exiting.")
        return 1

    run_enhanced_pipeline(
        light_curve_files=typed_light_curve_files,
        output_dir_str=str(output_dir)
    )
    logger.info("Pipeline execution finished.")
    return 0

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.exit(main())