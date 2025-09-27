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
    parser.add_argument('--split_type', type=str, default='all',
                        choices=['all', '50_50'],
                        help="Type of data split to use: 'all' for full dataset, '50_50' for 50/50 split of planets and false positives.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the number of files to process for testing purposes.")
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
    all_typed_light_curve_files = smart_data_fetcher(
        confirmed_planet_dir=args.planets_dir,
        false_positive_dir=args.false_positives_dir
    )

    if not all_typed_light_curve_files:
        logger.warning("No light curve files found. Exiting.")
        return 1

    typed_light_curve_files_to_process = []
    if args.split_type == '50_50':
        logger.info("Applying 50/50 split to data...")
        planets = [f for f in all_typed_light_curve_files if f['type'] == config.FILE_TYPE_CONFIRMED_PLANET]
        false_positives = [f for f in all_typed_light_curve_files if f['type'] == config.FILE_TYPE_FALSE_POSITIVE]

        # Take 50% of each, ensuring we don't go over the available count
        num_planets = len(planets) // 2
        num_false_positives = len(false_positives) // 2

        typed_light_curve_files_to_process.extend(planets[:num_planets])
        typed_light_curve_files_to_process.extend(false_positives[:num_false_positives])
        logger.info(f"Selected {len(planets[:num_planets])} planets and {len(false_positives[:num_false_positives])} false positives for 50/50 split.")
    else:
        typed_light_curve_files_to_process = all_typed_light_curve_files
        logger.info("Using all fetched data files.")

    if args.limit is not None and args.limit > 0:
        logger.info(f"Applying --limit argument: {args.limit}")
        # Ensure both classes are present when limiting for testing
        limited_planets = [f for f in typed_light_curve_files_to_process if f['type'] == config.FILE_TYPE_CONFIRMED_PLANET][:args.limit // 2]
        limited_false_positives = [f for f in typed_light_curve_files_to_process if f['type'] == config.FILE_TYPE_FALSE_POSITIVE][:args.limit // 2]
        
        typed_light_curve_files_to_process = limited_planets + limited_false_positives
        logger.info(f"Limiting processing to {len(typed_light_curve_files_to_process)} files (balanced) due to --limit argument.")
        
        if len(typed_light_curve_files_to_process) < 2:
            logger.error("Limit too small to get at least two classes. Please increase --limit.")
            return 1

    run_enhanced_pipeline(
        light_curve_files=typed_light_curve_files_to_process,
        output_dir_str=str(output_dir),
        timestamp=timestamp
    )
    logger.info("Pipeline execution finished.")
    return 0

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.exit(main())