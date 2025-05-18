#!/usr/bin/env python3
"""
Enhanced main entry point for the exoplanet detection pipeline.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import config
from pipeline.pipeline_runner import setup_logging
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from data.data_fetcher import fetch_kepler_data, fetch_exoplanet_labels


def main():
    """
    Main function to run the enhanced exoplanet detection pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Enhanced Exoplanet Detection Pipeline')
    parser.add_argument('--max-records', type=int, default=50,
                        help='Maximum number of records to process')
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Use synthetic data instead of real data')
    parser.add_argument('--multimodal', action='store_true',
                        help='Use multimodal fusion model')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--workers', type=int, default=config.DEFAULT_MAX_WORKERS,
                        help='Maximum number of parallel workers')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable data caching')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    
    # Enhanced pipeline options
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced pipeline with improved techniques')
    parser.add_argument('--augmentation-factor', type=int, default=config.AUGMENTATION_FACTOR,
                        help='Factor by which to augment positive examples')
    parser.add_argument('--include-tess', action='store_true',
                        help='Include TESS data in addition to Kepler data')
    parser.add_argument('--no-enhanced-augmentation', action='store_true',
                        help='Disable enhanced data augmentation')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting enhanced exoplanet detection pipeline at {datetime.now()}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = config.RESULTS_DIR
    
    # Handle data sources
    if args.synthetic_data:
        logger.info("Using synthetic data")
        obs_table = None
    else:
        logger.info(f"Fetching Kepler data (max records: {args.max_records})")
        obs_table = fetch_kepler_data(max_records=args.max_records, use_cache=not args.no_cache)
    
    # Get exoplanet labels for training
    exoplanet_labels = fetch_exoplanet_labels(use_cache=not args.no_cache)
    
    # Run the pipeline (enhanced by default)
    results = run_enhanced_pipeline(
        obs_table=obs_table,
        exoplanet_labels=exoplanet_labels,
        use_synthetic=args.synthetic_data,
        use_multimodal=args.multimodal,
        optimize_hyperparams=args.optimize,
        use_cache=not args.no_cache,
        max_workers=args.workers,
        enhanced_augmentation=not args.no_enhanced_augmentation,
        augmentation_factor=args.augmentation_factor,
        include_tess=args.include_tess
    )
    
    # Print results summary
    if isinstance(results, dict):
        print("\n=== Pipeline Results ===")
        print(f"Result directory: {results.get('result_dir')}")
        print(f"Files processed: {results.get('file_count', 0)}")
        print(f"Transits detected: {results.get('transit_count', 0)}")
        print(f"Report path: {results.get('report_path')}")
    else:
        print("\n=== Pipeline Failed ===")
        print("Check logs for more information.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
