#!/usr/bin/env python3
"""
Script for running large-scale exoplanet detection with real astronomical data.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import config
from pipeline.pipeline_runner import setup_logging
from pipeline.real_data_pipeline import run_real_data_pipeline
from data.real_data_fetcher import get_random_target_sample
from data.data_fetcher import fetch_kepler_false_positives


def main():
    """
    Main function to run large-scale exoplanet detection with real data.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run large-scale exoplanet detection with real data')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of light curves to sample')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=config.DEFAULT_MAX_WORKERS,
                       help='Number of parallel workers')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--no-rv', action='store_true',
                       help='Disable RV cross-validation')
    parser.add_argument('--target-list', type=str,
                       help='Path to file with target list (one ID per line)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable enhanced data augmentation')
    parser.add_argument('--augment-factor', type=int, default=config.AUGMENTATION_FACTOR,
                       help='Factor by which to augment positive examples')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for results')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--mission', type=str, choices=['Kepler', 'TESS', 'Both'],
                       default='Kepler', help='Mission data source')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting large-scale real data pipeline at {datetime.now()}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = config.RESULTS_DIR
    
    # Get target list if provided
    target_list = None
    if args.target_list:
        try:
            with open(args.target_list, 'r') as f:
                target_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(target_list)} targets from {args.target_list}")
        except Exception as e:
            logger.error(f"Error loading target list: {e}")
            return 1
    else:
        # Use random sample based on mission
        if args.mission == 'Both':
            # Combine targets from both missions
            kepler_targets = get_random_target_sample(args.sample_size // 2, mission="Kepler")
            tess_targets = get_random_target_sample(args.sample_size // 2, mission="TESS")
            target_list = kepler_targets + tess_targets
        else:
            target_list = get_random_target_sample(args.sample_size, mission=args.mission)
    
    # Run the pipeline
    results = run_real_data_pipeline(
        target_list=target_list,
        sample_size=args.sample_size,
        use_cache=not args.no_cache,
        include_rv_validation=not args.no_rv,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enhanced_augmentation=not args.no_augment,
        augmentation_factor=args.augment_factor
    )
    
    # Print results summary
    if isinstance(results, dict):
        print("\n=== Real Data Pipeline Results ===")
        print(f"Result directory: {results.get('result_dir')}")
        print(f"Files processed: {results.get('file_count', 0)}")
        print(f"Transits detected: {results.get('transit_count', 0)}")
        print(f"Report path: {results.get('report_path')}")
        
        # Print validation stats if available
        if results.get('rv_validation'):
            val_dir = os.path.join(results.get('result_dir', ''), 'validation')
            print(f"RV validation report: {val_dir}")
    else:
        print("\n=== Pipeline Failed ===")
        print("Check logs for more information.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
