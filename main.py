#!/usr/bin/env python3
"""
Main entry point for the enhanced exoplanet detection pipeline.
Now supports typed local data inputs for improved labeling.
Log file is now placed in the run-specific output directory.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import config
# --- THIS IS THE FIX ---
# Import setup_logging from its correct, new location
from utils.file_utils import setup_logging
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from data.data_fetcher import generate_sample_light_curves, fetch_exoplanet_labels
from data.real_data_fetcher import smart_data_fetcher


def main():
    """Main function to run the enhanced exoplanet detection pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced Exoplanet Detection Pipeline')
    
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Use synthetic data instead of real data.')
    parser.add_argument('--max-records', type=int, default=100,
                        help='Maximum number of targets to process if fetching new data or for synthetic generation.')
    parser.add_argument('--planets_dir', type=str,
                        help="Directory containing light curves for confirmed exoplanets.")
    parser.add_argument('--false_positives_dir', type=str,
                        help="Directory containing light curves for known false positives.")
    parser.add_argument('--verify-against-catalogs', action='store_true',
                        help='Verify detected transits against catalogs of known planets and false positives (runs after pipeline).')
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir_main_script = Path(config.RESULTS_DIR) / f"run_{timestamp}"
    output_dir_main_script.mkdir(parents=True, exist_ok=True)
    
    log_file_path = output_dir_main_script / 'main_pipeline.log'
    setup_logging(log_file_path=log_file_path)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Pipeline run initiated. Output directory: {output_dir_main_script}")
    logger.info(f"Command line arguments: {args}")

    if args.synthetic_data:
        logger.info("Generating synthetic data based on arguments.")
        typed_light_curve_files = generate_sample_light_curves(
            count=args.max_records,
            output_dir=output_dir_main_script / "synthetic_data"
        )
    else:
        logger.info("Using real data specified from directories.")
        typed_light_curve_files = smart_data_fetcher(
            confirmed_planet_dir=args.planets_dir,
            false_positive_dir=args.false_positives_dir
        )

    if not typed_light_curve_files:
        logger.warning("No light curve files found or generated. Exiting pipeline.")
        return 1

    pipeline_results = run_enhanced_pipeline(
        light_curve_files=typed_light_curve_files,
        output_dir_str=str(output_dir_main_script)
    )

    if args.verify_against_catalogs:
        logger.info("Verification step requested. Comparing results against known catalogs.")
        try:
            from validation.verifier import verify_results
            exoplanet_labels = fetch_exoplanet_labels(use_cache=True)
            verify_results(
                pipeline_results,
                exoplanet_labels,
                output_dir_main_script / 'verification'
            )
        except Exception as e:
            logger.error(f"Error during verification step: {e}", exc_info=True)

    logger.info("Pipeline execution attempt completed.")
    
    final_transit_count = pipeline_results.get('transit_count', 0) if pipeline_results else 0
    final_file_count = pipeline_results.get('successfully_processed_count', 0) if pipeline_results else 0

    print("\n====================================")
    print("    EXOPLANET DETECTION SUMMARY")
    print("====================================")
    print(f"Input files considered for processing: {len(typed_light_curve_files)}")
    print(f"Files successfully processed by pipeline: {final_file_count}")
    print(f"Total transit-like events detected across processed files: {final_transit_count}")
    if pipeline_results and pipeline_results.get('result_dir_actual'):
        print(f"Main results directory: {pipeline_results['result_dir_actual']}")
    if pipeline_results and pipeline_results.get('report_path'):
        print(f"Detailed HTML report: {pipeline_results['report_path']}")
    if args.verify_against_catalogs:
         print(f"Verification reports (if run) are in: {output_dir_main_script / 'verification'}")
    print("====================================\n")
    
    return 0

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    sys.exit(main())