#!/usr/bin/env python3
"""
Enhanced main entry point for the exoplanet detection pipeline with support for
multiple directories containing confirmed planets and false positives.
"""

import os
import sys
import logging
import argparse
import glob
import json
from datetime import datetime

# Import required functions
from pipeline.pipeline_runner import setup_logging
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from data.enhanced_data_fetcher import run_enhanced_data_fetcher, generate_sample_light_curves

def main():
    """Main function to run the enhanced exoplanet detection pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Enhanced Exoplanet Detection Pipeline')
    
    # Basic pipeline options
    parser.add_argument('--max-records', type=int, default=20,
                        help='Maximum number of targets to process')
    parser.add_argument('--verify-against-catalogs', action='store_true',
                        help='Verify detected transits against catalogs of known planets and false positives')
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Use synthetic data instead of real data')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of downloaded data')
    parser.add_argument('--multimodal', action='store_true',
                        help='Use multimodal fusion model')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced pipeline with improved models and data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    
    # Enhanced pipeline options
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    parser.add_argument('--augmentation-factor', type=int, default=3,
                        help='Factor by which to augment positive examples')
    parser.add_argument('--include-tess', action='store_true',
                        help='Include TESS data in the training set')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--specific-targets', type=str, default=None,
                        help='Comma-separated list of specific Kepler IDs to analyze')
    
    # Multiple directory options
    parser.add_argument('--local-data-dir', type=str, default=None,
                        help='Directory containing mixed Kepler data files')
    parser.add_argument('--planet-data-dir', type=str, default=None,
                        help='Directory containing confirmed planet light curves')
    parser.add_argument('--fp-data-dir', type=str, default=None,
                        help='Directory containing false positive light curves')
    parser.add_argument('--include-false-positives', action='store_true',
                        help='Include known false positives in processing')
                        
    args = parser.parse_args()
    
    # Initialize target_list
    target_list = None
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger("__main__")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting exoplanet detection pipeline at {timestamp}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        from config import RESULTS_DIR
        output_dir = os.path.join(RESULTS_DIR, f"pipeline_run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Process specific targets if provided
    if args.specific_targets:
        target_list = [target.strip() for target in args.specific_targets.split(',')]
        logger.info(f"Using {len(target_list)} specific targets: {target_list}")
    
    # Simple function to categorize FITS files
    def categorize_fits_files(directory):
        """Simple function to separate files into likely planets and false positives"""
        all_files = glob.glob(os.path.join(directory, "*.fits"))
        
        # Try to do basic classification by filename
        confirmed_planets = []
        false_positives = []
        unknown = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path).lower()
            
            # Check for obvious false positive markers in filename
            if 'false' in filename or 'fp' in filename:
                false_positives.append(file_path)
            # Check for obvious planet markers
            elif 'confirmed' in filename or 'planet' in filename:
                confirmed_planets.append(file_path)
            else:
                # Default to unknown
                unknown.append(file_path)
        
        return {
            'confirmed_planets': confirmed_planets,
            'false_positives': false_positives,
            'unknown': unknown
        }
    
    # Check if using local data from one or more directories
    if args.local_data_dir or args.planet_data_dir or args.fp_data_dir:
        logger.info("Using local Kepler data")
        
        # Initialize lists
        confirmed_files = []
        false_positive_files = []
        unknown_files = []
        
        # Process the general directory if provided
        if args.local_data_dir:
            logger.info(f"Processing general directory: {args.local_data_dir}")
            categorized_files = categorize_fits_files(args.local_data_dir)
            confirmed_files.extend(categorized_files['confirmed_planets'])
            false_positive_files.extend(categorized_files['false_positives'])
            unknown_files.extend(categorized_files['unknown'])
        
        # Process the confirmed planets directory if provided
        if args.planet_data_dir:
            logger.info(f"Processing confirmed planets directory: {args.planet_data_dir}")
            planet_files = glob.glob(os.path.join(args.planet_data_dir, "*.fits"))
            confirmed_files.extend(planet_files)
            logger.info(f"Added {len(planet_files)} files from confirmed planets directory")
        
        # Process the false positives directory if provided
        if args.fp_data_dir:
            logger.info(f"Processing false positives directory: {args.fp_data_dir}")
            fp_files = glob.glob(os.path.join(args.fp_data_dir, "*.fits"))
            false_positive_files.extend(fp_files)
            logger.info(f"Added {len(fp_files)} files from false positives directory")
        
        # Log the results
        logger.info(f"Found {len(confirmed_files)} confirmed planets")
        logger.info(f"Found {len(false_positive_files)} false positives")
        logger.info(f"Found {len(unknown_files)} uncategorized files")
        
        # Determine which files to process
        if args.include_false_positives:
            # Use all files
            light_curve_files = confirmed_files + false_positive_files + unknown_files
            logger.info(f"Processing all {len(light_curve_files)} files including false positives")
        else:
            # Skip false positives
            light_curve_files = confirmed_files + unknown_files
            logger.info(f"Processing {len(light_curve_files)} files, excluding {len(false_positive_files)} false positives")
        
        # Save the file lists for reference
        with open(os.path.join(output_dir, "file_categorization.json"), 'w') as f:
            json.dump({
                'confirmed_planets': [os.path.basename(f) for f in confirmed_files],
                'false_positives': [os.path.basename(f) for f in false_positive_files],
                'unknown': [os.path.basename(f) for f in unknown_files],
                'files_processed': [os.path.basename(f) for f in light_curve_files]
            }, f, indent=2)
        
        # Run pipeline with local files
        results = run_enhanced_pipeline(
            light_curve_files=light_curve_files,
            use_synthetic=False,
            use_multimodal=args.multimodal,
            optimize_hyperparams=args.optimize,
            max_workers=args.max_workers,
            enhanced_augmentation=args.enhanced_augmentation,
            augmentation_factor=args.augmentation_factor,
            include_tess=args.include_tess
        )
    
    # Using synthetic data option
    elif args.synthetic_data:
        logger.info("Using synthetic data for pipeline")
        # Generate synthetic light curves
        light_curve_files = generate_sample_light_curves(num_samples=args.max_records)
        
        # Run pipeline with synthetic data
        results = run_enhanced_pipeline(
            light_curve_files=light_curve_files,
            use_synthetic=True,
            use_multimodal=args.multimodal,
            optimize_hyperparams=args.optimize,
            max_workers=args.max_workers,
            enhanced_augmentation=args.enhanced_augmentation,
            augmentation_factor=args.augmentation_factor,
            include_tess=args.include_tess
        )
    
    # Using regular downloaded data
    else:
        # Use enhanced data fetcher with direct download from Kepler archive
        logger.info(f"Fetching Kepler data (max_targets={args.max_records})")
        light_curve_files, exoplanet_labels = run_enhanced_data_fetcher(
            max_targets=args.max_records,
            use_cache=not args.no_cache,
            max_workers=args.max_workers,
            target_list=target_list
        )
        
        # Run pipeline with the fetched data
        results = run_enhanced_pipeline(
            light_curve_files=light_curve_files,
            exoplanet_labels=exoplanet_labels,
            use_synthetic=False,
            use_multimodal=args.multimodal,
            optimize_hyperparams=args.optimize,
            use_cache=not args.no_cache,
            max_workers=args.max_workers,
            enhanced_augmentation=args.enhanced_augmentation,
            augmentation_factor=args.augmentation_factor,
            include_tess=args.include_tess
        )
    # After your pipeline runs and produces results
# (somewhere near where you're generating reports)
    if args.verify_against_catalogs:
        from validation.cross_validator import verify_detections_against_catalogs
        logger.info("Verifying detections against exoplanet catalogs")
        verification_results = verify_detections_against_catalogs(results, output_dir='result_dir')
        # Optionally, you could save these results:
        verification_path = os.path.join('result_dir', "verification_results.pkl")
        with open(verification_path, 'wb') as f:
            pickle.dump(verification_results, f)
        logger.info(f"Verification results saved to {verification_path}")
        logger.info("Pipeline completed successfully")

    if results and 'result_dir' in results:
        logger.info(f"Results saved to: {results['result_dir']}")
    
    # Print summary to console
    transit_count = results.get('transit_count', 0) if results else 0
    file_count = results.get('file_count', 0) if results else 0
    
    print("\n====================================")
    print("    EXOPLANET DETECTION SUMMARY")
    print("====================================")
    print(f"Files processed: {file_count}")
    print(f"Transits detected: {transit_count}")
    if results and 'result_dir' in results:
        print(f"Results directory: {results['result_dir']}")
    if results and 'report_path' in results:
        print(f"Detailed report: {results['report_path']}")
    print("====================================\n")
    
    return results


if __name__ == "__main__":
    main()