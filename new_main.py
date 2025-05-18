"""
Main entry point for the enhanced exoplanet detection pipeline.
"""

import argparse
import logging
import os
from datetime import datetime

# Import required functions
from pipeline.pipeline_runner import run_pipeline
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from data.data_fetcher import fetch_kepler_data, fetch_exoplanet_labels, download_light_curves, generate_sample_light_curves


def setup_logging():
    """Configure logging for the application."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"pipeline_log_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Exoplanet Detection Pipeline')
    
    # Basic pipeline options
    parser.add_argument('--max-records', type=int, default=100,
                        help='Maximum number of records to process')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of downloaded data')
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Use synthetic data instead of real data')
    parser.add_argument('--multimodal', action='store_true',
                        help='Use multimodal fusion model')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced pipeline with improved models and data')
    parser.add_argument('--sample-size', type=int, default=1500,
                        help='Number of synthetic samples to generate')
    
    # Enhanced pipeline options
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='Use enhanced data augmentation')
    parser.add_argument('--augmentation-factor', type=int, default=3,
                        help='Factor by which to augment positive examples')
    parser.add_argument('--include-tess', action='store_true',
                        help='Include TESS data in the training set')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--include-verified-data', action='store_true',
                        help='Include verified transit/non-transit data')
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Starting exoplanet detection pipeline")
    
    # Configure pipeline options
    use_cache = not args.no_cache
    
    if args.enhanced:
        logger.info("Using enhanced pipeline with improved models and data")
        
        if args.synthetic_data:
            logger.info("Using synthetic data for pipeline")
            results = run_enhanced_pipeline(
                use_synthetic=True,
                use_multimodal=args.multimodal,
                optimize_hyperparams=args.optimize,
                max_workers=args.max_workers,
                enhanced_augmentation=args.enhanced_augmentation,
                augmentation_factor=args.augmentation_factor,
                include_tess=args.include_tess,
                include_verified_data=args.include_verified_data
            )
            
        else:
            # Try to fetch data from MAST
            try:
                logger.info(f"Fetching Kepler data (max_records={args.max_records})")
                obs_table = fetch_kepler_data(max_records=args.max_records, use_cache=use_cache)
                
                logger.info("Fetching exoplanet labels")
                exoplanet_labels = fetch_exoplanet_labels(use_cache=use_cache)
                
                # Try to download light curves
                light_curve_files = download_light_curves(obs_table, use_cache, args.max_workers)
                
                # If no light curves found, use synthetic data
                if not light_curve_files:
                    logger.warning("No light curves found in MAST query, using synthetic data instead")
                    
                    # Generate synthetic light curves
                    light_curve_files = generate_sample_light_curves(num_samples=20)
                    
                    # Run pipeline with synthetic files
                    results = run_enhanced_pipeline(
                        light_curve_files=light_curve_files,
                        exoplanet_labels=exoplanet_labels,
                        use_synthetic=False,
                        use_multimodal=args.multimodal,
                        optimize_hyperparams=args.optimize,
                        use_cache=use_cache,
                        max_workers=args.max_workers,
                        enhanced_augmentation=args.enhanced_augmentation,
                        augmentation_factor=args.augmentation_factor,
                        include_tess=args.include_tess,
                        include_verified_data=args.include_verified_data
                    )
                else:
                    # Run enhanced pipeline with real data
                    results = run_enhanced_pipeline(
                        light_curve_files=light_curve_files,
                        exoplanet_labels=exoplanet_labels,
                        use_synthetic=False,
                        use_multimodal=args.multimodal,
                        optimize_hyperparams=args.optimize,
                        use_cache=use_cache,
                        max_workers=args.max_workers,
                        enhanced_augmentation=args.enhanced_augmentation,
                        augmentation_factor=args.augmentation_factor,
                        include_tess=args.include_tess,
                        include_verified_data=args.include_verified_data
                    )
            except Exception as e:
                logger.error(f"Error fetching data from MAST: {e}")
                logger.info("Falling back to synthetic data")
                
                # Generate synthetic light curves
                light_curve_files = generate_sample_light_curves(num_samples=20)
                
                # Still try to get exoplanet labels if possible
                try:
                    exoplanet_labels = fetch_exoplanet_labels(use_cache=use_cache)
                except Exception as label_error:
                    logger.error(f"Error fetching exoplanet labels: {label_error}")
                    exoplanet_labels = None
                
                # Run pipeline with synthetic files
                results = run_enhanced_pipeline(
                    light_curve_files=light_curve_files,
                    exoplanet_labels=exoplanet_labels,
                    use_synthetic=False,
                    use_multimodal=args.multimodal,
                    optimize_hyperparams=args.optimize,
                    use_cache=use_cache,
                    max_workers=args.max_workers,
                    enhanced_augmentation=args.enhanced_augmentation,
                    augmentation_factor=args.augmentation_factor,
                    include_tess=args.include_tess,
                    include_verified_data=args.include_verified_data
                )
    else:
        logger.info("Using standard pipeline")
        
        if args.synthetic_data:
            logger.info("Using synthetic data for pipeline")
            results = run_pipeline(
                use_synthetic=True,
                use_multimodal=args.multimodal,
                optimize_hyperparams=args.optimize
            )
        else:
            # Fetch data
            logger.info(f"Fetching Kepler data (max_records={args.max_records})")
            obs_table = fetch_kepler_data(max_records=args.max_records, use_cache=use_cache)
            
            logger.info("Fetching exoplanet labels")
            exoplanet_labels = fetch_exoplanet_labels(use_cache=use_cache)
            
            # Run pipeline
            results = run_pipeline(
                obs_table=obs_table,
                exoplanet_labels=exoplanet_labels,
                use_multimodal=args.multimodal,
                optimize_hyperparams=args.optimize,
                use_cache=use_cache
            )
    
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
