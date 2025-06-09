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
import glob
import json
import pickle 
from datetime import datetime
from pathlib import Path 

import config 
# setup_logging will be called after output_dir is known
from pipeline.pipeline_runner import setup_logging 
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline # Assuming this is the correct location
from data.data_fetcher import generate_sample_light_curves, fetch_exoplanet_labels 
from data.real_data_fetcher import smart_data_fetcher 


def main():
    """Main function to run the enhanced exoplanet detection pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced Exoplanet Detection Pipeline')
    
    # Basic pipeline options
    parser.add_argument('--max-records', type=int, default=100, 
                        help='Maximum number of targets to process if fetching new data or for synthetic generation.')
    parser.add_argument('--verify-against-catalogs', action='store_true',
                        help='Verify detected transits against catalogs of known planets and false positives (runs after pipeline).')
    parser.add_argument('--synthetic-data', action='store_true',
                        help='Use synthetic data instead of real data.')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of downloaded data.')
    parser.add_argument('--multimodal', action='store_true', default=config.USE_MULTIMODAL,
                        help=f'Use multimodal fusion model (default: {config.USE_MULTIMODAL}).')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization (Note: can be time-consuming).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results. Defaults to ./results/pipeline_run_TIMESTAMP.')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default=logging.getLevelName(config.LOG_LEVEL), help=f'Logging level (default: {logging.getLevelName(config.LOG_LEVEL)}).')
    
    # Enhanced pipeline options
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='Use enhanced data augmentation during training.')
    parser.add_argument('--augmentation-factor', type=int, default=config.AUGMENTATION_FACTOR,
                        help=f'Factor by which to augment positive examples (default: {config.AUGMENTATION_FACTOR}).')
    parser.add_argument('--include-tess', action='store_true',
                        help='Include TESS data in the training set (if available and supported by fetchers).')
    parser.add_argument('--max-workers', type=int, default=config.DEFAULT_MAX_WORKERS,
                        help=f'Maximum number of parallel workers (default: {config.DEFAULT_MAX_WORKERS}).')
    parser.add_argument('--specific-targets', type=str, default=None,
                        help='Comma-separated list of specific Kepler/TESS IDs to analyze (marks them as type "unknown" unless also in specific dirs).')
    
    # Data source options for improved labeling
    # Ensure this argument is defined:
    parser.add_argument('--local-data-dir', type=str, default=None,
                        help='Directory containing mixed Kepler/TESS data files. Pipeline will attempt to categorize.')
    parser.add_argument('--planet-data-dir', type=str, default=None,
                        help=f"Directory containing light curves of confirmed planets (files labeled '{config.FILE_TYPE_CONFIRMED_PLANET}').")
    parser.add_argument('--fp-data-dir', type=str, default=None,
                        help=f"Directory containing light curves of known false positives (files labeled '{config.FILE_TYPE_FALSE_POSITIVE}').")

    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_dir_main_script = Path(args.output_dir) 
    else:
        results_base_dir = Path(config.RESULTS_DIR) if hasattr(config, 'RESULTS_DIR') else Path.cwd() / "results"
        output_dir_main_script = results_base_dir / f"pipeline_run_{timestamp}"
    
    output_dir_main_script.mkdir(parents=True, exist_ok=True)
    
    log_level_numeric = getattr(logging, args.log_level.upper(), config.LOG_LEVEL)
    run_specific_log_file = output_dir_main_script / "pipeline.log" 
    
    setup_logging(log_level_to_set=log_level_numeric, log_file_path_override=run_specific_log_file) 
    
    logger = logging.getLogger(__name__) 
    logger.info(f"Starting exoplanet detection pipeline at {timestamp} with args: {args}")
    logger.info(f"Full results, including this log, will be saved to: {output_dir_main_script}")

    typed_light_curve_files = [] 

    if args.planet_data_dir:
        planet_data_path = Path(args.planet_data_dir)
        if planet_data_path.is_dir():
            logger.info(f"Processing confirmed planets directory: {planet_data_path}")
            planet_files = list(planet_data_path.glob("*.fits")) 
            for f_path in planet_files:
                typed_light_curve_files.append((str(f_path), config.FILE_TYPE_CONFIRMED_PLANET))
            logger.info(f"Found {len(planet_files)} files in confirmed planets directory.")
        else:
            logger.warning(f"Planet data directory not found: {planet_data_path}")
    
    if args.fp_data_dir:
        fp_data_path = Path(args.fp_data_dir)
        if fp_data_path.is_dir():
            logger.info(f"Processing false positives directory: {fp_data_path}")
            fp_files = list(fp_data_path.glob("*.fits"))
            for f_path in fp_files:
                typed_light_curve_files.append((str(f_path), config.FILE_TYPE_FALSE_POSITIVE))
            logger.info(f"Found {len(fp_files)} files in false positives directory.")
        else:
            logger.warning(f"False positive data directory not found: {fp_data_path}")

    # This is the block where the error occurs if args.local_data_dir is not an attribute
    if args.local_data_dir: 
        local_data_path = Path(args.local_data_dir)
        if local_data_path.is_dir():
            logger.info(f"Processing general local data directory: {local_data_path}")
            general_files = list(local_data_path.glob("*.fits"))
            for f_path_obj in general_files:
                f_path = str(f_path_obj)
                is_already_added = any(f_path == existing_f[0] for existing_f in typed_light_curve_files)
                if not is_already_added:
                    fn_lower = f_path_obj.name.lower()
                    if "false" in fn_lower or "fp" in fn_lower or "eb" in fn_lower: 
                        typed_light_curve_files.append((f_path, config.FILE_TYPE_FALSE_POSITIVE))
                        logger.debug(f"Categorized {f_path} from local_data_dir as FALSE_POSITIVE based on name.")
                    elif "planet" in fn_lower or "confirmed" in fn_lower or "pc" in fn_lower: 
                        typed_light_curve_files.append((f_path, config.FILE_TYPE_CONFIRMED_PLANET))
                        logger.debug(f"Categorized {f_path} from local_data_dir as CONFIRMED_PLANET based on name.")
                    else:
                        typed_light_curve_files.append((f_path, config.FILE_TYPE_UNKNOWN))
                        logger.debug(f"Categorized {f_path} from local_data_dir as UNKNOWN based on name.")
            logger.info(f"Processed {len(general_files)} files from general local directory.")
        else:
            logger.warning(f"Local data directory not found: {local_data_path}")
    
    unique_typed_files_dict = {}
    for f_path, f_type in typed_light_curve_files:
        if f_path not in unique_typed_files_dict: 
            unique_typed_files_dict[f_path] = f_type
    typed_light_curve_files = list(unique_typed_files_dict.items())
    logger.info(f"Total unique local files collected: {len(typed_light_curve_files)}")
    if typed_light_curve_files:
        from collections import Counter
        type_counts = Counter(ftype for _, ftype in typed_light_curve_files)
        logger.info(f"Breakdown of locally collected file types: {dict(type_counts)}")

    if args.synthetic_data:
        logger.info(f"Generating {args.max_records} synthetic light curves for pipeline.")
        synthetic_output_target_dir = Path(config.LIGHT_CURVE_DIR) / "synthetic" if hasattr(config, 'LIGHT_CURVE_DIR') else Path.cwd() / "data_files" / "light_curves" / "synthetic"
        
        synthetic_files_info = generate_sample_light_curves(
            num_samples=args.max_records, 
            output_dir=synthetic_output_target_dir 
        ) 
        if synthetic_files_info: 
            for f_path, label in synthetic_files_info: 
                logger.debug(f"Processing synthetic file: {f_path}, Original label from generator: {label} (type: {type(label)})")
                file_type = config.FILE_TYPE_SYNTHETIC_PLANET if int(label) == 1 else config.FILE_TYPE_SYNTHETIC_NOISE
                logger.debug(f"Assigned file_type: {file_type} for label {label}")
                typed_light_curve_files.append((f_path, file_type))
            logger.info(f"Added {len(synthetic_files_info)} synthetic files to processing list.")
        else:
            logger.warning("generate_sample_light_curves returned no files.")
    
    specific_target_list_ids = None
    if args.specific_targets:
        specific_target_list_ids = [target.strip() for target in args.specific_targets.split(',')]
        logger.info(f"Specific target IDs to query if fetching: {specific_target_list_ids}")

    if not typed_light_curve_files and not args.synthetic_data: 
        logger.info("No local/synthetic files. Attempting to fetch data from MAST...")
        fetched_files_paths = []
        if specific_target_list_ids:
            logger.info(f"Fetching for specific targets: {specific_target_list_ids}")
            # fetched_files_paths = smart_data_fetcher(target_list=specific_target_list_ids, ...)
        else:
            logger.info(f"Fetching general targets up to {args.max_records}")
            # fetched_files_paths = smart_data_fetcher(max_targets=args.max_records, ...)
        
        for f_path in fetched_files_paths:
            typed_light_curve_files.append((f_path, config.FILE_TYPE_UNKNOWN))
        if fetched_files_paths:
            logger.info(f"Fetched {len(fetched_files_paths)} files from MAST.")
        else:
            logger.info("No files fetched from MAST in this run.")
    
    if not typed_light_curve_files:
        logger.error("No light curve files available to process (local, synthetic, or fetched). Exiting pipeline.")
        return 1 

    logger.info(f"Final list of {len(typed_light_curve_files)} files to be processed by the pipeline:")
    for f_idx, (f_path, f_type) in enumerate(typed_light_curve_files):
        logger.debug(f"  {f_idx+1}. Path: {f_path}, Type: {f_type}")

    processed_files_log_path = output_dir_main_script / "files_for_processing_log.json"
    try:
        with open(processed_files_log_path, 'w') as f:
            json_serializable_typed_files = [(str(f_path), f_type) for f_path, f_type in typed_light_curve_files]
            json.dump(json_serializable_typed_files, f, indent=2)
        logger.info(f"List of files selected for processing saved to: {processed_files_log_path}")
    except Exception as e:
        logger.error(f"Could not save files_for_processing_log.json: {e}")

    exoplanet_labels_df = None
    try:
        exoplanet_labels_df = fetch_exoplanet_labels(use_cache=not args.no_cache)
        if exoplanet_labels_df is not None:
            logger.info(f"Successfully fetched/loaded {len(exoplanet_labels_df)} exoplanet catalog entries.")
        else:
            logger.warning("fetch_exoplanet_labels returned None or an empty DataFrame.")
    except Exception as e:
        logger.error(f"Could not fetch exoplanet labels: {e}. Labeling for 'unknown' files may be impacted.", exc_info=True)
        
    pipeline_results = run_enhanced_pipeline(
        light_curve_files_with_types=typed_light_curve_files,
        exoplanet_labels=exoplanet_labels_df, 
        use_synthetic=args.synthetic_data, 
        use_multimodal=args.multimodal,
        optimize_hyperparams=args.optimize,
        use_cache=not args.no_cache, 
        max_workers=args.max_workers,
        enhanced_augmentation=args.enhanced_augmentation,
        augmentation_factor=args.augmentation_factor,
        include_tess=args.include_tess,
        output_dir_base=output_dir_main_script 
    )
    
    if args.verify_against_catalogs and pipeline_results and pipeline_results.get('processed_results'):
        try:
            from validation.cross_validator import verify_detections_against_catalogs 
            logger.info("Verifying detections against exoplanet catalogs...")
            verification_output_dir = output_dir_main_script / "verification"
            verification_output_dir.mkdir(parents=True, exist_ok=True)
            
            verification_summary = verify_detections_against_catalogs(
                pipeline_results['processed_results'], 
                output_dir=str(verification_output_dir) 
            )
            
            verification_path = verification_output_dir / "verification_summary_object.pkl"
            with open(verification_path, 'wb') as f:
                pickle.dump(verification_summary, f)
            logger.info(f"Verification summary object saved to {verification_path}")
            if verification_summary and 'verification_stats' in verification_summary:
                 logger.info(f"Verification stats: {verification_summary.get('verification_stats')}")
            else:
                logger.warning("Verification summary or stats missing after verification step.")
        except ImportError:
            logger.error("Could not import 'verify_detections_against_catalogs' from validation.cross_validator. Skipping verification.")
        except Exception as e:
            logger.error(f"Error during verification step: {e}", exc_info=True)

    logger.info("Pipeline execution attempt completed.") 
    if pipeline_results and pipeline_results.get('result_dir_actual'): 
        logger.info(f"Actual results directory from pipeline runner: {pipeline_results['result_dir_actual']}")
    
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
    sys.exit(main())
