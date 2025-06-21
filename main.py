# FILE: main.py (Corrected and Final Version)

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
import pandas as pd

# Import project-level configuration and pipeline components
import config
from pipeline.pipeline_runner import setup_logging
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from data.data_fetcher import (
    generate_sample_light_curves,
    get_target_lists_from_archive,
    download_light_curves,
    fetch_exoplanet_labels
)

# Set up a logger for this module
logger = logging.getLogger(__name__)

def _gather_input_files(args):
    """
    Gathers input files by either generating synthetic data or fetching real,
    labeled data from the NASA Exoplanet Archive.
    """
    typed_light_curve_files = []

    # Priority 1: Use synthetic data if the flag is explicitly set
    if args.synthetic_data:
        logger.info(f"Generating {args.max_records} synthetic light curves as requested.")
        synthetic_files = generate_sample_light_curves(num_samples=args.max_records)
        for f_path, label in synthetic_files:
            file_type = config.FILE_TYPE_SYNTHETIC_PLANET if label == 1 else config.FILE_TYPE_SYNTHETIC_NOISE
            typed_light_curve_files.append((f_path, file_type))

    # Priority 2: Fetch real data from MAST if no synthetic data is generated
    else:
        logger.info("No synthetic data specified. Fetching real, labeled data from MAST archives...")
        
        num_per_class = args.max_records // 2
        if num_per_class == 0: num_per_class = 1
        
        confirmed_targets, fp_targets = get_target_lists_from_archive(num_per_class=num_per_class)
        
        if confirmed_targets:
            logger.info(f"Downloading {len(confirmed_targets)} confirmed planet light curves...")
            confirmed_files = download_light_curves(confirmed_targets)
            for f_path in confirmed_files:
                typed_light_curve_files.append((f_path, config.FILE_TYPE_CONFIRMED_PLANET))

        if fp_targets:
            logger.info(f"Downloading {len(fp_targets)} false positive light curves...")
            fp_files = download_light_curves(fp_targets)
            for f_path in fp_files:
                typed_light_curve_files.append((f_path, config.FILE_TYPE_FALSE_POSITIVE))

    # De-duplicate the final list and log the results
    unique_files_dict = {f_path: f_type for f_path, f_type in typed_light_curve_files}
    unique_files_list = list(unique_files_dict.items())
    
    logger.info(f"Total unique files collected for processing: {len(unique_files_list)}")
    if unique_files_list:
        type_counts = Counter(ftype for _, ftype in unique_files_list)
        logger.info(f"Breakdown of collected file types: {dict(type_counts)}")
        
    return unique_files_list


def main():
    """Main function to run the exoplanet detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Exoplanet Detection Pipeline using lightkurve and TensorFlow."
    )
    
    parser.add_argument('--max-records', type=int, default=10,
                        help="Total number of real records to fetch (half confirmed, half false positive), or number of synthetic records.")
    parser.add_argument('--synthetic-data', action='store_true',
                        help="Generate and use synthetic data instead of fetching real data.")
    # Add other arguments here as needed
    
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.RESULTS_DIR) / f"pipeline_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file_path_override=output_dir / "pipeline.log")
    
    logger.info(f"Pipeline run started. Results will be saved to: {output_dir}")
    logger.info(f"Running with arguments: {args}")

    # Gather files
    typed_light_curve_files = _gather_input_files(args)
    if not typed_light_curve_files:
        logger.error("No light curve files were found or generated. Exiting pipeline.")
        return 1

    # Fetch labels (can be an empty DataFrame if fetch fails)
    exoplanet_labels_df = fetch_exoplanet_labels()

    # Run pipeline
    run_enhanced_pipeline(
        light_curve_files_with_types=typed_light_curve_files,
        exoplanet_labels=exoplanet_labels_df, # <-- THIS IS THE CORRECTED LINE
        output_dir_base=output_dir
        # Pass other args from your parser to the pipeline as needed, e.g.:
        # enhanced_augmentation=args.enhanced_augmentation
    )

    print("\n--- ✅ PIPELINE FINISHED ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())

