# enhanced_main.py

import logging
import config
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from utils.file_utils import setup_logging
from data.real_data_fetcher import smart_data_fetcher # Import smart_data_fetcher
from datetime import datetime # Import datetime for timestamp

def main():
    """
    Main function to run the enhanced exoplanet detection pipeline.
    """
    setup_logging(config.DEFAULT_LOG_FILE)

    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced exoplanet detection pipeline.")

    # Prepare light curve files
    all_light_curve_files = smart_data_fetcher(config.CONFIRMED_PLANETS_DIR, config.FALSE_POSITIVES_DIR)

    # --- Subset for testing ---
    confirmed_files = [f for f in all_light_curve_files if f['type'] == 'confirmed_planet']
    false_positive_files = [f for f in all_light_curve_files if f['type'] == 'false_positive']

    # Take a small subset of each for testing
    subset_size = 10
    light_curve_files = confirmed_files[:subset_size] + false_positive_files[:subset_size]
    logger.info(f"Running pipeline with a subset of {len(light_curve_files)} files for testing.")
    # --- End subset for testing ---

    # Generate a unique output directory string
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir_str = str(config.RESULTS_DIR / f"run_{timestamp}")

    # Run the main pipeline with the correct arguments
    run_enhanced_pipeline(light_curve_files, output_dir_str)

    logger.info("Enhanced exoplanet detection pipeline finished.")

if __name__ == "__main__":
    main()