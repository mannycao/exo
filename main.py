# main.py

import argparse
import logging
import sys
from pathlib import Path

# --- Correct, Modern Imports ---
import config
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from utils.file_utils import setup_logging

def main():
    """
    The single, consolidated entry point for the Exoplanet Detection Pipeline.
    It handles command-line arguments and launches the enhanced pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the Exoplanet Detection Pipeline.")
    # Add command-line arguments to allow specifying different data sources
    parser.add_argument(
        '--planets_dir',
        type=str,
        help="Directory containing light curves for confirmed exoplanets."
    )
    parser.add_argument(
        '--false_positives_dir',
        type=str,
        help="Directory containing light curves for known false positives."
    )
    args = parser.parse_args()

    # --- Use the modern setup from enhanced_main.py ---
    # Setup logging using the settings from the config module
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting the Exoplanet Detection Pipeline.")

    # Pass the config and any command-line arguments to the pipeline
    # Note: The current mock-data pipeline doesn't use the args,
    # but they are ready for when you switch back to real data.
    run_enhanced_pipeline(config, args)

    logger.info("Exoplanet Detection Pipeline has finished.")


if __name__ == "__main__":
    # Ensure the project root is in the system path for clean imports
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    main()