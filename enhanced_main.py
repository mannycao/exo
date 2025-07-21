# enhanced_main.py

import logging
import config
from pipeline.enhanced_pipeline_runner import run_enhanced_pipeline
from utils.file_utils import setup_logging  # This import will now succeed

def main():
    """
    Main function to run the enhanced exoplanet detection pipeline.
    """
    # Setup logging using the settings from the config module
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced exoplanet detection pipeline.")

    # Run the main pipeline, passing the config module to it
    run_enhanced_pipeline(config)

    logger.info("Enhanced exoplanet detection pipeline finished.")

if __name__ == "__main__":
    main()