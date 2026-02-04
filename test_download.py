# save this as test_download.py in your project's root directory

import os
import shutil
from datetime import datetime, timedelta
import logging
import sys

# Add the project root to the system path to allow importing discovery_stack
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from discovery_stack.registry import TargetRegistry
from discovery_stack.sentinel import DataSentinel, _download_file_direct # Import _download_file_direct for potential mocking/testing if needed

# --- Setup Logger for this test script ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Changed to DEBUG for more verbose output
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# ---

# --- Configuration for the test ---
TEST_DOWNLOAD_DIR = "test_downloads_temp/"
TEST_REGISTRY_FILE = "test_processed_targets.json"
MAX_DISK_USAGE_PCT = 85.0
TEST_MISSIONS = ["TESS", "KEPLER", "K2"] # Test with all supported missions
START_DATE_DAYS_AGO = 365 # Look for data from the last 1 year (365 days)
MAX_DOWNLOADS_PER_RUN = 20 # Limit the number of downloads for a quick test
# ---

def run_download_test():
    logger.info("Starting download test (querying MAST and performing direct downloads)...")

    # Clean up previous test runs if they exist
    if os.path.exists(TEST_DOWNLOAD_DIR):
        shutil.rmtree(TEST_DOWNLOAD_DIR)
        logger.info(f"Cleaned up previous {TEST_DOWNLOAD_DIR}")
    if os.path.exists(TEST_REGISTRY_FILE):
        os.remove(TEST_REGISTRY_FILE)
        logger.info(f"Cleaned up previous {TEST_REGISTRY_FILE}")

    os.makedirs(TEST_DOWNLOAD_DIR, exist_ok=True)
    logger.info(f"Created test download directory: {TEST_DOWNLOAD_DIR}")

    try:
        # Initialize components
        registry = TargetRegistry(TEST_REGISTRY_FILE)
        # DataSentinel now initialized with download_dir and max_disk_usage_pct
        sentinel = DataSentinel(registry, TEST_DOWNLOAD_DIR, MAX_DISK_USAGE_PCT)

        start_date_str = (datetime.now() - timedelta(days=START_DATE_DAYS_AGO)).strftime("%Y-%m-%d")

        logger.info(f"Attempting to poll MAST for {TEST_MISSIONS} data starting on or after {start_date_str} and download up to {MAX_DOWNLOADS_PER_RUN} files...")
        
        sentinel.poll_mast_and_download(
            missions=TEST_MISSIONS,
            start_date=start_date_str,
            max_downloads=MAX_DOWNLOADS_PER_RUN
        )
        logger.info("MAST polling and direct download attempt complete.")

        # Verify if any files were downloaded by checking the directory
        downloaded_files_in_dir = [f for f in os.listdir(TEST_DOWNLOAD_DIR) if f.endswith('.fits')]
        if downloaded_files_in_dir:
            logger.info(f"Successfully downloaded {len(downloaded_files_in_dir)} .fits files to {TEST_DOWNLOAD_DIR}")
            for f in downloaded_files_in_dir:
                logger.info(f"  - {f}")
        else:
            logger.warning(f"No .fits files were downloaded to {TEST_DOWNLOAD_DIR}. This might be due to: \n" 
                           f"  - No new data available on MAST for {TEST_MISSIONS} since {start_date_str}.\n"
                           f"  - Files already present in the test registry (though it was cleared).\n"
                           f"  - Disk space limit reached (check logs for 'Disk space limit reached').\n"
                           f"  - An issue during the query or direct download (check logs for errors).")
            logger.warning("Consider adjusting START_DATE_DAYS_AGO or MAX_DOWNLOADS_PER_RUN if you expect more results.")

        # Optionally, you can also use sentinel.check_for_new_data to see what it finds
        targets_found_by_sentinel = list(sentinel.check_for_new_data(TEST_DOWNLOAD_DIR))
        if targets_found_by_sentinel:
            logger.info(f"DataSentinel found {len(targets_found_by_sentinel)} new targets in {TEST_DOWNLOAD_DIR} for processing:")
            for target in targets_found_by_sentinel:
                logger.info(f"  - {target.source_file} (Mission: {target.mission}, ID: {target.target_id}, Sector: {target.sector})")
        else:
            logger.warning("DataSentinel found no new targets in the test download directory.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
    finally:
        logger.info("Download test finished.")
        # Optional: Keep the downloaded files for inspection, or uncomment to clean up
        # if os.path.exists(TEST_DOWNLOAD_DIR):
        #     shutil.rmtree(TEST_DOWNLOAD_DIR)
        #     logger.info(f"Cleaned up {TEST_DOWNLOAD_DIR}")
        # if os.path.exists(TEST_REGISTRY_FILE):
        #     os.remove(TEST_REGISTRY_FILE)
        #     logger.info(f"Cleaned up {TEST_REGISTRY_FILE}")


if __name__ == "__main__":
    run_download_test()