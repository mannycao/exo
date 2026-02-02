import logging
import os
import re
import time
import subprocess

import config as lite_config
from data.enhanced_data_fetcher import download_kepler_light_curve, generate_kepler_target_list

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# State file to keep track of processed files
STATE_FILE = os.path.join(lite_config.RESULTS_DIR, "processed_files.log")

def get_processed_files():
    """Reads the state file and returns a set of processed file paths."""
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, 'r') as f:
        return set(line.strip() for line in f)

def mark_file_as_processed(file_path):
    """Appends a file path to the state file."""
    with open(STATE_FILE, 'a') as f:
        f.write(f"{file_path}\n")

def get_tic_id_from_filename(filename):
    """Extracts the KIC/TIC ID from a Kepler FITS filename."""
    # Example: kplr011904151-2009350155506_llc.fits -> 11904151
    match = re.search(r'kplr(\d{9})', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    """
    Main function to run the continuous scanning pipeline.
    """
    logger.info("--- Starting Continuous Discovery Pipeline Scan ---")

    # For demonstration, ensure some data exists.
    # In a real scenario, an external process would be adding new files.
    logger.info("Checking for lightcurve data. Downloading a sample set if needed.")
    target_list = generate_kepler_target_list(num_targets=5)
    download_kepler_targets(target_list, max_workers=2)

    # Get all lightcurve files from the data directory
    lightcurve_dir = lite_config.LIGHT_CURVE_DIR
    all_files = [os.path.join(lightcurve_dir, f) for f in os.listdir(lightcurve_dir) if f.endswith('.fits')]
    
    # Filter out already processed files
    processed_files = get_processed_files()
    new_files_to_process = [f for f in all_files if f not in processed_files]

    if not new_files_to_process:
        logger.info("No new lightcurves to process.")
        print("Pipeline run complete. No new files were found.")
        return

    logger.info(f"Found {len(new_files_to_process)} new lightcurves to analyze.")

    for file_path in new_files_to_process:
        logger.info(f"--- Processing file: {os.path.basename(file_path)} ---")
        
        # 1. Extract TIC ID from filename
        tic_id = get_tic_id_from_filename(os.path.basename(file_path))
        if not tic_id:
            logger.warning(f"Could not extract TIC ID from {file_path}. Skipping.")
            mark_file_as_processed(file_path) # Mark as processed to avoid retrying
            continue

        # 2. Run the main discovery pipeline as a subprocess
        try:
            # We call run_discovery_v2.py as a separate process to ensure
            # that each file is processed in a clean environment.
            command = [
                "python",
                "run_discovery_v2.py",
                "--tic_id", str(tic_id),
                "--mission", "Kepler"
            ]
            
            logger.info(f"Executing command: {' '.join(command)}")
            
            # Using subprocess.run to wait for completion
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True # This will raise an exception if the script fails
            )
            
            # Print the output from the subprocess
            print(result.stdout)
            if result.stderr:
                print("--- Subprocess Errors ---")
                print(result.stderr)
            
            logger.info(f"Successfully processed TIC {tic_id}.")
            
            # 3. Mark the file as processed only on success
            mark_file_as_processed(file_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to process {file_path} for TIC {tic_id}. Subprocess failed.")
            print(f"--- FAILED to process {file_path} ---")
            print(e.stdout)
            print(e.stderr)
            # We don't mark it as processed, so it will be retried on the next run.
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {file_path}: {e}", exc_info=True)
            
        # Optional: Add a delay between processing files
        time.sleep(2) 
        
    logger.info("--- Continuous Discovery Pipeline Scan Finished ---")

if __name__ == "__main__":
    main()
