import subprocess
import os
import sys
import logging
from pathlib import Path
import shutil
import time

# Setup basic logging for the system test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "system_test_output"

# Import create_mock_fits_file
from data.data_fetcher import create_mock_fits_file

def run_command(command, description):
    logger.info(f"Running: {description} (Command: {' '.join(command)})")
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        logger.error(f"--- {description} FAILED ---")
        logger.error(f"STDOUT:\n{process.stdout}")
        logger.error(f"STDERR:\n{process.stderr}")
        return False
    logger.info(f"--- {description} SUCCEEDED ---\n") # Added newline for better readability
    logger.debug(f"STDOUT:\n{process.stdout}")
    return True

def setup_dummy_data():
    logger.info("Setting up dummy data for system test...")
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create dummy CSVs for download_script.py
    confirmed_csv = SAMPLE_DATA_DIR / "confirmed_planets_dummy.csv"
    fp_csv = SAMPLE_DATA_DIR / "false_positives_dummy.csv"

    with open(confirmed_csv, "w") as f:
        f.write("kepid\n100000000\n100000001\n") # Dummy KIC IDs
    with open(fp_csv, "w") as f:
        f.write("kepid\n200000000\n200000001\n") # Dummy KIC IDs
    
    logger.info("Dummy CSVs created.")

    # Create dummy .fits files for main.py, run_hp_tuning.py, run_bayesian_inference.py
    planets_dir = SAMPLE_DATA_DIR / "confirmed_planets"
    false_positives_dir = SAMPLE_DATA_DIR / "false_positives"
    planets_dir.mkdir(exist_ok=True)
    false_positives_dir.mkdir(exist_ok=True)

    # Generate a few dummy FITS files
    for i in range(50):
        create_mock_fits_file(planets_dir / f"kplr00000000{i}-2009131105131_llc.fits")
        create_mock_fits_file(false_positives_dir / f"kplr00000010{i}-2009131105131_llc.fits")
    logger.info("Dummy .fits files created.")


def cleanup():
    logger.info("Cleaning up system test output and dummy data...")
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    if SAMPLE_DATA_DIR.exists():
        # Only remove if it was created by this script (check for dummy files)
        if (SAMPLE_DATA_DIR / "confirmed_planets_dummy.csv").exists():
            shutil.rmtree(SAMPLE_DATA_DIR)
    logger.info("Cleanup complete.")

def main():
    cleanup() # Ensure clean slate
    setup_dummy_data()
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_success = True

    # Test 1: download_script.py
    download_output_dir = TEST_OUTPUT_DIR / "download_output"
    download_output_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "download_script.py"),
        "--confirmed_csv", str(SAMPLE_DATA_DIR / "confirmed_planets_dummy.csv"),
        "--fp_csv", str(SAMPLE_DATA_DIR / "false_positives_dummy.csv"),
        "--output_base", str(download_output_dir),
        "--max_workers", "2" # Use a small number of workers for test
    ]
    if not run_command(cmd, "Download Script"):
        overall_success = False

    # Test 2: main.py (full dataset)
    main_output_dir_all = TEST_OUTPUT_DIR / "main_output_all"
    main_output_dir_all.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--planets_dir", str(SAMPLE_DATA_DIR / "confirmed_planets"),
        "--false_positives_dir", str(SAMPLE_DATA_DIR / "false_positives"),
        "--split_type", "all"
    ]
    # main.py creates a timestamped directory, so we need to find it
    initial_results_count = len(list(PROJECT_ROOT.glob("results/run_*")))
    if not run_command(cmd, "Main Pipeline (All Data)"):
        overall_success = False
    else:
        # Verify main.py created an output directory
        new_results_dirs = list(PROJECT_ROOT.glob("results/run_*"))
        if len(new_results_dirs) > initial_results_count:
            latest_run_dir = sorted(new_results_dirs, key=os.path.getmtime, reverse=True)[0]
            logger.info(f"Main pipeline output in: {latest_run_dir}")
            # Check for expected summary file
            if not list(latest_run_dir.glob("pipeline_summary_*.json")):
                logger.error(f"Main pipeline: pipeline_summary_*.json not found in {latest_run_dir}")
                overall_success = False
        else:
            logger.error("Main pipeline: No new run directory created in results.")
            overall_success = False

    # Test 3: main.py (50/50 split)
    main_output_dir_50_50 = TEST_OUTPUT_DIR / "main_output_50_50"
    main_output_dir_50_50.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--planets_dir", str(SAMPLE_DATA_DIR / "confirmed_planets"),
        "--false_positives_dir", str(SAMPLE_DATA_DIR / "false_positives"),
        "--split_type", "50_50"
    ]
    initial_results_count = len(list(PROJECT_ROOT.glob("results/run_*")))
    if not run_command(cmd, "Main Pipeline (50/50 Split)"):
        overall_success = False
    else:
        new_results_dirs = list(PROJECT_ROOT.glob("results/run_*"))
        if len(new_results_dirs) > initial_results_count:
            latest_run_dir = sorted(new_results_dirs, key=os.path.getmtime, reverse=True)[0]
            logger.info(f"Main pipeline (50/50) output in: {latest_run_dir}")
            if not list(latest_run_dir.glob("pipeline_summary_*.json")):
                logger.error(f"Main pipeline (50/50): pipeline_summary_*.json not found in {latest_run_dir}")
                overall_success = False
        else:
            logger.error("Main pipeline (50/50): No new run directory created in results.")
            overall_success = False

    # Test 4: run_hp_tuning.py
    hp_output_dir = TEST_OUTPUT_DIR / "hp_tuning_output"
    hp_output_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_hp_tuning.py"),
        "--planets_dir", str(SAMPLE_DATA_DIR / "confirmed_planets"),
        "--false_positives_dir", str(SAMPLE_DATA_DIR / "false_positives"),
        "--n_iter", "1", # Small number of iterations for test
        "--cv", "2",     # Small number of CV folds for test
        "--max_workers", "1" # Limit workers for test stability
    ]
    if not run_command(cmd, "Hyperparameter Tuning Script"):
        overall_success = False
    # Note: run_hp_tuning.py doesn't create a specific output directory for results,
    # it prints to console. We rely on the script's exit code for success.

    # Test 5: run_bayesian_inference.py
    # This requires a trained model. We'll use a dummy path and rely on the script's error handling
    # or create a very simple dummy model if needed. For now, let's assume a dummy model path.
    # A more robust test would involve training a tiny model first.
    dummy_model_path = TEST_OUTPUT_DIR / "dummy_model.h5"
    # Create a dummy empty file to simulate a model
    open(dummy_model_path, 'a').close() 

    bayesian_output_dir = TEST_OUTPUT_DIR / "bayesian_output"
    bayesian_output_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_bayesian_inference.py"),
        "--model_path", str(dummy_model_path),
        "--planets_dir", str(SAMPLE_DATA_DIR / "confirmed_planets"),
        "--false_positives_dir", str(SAMPLE_DATA_DIR / "false_positives"),
        "--output_dir", str(bayesian_output_dir),
        "--n_samples", "5" # Small number of samples for test
    ]
    if not run_command(cmd, "Bayesian Inference Script"):
        overall_success = False
    else:
        # Verify bayesian_inference.py created an output file
        if not list(bayesian_output_dir.glob("inference_results_optimized.csv")):
            logger.error(f"Bayesian inference: inference_results_optimized.csv not found in {bayesian_output_dir}")
            overall_success = False

    if overall_success:
        logger.info("All system tests PASSED!")
    else:
        logger.error("One or more system tests FAILED!")
        sys.exit(1)

    cleanup()

if __name__ == "__main__":
    main()
