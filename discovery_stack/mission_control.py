import logging
import os
import time
import shutil
import threading
from datetime import datetime, timedelta
from typing import List, Optional

try:
    from .registry import TargetRegistry
    from .sentinel import DataSentinel, TargetConfig
    from .ingest import IngestionEngine
    from .governance import DiscoveryGovernor as Governor
    from .wrappers import BayesianWrapper
    from .types import TransitHypothesis
    from .csv_logger import CSVLogger
    # Removed: from data.download_from_shell import parse_and_download
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure all discovery_stack components are in the correct path.")
    raise

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# --- Configuration Constants ---
DOWNLOAD_FOLDER = "data_files/incoming/"
REGISTRY_FILE = "processed_targets.json"
CSV_LOG_FILE = "discovery_log.csv"
MAX_DISK_USAGE_PCT = 85.0 # Re-added
START_DATE_DAYS_AGO = 365 # Re-added (1 year lookback)
POLL_INTERVAL_SECONDS = 3600  # 1 hour
MISSIONS_TO_POLL = ["TESS", "KEPLER", "K2"] # Re-added
# Removed: SHELL_SCRIPT_PATH = "data/Kepler.sh"

def run_daemon(stop_event: Optional[threading.Event] = None):
    """The main daemon loop for continuous monitoring, download, and processing."""
    logger.info("Starting Continuous Discovery Service...")
    
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    # --- Initialize Components ---
    registry = TargetRegistry(REGISTRY_FILE)
    # DataSentinel now initialized with download_dir and max_disk_usage_pct
    sentinel = DataSentinel(registry, DOWNLOAD_FOLDER, MAX_DISK_USAGE_PCT)
    ingestion_engine = IngestionEngine()
    bayesian_wrapper = BayesianWrapper()
    governor = Governor()
    csv_logger = CSVLogger(CSV_LOG_FILE)

    logger.info(f"Service initialized. Download folder: {DOWNLOAD_FOLDER}")
    logger.info(f"Disk usage limit: {MAX_DISK_USAGE_PCT}%")
    logger.info(f"Polling MAST for {MISSIONS_TO_POLL} data since {START_DATE_DAYS_AGO} days ago.")

    while not (stop_event and stop_event.is_set()):
        logger.info("--- Starting new discovery cycle ---")

        # 1. Download Phase: Poll MAST for new data from multiple missions
        try:
            start_date = (datetime.now() - timedelta(days=START_DATE_DAYS_AGO)).strftime("%Y-%m-%d")
            sentinel.poll_mast_and_download(missions=MISSIONS_TO_POLL, start_date=start_date)
        except Exception as e:
            logger.error(f"Critical error during MAST polling/download phase: {e}", exc_info=True)

        # 2. Processing Phase: Scan local folder for downloaded files
        targets_to_process: List[TargetConfig] = []
        try:
            targets_generator = sentinel.check_for_new_data(DOWNLOAD_FOLDER) # Pass watch_folder
            targets_to_process = list(targets_generator)
        except Exception as e:
            logger.error(f"Error checking for new data in download folder: {e}", exc_info=True)

        if not targets_to_process:
            logger.info(f"No new targets to process. Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
            if stop_event:
                stop_event.wait(POLL_INTERVAL_SECONDS)
            else:
                time.sleep(POLL_INTERVAL_SECONDS)
            continue

        logger.info(f"Found {len(targets_to_process)} new target(s) to process.")
        for target in targets_to_process:
            if stop_event and stop_event.is_set():
                break

            logger.info(f"Processing target: {target.mission} {target.target_id}, Sector {target.sector} from {target.source_file}")
            final_status = "Error"
            
            try:
                hypotheses: List[TransitHypothesis] = ingestion_engine.scan_target(target)

                if not hypotheses:
                    logger.info(f"No hypotheses generated for target {target.target_id}. Marking as processed.")
                    final_status = "Processed (No Signal)"
                else:
                    for hypothesis in hypotheses:
                        # Run through ML and Governance
                        hypothesis = bayesian_wrapper.predict(hypothesis)
                        hypothesis = governor.audit(hypothesis)
                        
                        # Log detailed metrics to CSV
                        csv_logger.log_hypothesis(hypothesis)
                        
                        # Update final status for the registry
                        if hypothesis.status.value == "Candidate":
                            final_status = "Candidate"
                
                # Mark as processed in JSON registry
                registry.mark_processed(
                    target.target_id, # Use target.target_id here
                    target.sector, 
                    final_status, 
                    datetime.utcnow().isoformat()
                )
                
                # Cleanup: Remove the processed file
                try:
                    os.remove(target.source_file)
                    logger.info(f"Cleaned up processed file: {target.source_file}")
                except OSError as e:
                    logger.error(f"Error removing file {target.source_file}: {e}")

            except Exception as e:
                logger.error(f"Error processing target {target.target_id}-{target.sector}: {e}", exc_info=True)
                registry.mark_processed(
                    target.target_id, # Use target.target_id here
                    target.sector,
                    "Error",
                    datetime.utcnow().isoformat()
                )

        logger.info(f"Discovery cycle finished. Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        if stop_event:
            stop_event.wait(POLL_INTERVAL_SECONDS)
        else:
            time.sleep(POLL_INTERVAL_SECONDS)

    logger.info("Continuous Discovery Service stopping.")

if __name__ == "__main__":
    try:
        run_daemon()
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")