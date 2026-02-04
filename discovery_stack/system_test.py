import unittest
import os
import shutil
import threading
import time
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

# Add project root to path to allow imports from other directories
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from discovery_stack.mission_control import run_daemon, WATCH_FOLDER, PROCESSED_FOLDER, REGISTRY_FILE
import config as proj_config

# --- Test Configuration ---
TEST_WATCH_FOLDER = "test_data/incoming"
TEST_PROCESSED_FOLDER = "test_data/processed"
TEST_REGISTRY_FILE = "test_data/processed_targets.json"
TEST_LIGHT_CURVE_DIR = "test_data/light_curves"
TEST_POLL_INTERVAL = 1  # seconds

# Use a known KIC ID from the data fetcher's known list
TEST_KIC_ID = "11904151" 

def create_dummy_fits_file(path: str, time_data, flux_data):
    """Creates a minimal FITS file with a light curve."""
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = 'KEPLER'
    primary_hdu.header['OBJECT'] = f'KIC {TEST_KIC_ID}'
    primary_hdu.header['KEPLERID'] = TEST_KIC_ID
    
    col1 = fits.Column(name='TIME', format='D', array=time_data)
    col2 = fits.Column(name='PDCSAP_FLUX', format='E', array=flux_data)
    
    cols = fits.ColDefs([col1, col2])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header['EXTNAME'] = 'LIGHTCURVE'
    
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(path, overwrite=True)

class TestContinuousMonitoringService(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.test_dirs = [TEST_WATCH_FOLDER, TEST_PROCESSED_FOLDER, TEST_LIGHT_CURVE_DIR]
        for d in self.test_dirs:
            os.makedirs(d, exist_ok=True)

        # 1. Create a dummy light curve file in the *real* light curve cache
        # This simulates the file having been downloaded previously.
        time_data = np.arange(0, 10, 0.1)
        flux_data = np.random.randn(len(time_data))
        # Add a fake transit to ensure the ingestion engine finds something
        flux_data[20:30] -= 0.1
        
        self.cached_lc_path = os.path.join(proj_config.LIGHT_CURVE_DIR, f'kplr{str(TEST_KIC_ID).zfill(9)}_llc.fits')
        os.makedirs(proj_config.LIGHT_CURVE_DIR, exist_ok=True)
        create_dummy_fits_file(self.cached_lc_path, time_data, flux_data)

        # 2. Create the trigger file in the watched folder
        self.trigger_filename = f"tess2024-s0015-{str(TEST_KIC_ID).zfill(12)}-0120-s_lc.fits"
        self.trigger_filepath = os.path.join(TEST_WATCH_FOLDER, self.trigger_filename)
        with open(self.trigger_filepath, 'w') as f:
            f.write("dummy content")

        # 3. Clean up registry from previous runs
        if os.path.exists(TEST_REGISTRY_FILE):
            os.remove(TEST_REGISTRY_FILE)

    def tearDown(self):
        """Clean up the test environment."""
        for d in self.test_dirs:
            shutil.rmtree(d, ignore_errors=True)
        
        if os.path.exists(self.cached_lc_path):
            os.remove(self.cached_lc_path)
        
        if os.path.exists(TEST_REGISTRY_FILE):
            os.remove(TEST_REGISTRY_FILE)

    def test_service_end_to_end(self):
        """
        Tests the full pipeline from file detection to processing.
        """
        stop_event = threading.Event()
        daemon_thread = threading.Thread(
            target=run_daemon,
            args=(TEST_WATCH_FOLDER, TEST_POLL_INTERVAL, stop_event),
            kwargs={'registry_file': TEST_REGISTRY_FILE, 'processed_folder': TEST_PROCESSED_FOLDER}
        )

        try:
            daemon_thread.start()

            # Wait for the file to be processed
            # Max wait time: 10 seconds
            max_wait = 10
            wait_time = 0
            file_moved = False
            while wait_time < max_wait:
                if not os.path.exists(self.trigger_filepath):
                    file_moved = True
                    break
                time.sleep(0.5)
                wait_time += 0.5
            
            self.assertTrue(file_moved, "The trigger file was not moved to the processed folder.")

            # Check if the file exists in the processed folder
            processed_filepath = os.path.join(TEST_PROCESSED_FOLDER, self.trigger_filename)
            self.assertTrue(os.path.exists(processed_filepath), "File does not exist in processed folder.")

            # Check the registry
            self.assertTrue(os.path.exists(TEST_REGISTRY_FILE), "Registry file was not created.")
            with open(TEST_REGISTRY_FILE, 'r') as f:
                registry_data = json.load(f)
            
            target_key = f"{TEST_KIC_ID}-15" # Sector 15 is parsed from the filename
            self.assertIn(target_key, registry_data, "Target was not found in the registry.")
            
            # Check the status. It can be Candidate, Rejected, etc., but should not be Error.
            self.assertNotEqual(registry_data[target_key]['status'], 'Error', "Processing resulted in an Error status.")
            self.assertIn(registry_data[target_key]['status'], ["ML_CANDIDATE", "PHYSICS_CLEARED", "REJECTED"], "Status is not one of the expected values.")

        finally:
            # Stop the daemon
            stop_event.set()
            daemon_thread.join(timeout=5)
            self.assertFalse(daemon_thread.is_alive(), "Daemon thread did not stop.")

if __name__ == "__main__":
    unittest.main()
