import unittest
import subprocess
import os
import json
import tempfile
import numpy as np
# from astropy.io import fits # No longer needed

# Import functions from our codebase to help with the test
# from data.data_fetcher import create_mock_fits_file # No longer needed
import config as lite_config

class TestDiscoveryPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for the test."""
        self.test_dir = tempfile.TemporaryDirectory()
        # Ensure RESULTS_DIR exists for the decision log
        os.makedirs(lite_config.RESULTS_DIR, exist_ok=True)
        
        # Reset the decision log for a clean run
        self.decision_log_path = os.path.join(lite_config.RESULTS_DIR, "discovery_decision_log.json")
        if os.path.exists(self.decision_log_path):
            os.remove(self.decision_log_path)

    def tearDown(self):
        """Clean up the temporary environment."""
        self.test_dir.cleanup()
        # Optionally clean up the decision log if it was created during the test
        if os.path.exists(self.decision_log_path):
            os.remove(self.decision_log_path)

    def test_pipeline_structural_integrity(self):
        """
        Tests that the pipeline runs end-to-end without crashing.
        This is a 'structural integrity' test, not a test of scientific correctness,
        due to the placeholder components in the pipeline.
        """
        # 1. Setup: Generate synthetic light curve data directly
        time = np.linspace(0, 100, 1000) # 100 days, 1000 points
        flux = np.ones_like(time) + np.random.normal(0, 0.001, time.shape) # Base flux + noise

        # Inject a transit
        period = 10.0
        t0 = 5.0
        duration = 0.1
        depth = 0.05 # Making it a significant depth for detection

        for transit_center in np.arange(t0, time.max(), period):
            transit_mask = (time > (transit_center - duration/2)) & \
                           (time < (transit_center + duration/2))
            flux[transit_mask] -= depth
        
        # We need a TIC ID, but since we're not using a FITS file, we can just pick one.
        # This TIC ID won't be used for fetching, but for identification within the pipeline.
        test_tic_id = 987654321 

        # 2. Execution: Run the main pipeline script
        # We're calling `run_pipeline` directly now for better control and
        # to pass the generated time/flux data.
        from run_discovery_v2 import run_pipeline

        # Temporarily redirect stdout to capture print statements from run_pipeline
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            run_pipeline(tic_id=test_tic_id, mission="Synthetic", sector="1")
        finally:
            sys.stdout = old_stdout # Restore stdout
            pipeline_output = mystdout.getvalue()
        
        # Print pipeline output for debugging
        print("--- Pipeline STDOUT ---")
        print(pipeline_output)
        print("--- End Pipeline Output ---")

        # 3. Assertions
        # The pipeline should run without raising unhandled exceptions.
        # Check that the decision log was created
        self.assertTrue(os.path.exists(self.decision_log_path), "The decision log file should be created.")
        
        # Check that the log is valid JSON and not empty
        with open(self.decision_log_path, 'r') as f:
            try:
                log_data = json.load(f)
                self.assertIsInstance(log_data, list, "Log file should contain a JSON list.")
                # We expect at least one hypothesis now due to injected transit
                self.assertGreater(len(log_data), 0, "The decision log should not be empty.")
                
                # Check the first entry
                first_entry = log_data[0]
                self.assertIn('hypothesis_id', first_entry)
                self.assertIn('new_status', first_entry)
                self.assertIn('reason', first_entry)

            except (json.JSONDecodeError, IndexError) as e:
                self.fail(f"Decision log is not valid or is empty: {e}")

    # Helper to generate the time and flux for the test
    def _generate_synthetic_lightcurve(self, period=10.0, t0=5.0, duration=0.1, depth=0.05):
        time = np.linspace(0, 100, 1000) # 100 days, 1000 points
        flux = np.ones_like(time) + np.random.normal(0, 0.001, time.shape) # Base flux + noise

        for transit_center in np.arange(t0, time.max(), period):
            transit_mask = (time > (transit_center - duration/2)) & \
                           (time < (transit_center + duration/2))
            flux[transit_mask] -= depth
        return time, flux

if __name__ == '__main__':
    unittest.main()