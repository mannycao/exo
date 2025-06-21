# test_imports.py
import sys
from pprint import pprint

print("--- Starting Minimal Import Test ---")
print(f"Python Executable: {sys.executable}")
print("-" * 40)

try:
    print("Step 1: Importing 'pipeline.pipeline_runner'...")
    # This is the first major import in your main.py
    from pipeline.pipeline_runner import setup_logging
    print("   ✅ SUCCESS: Imported 'pipeline.pipeline_runner'.")
    print("-" * 40)

    # Note: pipeline_runner.py automatically imports data_fetcher.py,
    # so the error might have already happened. If not, this next step
    # will re-check it explicitly.

    print("Step 2: Directly importing 'data.data_fetcher'...")
    from data.data_fetcher import fetch_exoplanet_labels
    print("   ✅ SUCCESS: Directly imported 'data.data_fetcher'.")
    print("-" * 40)

    print("Step 3: Directly importing the failing submodule...")
    # This is the line that fails inside data_fetcher.py
    from astroquery.ipac.nexsci.tap import TapPlus
    print("   ✅ SUCCESS: Directly imported 'astroquery.ipac.nexsci.tap'.")
    print("-" * 40)

    print("✅✅✅ All imports in the chain were successful.")

except Exception as e:
    print(f"\n❌ FAILED during the import chain test.")
    print(f"    Error Type: {type(e).__name__}")
    print(f"    Error Message: {e}")
    print("\n--- Final sys.path at time of failure ---")
    pprint(sys.path)

print("\n--- Test Complete ---")