import pandas as pd
from pathlib import Path
import sys
import os

# Assume config.py is in the project root
# To import config, we need to add the project root to sys.path
# Let's find the project root from the current working directory, which is reliable
# when the agent is invoked from the project root.
# If not, the user context tells us the project root is /Users/emmanuel/proj/phd
project_root = Path('/Users/emmanuel/proj/phd')

# Add project root to sys.path to import config
sys.path.insert(0, str(project_root))

try:
    import config
except ImportError:
    print(f"Error: Could not import config.py. Ensure project root ({project_root}) is correct and config.py exists.")
    sys.exit(1)


# Construct the path to full_metadata.csv relative to the project root
full_metadata_path = project_root / 'data_files' / 'full_metadata.csv'

# Ensure the path exists
if not full_metadata_path.exists():
    print(f"Error: Metadata file not found at {full_metadata_path}")
    sys.exit(1)

try:
    df = pd.read_csv(full_metadata_path, dtype={'target_id': str})
    # Filter for the specific target_id
    result = df[df['target_id'] == '48503881']

    if not result.empty:
        print("Target ID '48503881' found in full_metadata.csv:")
        print(result)
    else:
        print("Target ID '48503881' NOT found in full_metadata.csv.")
except Exception as e:
    print(f"An error occurred: {e}")