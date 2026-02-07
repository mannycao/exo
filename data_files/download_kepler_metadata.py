import pandas as pd
from pathlib import Path

# Define the input path for the local kep.txt file
KEPLER_LOCAL_FILE = Path(__file__).resolve().parents[1] / "kep.txt"

# Define the output path for the cleaned CSV
output_dir = Path(__file__).resolve().parent
OUTPUT_CSV_PATH = output_dir / "kepler_metadata.csv"

def download_and_clean_kepler_metadata():
    """
    Reads Kepler metadata from a local kep.txt file, cleans the data,
    and saves it to a CSV file.
    """
    print(f"Reading Kepler metadata from local file: {KEPLER_LOCAL_FILE}")
    try:
        # 1. Use pandas to read the local kep.txt file
        df = pd.read_csv(KEPLER_LOCAL_FILE)
        print("Data loaded. Cleaning data...")

        # 2. Clean the Data
        # Rename column kepid to target_id
        if 'kepid' in df.columns:
            df.rename(columns={'kepid': 'target_id'}, inplace=True)
        # Rename column koi_period to period
        if 'koi_period' in df.columns:
            df.rename(columns={'koi_period': 'period'}, inplace=True)
        
        # Select relevant columns. true_label will be added during merge_metadata.py
        final_columns = ['target_id', 'period']
        # Filter out columns that might not exist in case of input file variations
        df = df[[col for col in final_columns if col in df.columns]]

        # 3. Save the cleaned dataframe
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        # 4. Print confirmation
        print(f"Processed metadata for {len(df)} KOIs and saved to {OUTPUT_CSV_PATH}")

    except FileNotFoundError:
        print(f"Error: {KEPLER_LOCAL_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_clean_kepler_metadata()
