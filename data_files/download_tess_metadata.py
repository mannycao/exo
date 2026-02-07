import pandas as pd
from pathlib import Path

# Define the input path for the local tess.txt file
TESS_LOCAL_FILE = Path(__file__).resolve().parents[1] / "tess.txt"

# Define the output path for the cleaned CSV
output_dir = Path(__file__).resolve().parent
OUTPUT_CSV_PATH = output_dir / "tess_metadata.csv"

def download_and_clean_tess_metadata():
    """
    Reads TESS metadata from a local tess.txt file, cleans the data,
    and saves it to a CSV file.
    """
    print(f"Reading TESS TOI Catalog from local file: {TESS_LOCAL_FILE}")
    try:
        # 1. Use pandas to read the local tess.txt file
        df = pd.read_csv(TESS_LOCAL_FILE)
        print("Data loaded. Cleaning data...")

        # 2. Clean the Data
        # Rename column tid to target_id
        if 'tid' in df.columns:
            df.rename(columns={'tid': 'target_id'}, inplace=True)
        # Rename column pl_orbper to period
        if 'pl_orbper' in df.columns:
            df.rename(columns={'pl_orbper': 'period'}, inplace=True)
        
        # Select relevant columns. true_label will be added during merge_metadata.py
        final_columns = ['target_id', 'period']
        # Filter out columns that might not exist in case of input file variations
        df = df[[col for col in final_columns if col in df.columns]]

        # 3. Save the cleaned dataframe
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        # 4. Print confirmation
        print(f"Processed metadata for {len(df)} TOIs and saved to {OUTPUT_CSV_PATH}")

    except FileNotFoundError:
        print(f"Error: {TESS_LOCAL_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_clean_tess_metadata()
