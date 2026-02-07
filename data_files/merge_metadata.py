import pandas as pd
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Define input and output paths
output_dir = Path(__file__).resolve().parent
project_root = output_dir.parents[0] # Assuming project root is one level up from data_files

TESS_METADATA_PATH = output_dir / "tess_metadata.csv"
KEPLER_METADATA_PATH = output_dir / "kepler_metadata.csv"
CONFIRMED_PLANETS_PATH = project_root / "confirmed_planets.csv"
FALSE_POSITIVES_PATH = project_root / "false_positives.csv"
FULL_METADATA_PATH = output_dir / "full_metadata.csv"


def merge_metadata_files():
    """
    Loads TESS and Kepler period data, and confirmed/false positive labels,
    merges them into a unified dataframe, and saves it to a CSV file.
    The final dataframe will contain 'target_id', 'period', and 'true_label'.
    """
    print(f"Starting metadata merge process...")
    
    try:
        # Load period data from Kepler and TESS
        df_tess_periods = pd.read_csv(TESS_METADATA_PATH)
        df_kepler_periods = pd.read_csv(KEPLER_METADATA_PATH)

        # Ensure consistent column types for merging, especially target_id
        df_tess_periods['target_id'] = df_tess_periods['target_id'].astype(str)
        df_kepler_periods['target_id'] = df_kepler_periods['target_id'].astype(str)

        # Concatenate period data
        periods_df = pd.concat([df_tess_periods, df_kepler_periods], ignore_index=True)
        # Drop duplicates based on target_id, keeping the first if any
        periods_df.drop_duplicates(subset=['target_id'], keep='first', inplace=True)
        print(f"Loaded and consolidated {len(periods_df)} unique targets with period data.")

        # Load true labels from confirmed_planets.csv and false_positives.csv
        df_confirmed = pd.read_csv(CONFIRMED_PLANETS_PATH)
        df_false_positives = pd.read_csv(FALSE_POSITIVES_PATH)

        # Prepare labels dataframe
        labels_confirmed = df_confirmed[['kepid', 'koi_disposition']].copy()
        labels_confirmed.rename(columns={'kepid': 'target_id', 'koi_disposition': 'true_label'}, inplace=True)
        labels_confirmed['target_id'] = labels_confirmed['target_id'].astype(str)

        labels_false_positives = df_false_positives[['kepid', 'koi_disposition']].copy()
        labels_false_positives.rename(columns={'kepid': 'target_id', 'koi_disposition': 'true_label'}, inplace=True)
        labels_false_positives['target_id'] = labels_false_positives['target_id'].astype(str)
        
        # Combine all labels
        labels_df = pd.concat([labels_confirmed, labels_false_positives], ignore_index=True)
        labels_df.drop_duplicates(subset=['target_id'], keep='first', inplace=True)

        # Standardize true_label values
        labels_df['true_label'] = labels_df['true_label'].replace({'CANDIDATE': 'CANDIDATE', 'CONFIRMED': 'CONFIRMED', 'FALSE POSITIVE': 'FALSE_POSITIVE'})
        print(f"Loaded and consolidated {len(labels_df)} unique targets with true labels.")

        # Merge periods_df with labels_df
        # Use a left merge to keep all entries from periods_df and add labels
        final_metadata_df = pd.merge(periods_df, labels_df, on='target_id', how='left')

        # Fill any missing true_label values with 'UNKNOWN'
        final_metadata_df['true_label'].fillna('UNKNOWN', inplace=True)
        
        # Sort by target_id for consistency
        final_metadata_df.sort_values(by='target_id', inplace=True)
        
        # Save the unified and cleaned dataframe
        final_metadata_df.to_csv(FULL_METADATA_PATH, index=False)
        
        print(f"Merged metadata saved to {FULL_METADATA_PATH} with {len(final_metadata_df)} entries.")
        print(f"Breakdown of labels in {FULL_METADATA_PATH}:\n{final_metadata_df['true_label'].value_counts()}")

    except FileNotFoundError as e:
        logger.error(f"Error: Required metadata file not found: {e}. Please ensure all input CSVs exist.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during metadata merging: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    merge_metadata_files()
