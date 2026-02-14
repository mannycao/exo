import pandas as pd
from pathlib import Path
import os

def generate_tess_tce_catalog():
    project_root = Path(__file__).resolve().parent
    tess_confirmed_planets_path = project_root / "tess_confirmed_planets.csv"
    tess_metadata_path = project_root / "data_files" / "tess_metadata.csv"
    output_tce_catalog_path = project_root / "data_files" / "tess_tce_catalog.csv"

    # 1. Read tess_confirmed_planets.csv
    try:
        df_confirmed_tess = pd.read_csv(tess_confirmed_planets_path)
        df_confirmed_tess.rename(columns={'tic_id': 'target_id'}, inplace=True)
        # Ensure target_id is string for merging
        df_confirmed_tess['target_id'] = df_confirmed_tess['target_id'].astype(str)
        print(f"Loaded {len(df_confirmed_tess)} TIC IDs from {tess_confirmed_planets_path}")
    except FileNotFoundError:
        print(f"Error: {tess_confirmed_planets_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {tess_confirmed_planets_path}: {e}")
        return

    # 2. Read data_files/tess_metadata.csv
    try:
        df_tess_metadata = pd.read_csv(tess_metadata_path)
        # Ensure target_id is string for merging
        df_tess_metadata['target_id'] = df_tess_metadata['target_id'].astype(str)
        print(f"Loaded {len(df_tess_metadata)} entries from {tess_metadata_path}")
    except FileNotFoundError:
        print(f"Error: {tess_metadata_path} not found.")
        return
    except Exception as e:
        print(f"Error reading {tess_metadata_path}: {e}")
        return

    # 3. Merge them, prioritizing tess_metadata.csv for period and true_label
    # Use a left merge to keep all TIC IDs from df_confirmed_tess
    # and add period and true_label from df_tess_metadata
    merged_df = pd.merge(
        df_confirmed_tess,
        df_tess_metadata[['target_id', 'period', 'true_label']],
        on='target_id',
        how='left'
    )
    print(f"Merged data. Initial size: {len(merged_df)}")

    # 4. Fill missing values for period with 0 and true_label with 'CANDIDATE'
    merged_df['period'].fillna(0, inplace=True)
    merged_df['true_label'].fillna('CANDIDATE', inplace=True)
    print("Filled missing period with 0 and true_label with 'CANDIDATE'.")

    # Ensure column order and rename for tess_tce_catalog.csv expectations in download_tess_tce.py
    # The download_tess_tce.py script expects 'tic_id', 'period', and 'tce_disposition'/'tce_tags'
    # We will map 'target_id' to 'tic_id' and 'true_label' to 'tce_disposition'
    final_tce_catalog_df = merged_df[['target_id', 'period', 'true_label']].copy()
    final_tce_catalog_df.rename(columns={'target_id': 'tic_id', 'true_label': 'tce_disposition'}, inplace=True)

    # 5. Save the resulting dataframe as data_files/tess_tce_catalog.csv
    output_tce_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    final_tce_catalog_df.to_csv(output_tce_catalog_path, index=False)
    print(f"Generated {output_tce_catalog_path} with {len(final_tce_catalog_df)} entries.")
    print(f"Breakdown of labels in {output_tce_catalog_path}:\n{final_tce_catalog_df['tce_disposition'].value_counts()}")

if __name__ == "__main__":
    generate_tess_tce_catalog()
