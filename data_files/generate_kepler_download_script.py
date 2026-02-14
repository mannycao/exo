import pandas as pd
import os
import re

def generate_download_script(csv_path, output_script_path, output_dir):
    """
    Generates a shell script to download Kepler light curves based on target_ids
    in the provided CSV.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    kepler_ids = df['target_id'].unique()

    with open(output_script_path, 'w') as f:
        # Correct shebang line - \n is already a newline in Python
        f.write("#!/bin/bash\n") # Single newline here
        f.write("\n") # Additional newline for spacing
        
        f.write("# This script attempts to download Kepler light curves based on target_ids from the CSV.\n")
        f.write("# WARNING: The exact URL paths for Kepler light curves are complex and often derived\n")
        f.write("# from an internal directory structure (e.g., /Kepler/00X/YYY/ZZZ/) that cannot be\n")
        f.write("# programmatically determined from the target_id alone with the available information.\n")
        f.write("# The URLs generated below are generic placeholders and may NOT work directly.\n")
        f.write("# You may need to manually adjust the URL structure based on the actual archive for each object.\n")
        f.write("# Ensure 'wget' is installed on your system.\n\n")

        f.write(f"mkdir -p {output_dir}\n\n")
        f.write(f"cd {output_dir}\n\n")

        for kplr_id in kepler_ids:
            base_url = "http://exoplanetarchive.ipac.caltech.edu:80/data/ETSS/Kepler"
            
            # Reverting to the most explicit placeholder for the directory structure
            generic_path_segment = f"UNKNOWN_PATH_SEGMENT_1/UNKNOWN_PATH_SEGMENT_2/UNKNOWN_PATH_SEGMENT_3"
            
            fits_filename = f"kplr{kplr_id}_llc.fits"
            tbl_filename = f"kplr{kplr_id}_llc_lc.tbl"

            f.write(f"# Placeholder URLs for Kepler ID {kplr_id}\n")
            f.write(f"# Please replace UNKNOWN_PATH_SEGMENT_1/UNKNOWN_PATH_SEGMENT_2/UNKNOWN_PATH_SEGMENT_3\n")
            f.write(f"# with the correct directory segments from the archive (e.g., 005/159/31 for kplr010000941)\n")
            f.write(f"# Example from original script: {base_url}/005/159/31/kplr010000941-YYYYDDDHHMMSS_llc.fits\n")
            f.write(f"wget -O '{fits_filename}' '{base_url}/{generic_path_segment}/kplr{kplr_id}_llc.fits' -a {fits_filename}.log\n")
            f.write(f"wget -O '{tbl_filename}' '{base_url}/{generic_path_segment}/kplr{kplr_id}_llc_lc.tbl' -a {tbl_filename}.log\n")
            f.write("\n")

    print(f"Download script generated at: {output_script_path}")
    print("WARNING: Review the generated script. The URLs are placeholders and likely need manual correction.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    csv_file = os.path.join(current_dir, 'data_files', 'full_metadata_v2.csv')
    output_script_file = os.path.join(current_dir, 'data_files', 'download_kepler_lcs.sh')
    light_curves_dir = os.path.join(current_dir, 'data_files', 'light_curves')

    generate_download_script(csv_file, output_script_file, light_curves_dir)