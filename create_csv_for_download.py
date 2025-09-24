import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as ExoplanetArchive
import os

def create_csvs(output_dir="."):
    # Fetch confirmed planets
    print("Fetching confirmed planets from NASA Exoplanet Archive...")
    confirmed_planets_table = ExoplanetArchive.query_criteria(
        table="cumulative",
        select="kepid, kepoi_name, koi_disposition",
        where="koi_disposition='CONFIRMED'"
    )
    confirmed_df = confirmed_planets_table.to_pandas()
    confirmed_df = confirmed_df.rename(columns={'kepid': 'kepid'})
    confirmed_csv_path = os.path.join(output_dir, "confirmed_planets.csv")
    confirmed_df.to_csv(confirmed_csv_path, index=False)
    print(f"Created {confirmed_csv_path} with {len(confirmed_df)} confirmed planets.")

    # Fetch false positives
    print("Fetching false positives from NASA Exoplanet Archive...")
    false_positives_table = ExoplanetArchive.query_criteria(
        table="cumulative",
        select="kepid, kepoi_name, koi_disposition",
        where="koi_disposition='FALSE POSITIVE'"
    )
    false_positives_df = false_positives_table.to_pandas()
    false_positives_df = false_positives_df.rename(columns={'kepid': 'kepid'})
    false_positives_csv_path = os.path.join(output_dir, "false_positives.csv")
    false_positives_df.to_csv(false_positives_csv_path, index=False)
    print(f"Created {false_positives_csv_path} with {len(false_positives_df)} false positives.")

if __name__ == "__main__":
    create_csvs()
