
import pandas as pd
import re

# Read the exoplanet_labels.csv file
df = pd.read_csv("/Users/emmanuel/proj/exo/data/metadata/exoplanet_labels.csv")

# Filter for TESS confirmed planets with 'Transit' discovery method
# Assuming TESS IDs are in 'hostname' column and start with 'TOI' or 'TIC'
tess_df = df[
    (df['hostname'].str.startswith('TOI', na=False) | df['hostname'].str.startswith('TIC', na=False)) & 
    (df['discoverymethod'] == 'Transit')
].copy()

# Extract TIC ID from hostname or pl_name
def extract_tic_id(row):
    if row['hostname'].startswith('TOI'):
        match = re.search(r'TOI-(\d+)', row['hostname'])
        if match:
            return int(match.group(1))
    elif row['hostname'].startswith('TIC'):
        match = re.search(r'TIC (\d+)', row['hostname'])
        if match:
            return int(match.group(1))
    return None

tess_df['tic_id'] = tess_df.apply(extract_tic_id, axis=1)

# Drop rows where tic_id could not be extracted
tess_df.dropna(subset=['tic_id'], inplace=True)

# Convert tic_id to integer
tess_df['tic_id'] = tess_df['tic_id'].astype(int)

# Save only the unique tic_ids to a new CSV file
tess_df[['tic_id']].drop_duplicates().to_csv("tess_confirmed_planets.csv", index=False)

print("Created tess_confirmed_planets.csv with TESS IDs.")
