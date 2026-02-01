
import os
import pandas as pd
from pipeline.result_analyzer import analyze_pipeline_results

# 1. Read the HTML report and extract the table
html_file = 'results/run_20251230-084051/pipeline_report_20251230-084051.html'
tables = pd.read_html(html_file)
# Assuming the "Detected Transits" table is the first one
df = tables[0]

# 2. Transform the DataFrame into the required list of dictionaries
results_list = []
for _, row in df.iterrows():
    results_list.append({
        'success': True,
        'transit_count': 1,
        'periodicity': row['Period (days)'],
        'planet_properties': {
            'radius_earth': row['Planet Radius (Earth)']
        }
    })

# 3. Create the output directory if it doesn't exist
output_dir = 'paper_figures/'
os.makedirs(output_dir, exist_ok=True)

# 4. Call the analyze_pipeline_results function
analyze_pipeline_results(results_list, output_dir=output_dir)

print(f"Distribution plots saved in {output_dir}")
