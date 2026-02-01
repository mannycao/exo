import pandas as pd
import json

def analyze_disagreement_snr():
    # Load the disagreement data
    disagreement_df = pd.read_csv('representation_disagreement.csv')

    # Filter for high disagreement
    high_disagreement = disagreement_df[disagreement_df['Delta'] > 0.5]

    # Load the original pipeline results to get file paths
    with open('results/run_20251230-084051/processed_data/all_pipeline_results.json', 'r') as f:
        original_pipeline_results = json.load(f)

    # Load the new pipeline results
    with open('results/run_20260125-143515/processed_data/all_pipeline_results.json', 'r') as f:
        new_pipeline_results = json.load(f)

    # Create a set of file_paths from the new run for quick lookup
    new_results_filepaths = {result['file_path'] for result in new_pipeline_results}
    
    # Create a dictionary for quick lookup of new pipeline results by file_path
    new_results_dict = {result['file_path']: result for result in new_pipeline_results}

    print("BLS Max Power for High-Disagreement Cases (found in the new run):")
    print("=" * 70)
    
    target_ids = high_disagreement['Target_ID'].astype(int).tolist()

    found_cases = 0
    for target_id in target_ids:
        if target_id < len(original_pipeline_results):
            original_result = original_pipeline_results[target_id]
            file_path = original_result.get('file_path')

            if file_path in new_results_filepaths:
                found_cases += 1
                bls_power = new_results_dict[file_path].get('bls_max_power', 'N/A')
                ground_truth = high_disagreement[high_disagreement['Target_ID'] == target_id]['Ground_Truth_Label'].iloc[0]
                print(f"File: {file_path}, BLS Max Power: {bls_power}, Ground Truth: {ground_truth}")
        else:
            # This case should ideally not happen if disagreement file is aligned with original results
            print(f"Warning: Target_ID {target_id} out of bounds for original results.")

    if found_cases == 0:
        print("\nNo high-disagreement cases from the original run were found in the new limited run.")
        print("This is expected if the limited run did not include those specific files.")


if __name__ == '__main__':
    analyze_disagreement_snr()
