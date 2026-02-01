
import pandas as pd
import numpy as np
import os

def audit_table_numbers():
    # --- Setup ---
    disagreement_csv_path = 'representation_disagreement.csv'

    if not os.path.exists(disagreement_csv_path):
        print(f"Error: {disagreement_csv_path} not found. Please ensure Experiment 1 has been run to generate it.")
        return

    df = pd.read_csv(disagreement_csv_path)

    # Calculate Delta if it doesn't exist (it should from previous steps)
    if 'Delta' not in df.columns:
        df['Delta'] = np.abs(df['Prob_1D'] - df['Prob_2D'])

    # Filter for "High Disagreement"
    high_disagreement_df = df[df['Delta'] > 0.5]

    # Count Latent_Faults (High Disagreement + Ground_Truth==0)
    latent_faults_count = len(high_disagreement_df[high_disagreement_df['Ground_Truth_Label'] == 0])

    # Count False_Alerts (High Disagreement + Ground_Truth==1)
    false_alerts_count = len(high_disagreement_df[high_disagreement_df['Ground_Truth_Label'] == 1])

    # --- Output ---
    print("\n--- LaTeX Table Rows for Manuscript ---")
    print(f"Latent Faults & {latent_faults_count} \\")
    print(f"False Alerts  & {false_alerts_count} \\")
    print("-------------------------------------")

    # --- Assertion ---
    expected_latent_faults = 96
    if latent_faults_count != expected_latent_faults:
        print("\n" + "="*70)
        print("!!!!!!!!!!!!!!!!!!!!!!!!! HUGE WARNING !!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Latent Faults count mismatch: Expected {expected_latent_faults}, Got {latent_faults_count}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("="*70 + "\n")
    else:
        print(f"\nAssertion Passed: Latent Faults count is {expected_latent_faults}.")


if __name__ == '__main__':
    audit_table_numbers()
