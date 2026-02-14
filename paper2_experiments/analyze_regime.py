import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
SURVEY_RESULTS_PATH = 'paper2_experiments/survey_results_final.csv'
OUTPUT_PLOT_PATH = 'paper2_experiments/Figure1_Regime_Performance.png'
SHORT_PERIOD_THRESHOLD = 20.0
LONG_PERIOD_THRESHOLD = 100.0
HIGH_DISAGREEMENT_THRESHOLD = 0.5

def load_data(file_path):
    """
    Loads the survey results CSV, converts 'target_id' to integer,
    and removes rows with NaN 'period'.
    """
    if not os.path.exists(file_path):
        print(f"Error: Survey results file not found at {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}. Initial shape: {df.shape}")

    # Filter out rows where 'period' is NaN
    initial_rows = len(df)
    df.dropna(subset=['period'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with NaN 'period'. New shape: {df.shape}")
        
    # Ensure target_id is integer type
    df['target_id'] = pd.to_numeric(df['target_id'], errors='coerce').astype('Int64')
    # Ensure 'period' is numeric type
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    
    return df

def categorize_regime(period):
    """
    Categorizes the orbital period into 'Short', 'Medium', or 'Long' regimes.
    """
    if period < SHORT_PERIOD_THRESHOLD:
        return 'Short (<20d)'
    elif SHORT_PERIOD_THRESHOLD <= period <= LONG_PERIOD_THRESHOLD:
        return 'Medium (20-100d)'
    else:
        return 'Long (>100d)'

def calculate_disagreement_rates(df):
    """
    Calculates the fraction of candidates with high disagreement (>0.5) in each orbital period regime.
    """
    df['regime'] = df['period'].apply(categorize_regime)
    
    # Calculate total candidates per regime
    total_candidates_per_regime = df.groupby('regime').size()
    
    # Identify high disagreement candidates
    df['high_disagreement'] = (df['disagreement'] > HIGH_DISAGREEMENT_THRESHOLD)
    
    # Calculate high disagreement candidates per regime
    high_disagreement_per_regime = df[df['high_disagreement']].groupby('regime').size()
    
    # Calculate the fraction
    # Handle cases where a regime might have no candidates or no high disagreement candidates
    disagreement_fraction = high_disagreement_per_regime.div(total_candidates_per_regime).fillna(0)
    
    # Ensure all regimes are present, even if their fraction is 0
    all_regimes = ['Short (<20d)', 'Medium (20-100d)', 'Long (>100d)']
    disagreement_fraction = disagreement_fraction.reindex(all_regimes, fill_value=0)
    
    return disagreement_fraction

def plot_disagreement_rates(disagreement_rates):
    """
    Generates and saves a bar chart of disagreement rates by orbital period regime.
    """
    plt.figure(figsize=(10, 6))
    disagreement_rates.plot(kind='bar', color='skyblue')
    plt.title('Fraction of Candidates Flagged as High Disagreement by Orbital Period Regime')
    plt.xlabel('Orbital Period Regime')
    plt.ylabel('Fraction of High Disagreement Candidates (>0.5)')
    plt.ylim(0, 1) # Fraction should be between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"Plot saved to {OUTPUT_PLOT_PATH}")

def main():
    df_survey = load_data(SURVEY_RESULTS_PATH)
    if df_survey is None:
        return
    
    disagreement_rates = calculate_disagreement_rates(df_survey)
    print("\nCalculated Disagreement Rates:")
    print(disagreement_rates)
    
    plot_disagreement_rates(disagreement_rates)

if __name__ == "__main__":
    main()
