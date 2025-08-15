import os
import sys
import pickle
from pathlib import Path
from multi_method_fetcher import fetch_all_data_for_target

def scan_targets(target_dirs, output_path=None):
    multi_method_targets = []
    for target_dir in target_dirs:
        for fname in os.listdir(target_dir):
            if not fname.endswith('.fits'):
                continue
            fpath = os.path.join(target_dir, fname)
            data = fetch_all_data_for_target(fpath)
            if data is None:
                continue
            if (data['rv_mask'] > 0.5) or (data['imaging_mask'] > 0.5):
                multi_method_targets.append({
                    'kepler_id': data['kepler_id'],
                    'file': fpath,
                    'rv_mask': float(data['rv_mask']),
                    'imaging_mask': float(data['imaging_mask'])
                })
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(multi_method_targets, f)
    print(f"Found {len(multi_method_targets)} targets with RV or imaging data.")
    for t in multi_method_targets:
        print(f"KIC {t['kepler_id']} | {t['file']} | RV: {t['rv_mask']} | Imaging: {t['imaging_mask']}")

if __name__ == '__main__':
    # Example usage: python data/print_multi_method_targets.py ../exo-main/kepler_local_data/confirmed_planets ../exo-main/kepler_local_data/false_positives
    if len(sys.argv) < 2:
        print("Usage: python data/print_multi_method_targets.py <dir1> [<dir2> ...]")
        sys.exit(1)
    scan_targets(sys.argv[1:], output_path="results/multi_method_targets.pkl")
