#!/usr/bin/env python3
"""
Debug script for fusion pipeline to identify and fix issues.
"""

import logging
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config_user import get_config
from data.fusion_dataset_generator import create_fusion_dataset
from utils.file_utils import setup_logging

def check_data_paths():
    """Check if required data paths exist."""
    cfg = get_config()
    
    print("🔍 Checking data paths...")
    
    # Check planets directory
    planets_dir = Path(cfg.data.planets_dir)
    if planets_dir.exists():
        print(f"✅ Planets directory exists: {planets_dir}")
        files = list(planets_dir.glob("*.csv"))
        print(f"   Found {len(files)} CSV files")
    else:
        print(f"❌ Planets directory missing: {planets_dir}")
        return False
    
    # Check false positives directory
    fp_dir = Path(cfg.data.false_positives_dir)
    if fp_dir.exists():
        print(f"✅ False positives directory exists: {fp_dir}")
        files = list(fp_dir.glob("*.csv"))
        print(f"   Found {len(files)} CSV files")
    else:
        print(f"❌ False positives directory missing: {fp_dir}")
        return False
    
    return True

def test_dataset_creation(max_files=10):
    """Test dataset creation with limited files."""
    print("\n🧪 Testing dataset creation...")
    
    cfg = get_config()
    try:
        dataset = create_fusion_dataset(
            cfg.data.planets_dir,
            cfg.data.false_positives_dir,
            max_files=max_files,
            ttv_dim=7,
            use_rv=False,
            use_imaging=False
        )
        
        if dataset is None:
            print("❌ Dataset creation returned None")
            return False
            
        print(f"✅ Dataset created successfully")
        print(f"   Samples: {len(dataset['y'])}")
        print(f"   Features - Transit image: {dataset['X_transit_image'].shape}")
        print(f"   Features - Transit timeseries: {dataset['X_transit_timeseries'].shape}")
        print(f"   Features - TTV: {dataset['X_ttv'].shape}")
        print(f"   Labels: {dataset['y'].shape}")
        
        # Check class balance
        unique, counts = zip(*[(int(k), int(v)) for k, v in dict(zip(*np.unique(dataset['y'], return_counts=True))).items()])
        print(f"   Class distribution: {dict(zip(unique, counts))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_config():
    """Create a minimal configuration for testing."""
    print("\n⚙️ Creating minimal test configuration...")
    
    cfg = get_config()
    
    # Override for minimal test
    cfg.training.batch_size = 8  # Small batch size for testing
    cfg.training.epochs = 2    # Very few epochs
    cfg.training.learning_rate = 0.001
    
    print(f"✅ Configuration updated:")
    print(f"   Batch size: {cfg.training.batch_size}")
    print(f"   Epochs: {cfg.training.epochs}")
    print(f"   Learning rate: {cfg.training.learning_rate}")
    
    return cfg

def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description='Debug fusion pipeline issues')
    parser.add_argument('--check-data', action='store_true', help='Check data paths')
    parser.add_argument('--test-dataset', action='store_true', help='Test dataset creation')
    parser.add_argument('--max-files', type=int, default=5, help='Max files for testing')
    
    args = parser.parse_args()
    
    print("🚀 Fusion Pipeline Debug Tool")
    print("=" * 50)
    
    success = True
    
    if args.check_data or not any([args.check_data, args.test_dataset]):
        success &= check_data_paths()
    
    if args.test_dataset or not any([args.check_data, args.test_dataset]):
        success &= test_dataset_creation(args.max_files)
    
    if success:
        print("\n✅ All checks passed! Pipeline should work.")
        print("\n🎯 Next steps:")
        print("   1. Run: python run_fusion_pipeline.py --batch_size 8 --max_files 20")
        print("   2. Check the results directory for outputs")
        print("   3. Monitor the log file for progress")
    else:
        print("\n❌ Issues found. Please fix the above problems before running the full pipeline.")

if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues
    main()
