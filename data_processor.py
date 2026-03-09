import os
import requests
import zipfile
import shutil
import numpy as np
# rasterio is no longer needed as we're using PIL for PNGs
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# --- CONFIGURATION ---
# Official TU Munich Link for SEN1-2 (Spring Subset)
# DATA_URL is commented out as local data is provided
# DATA_URL = "https://mediatum.ub.tum.de/doc/1436631/1436631.zip" 

BASE_DIR = Path("./processed_sen12_data") # Changed base directory for processed output
RAW_DIR = Path("./ROIs1158_spring") # Point directly to the downloaded data

PROCESSED_DIR = BASE_DIR / "processed" # Keep processed data separate

SUBSET_LIMIT = 15000 # Stop after this many pairs

def setup_directories():
    """Creates the standard folder structure for PyTorch ImageFolder."""
    for split in ['train', 'val', 'test']:
        for mod in ['eo', 'sar']:
            (PROCESSED_DIR / split / mod).mkdir(parents=True, exist_ok=True)

def normalize_sar(sar_array):
    """
    Systems Engineering Note: SAR data is high dynamic range. 
    Assuming input is already 0-255 uint8 from PNG, no further normalization needed.
    If original TIFs were different, this would require re-evaluation.
    """
    # Assuming PNGs are already scaled to 0-255 or similar range,
    # and likely already clipped. Just ensure type.
    return sar_array.astype(np.uint8)

def normalize_eo(eo_array):
    """
    Converts Sentinel-2 raw bands to visual RGB.
    Assuming input is already 0-255 uint8 from PNG, no further normalization needed.
    """
    # Assuming PNGs are already scaled to 0-255 or similar range.
    return eo_array.astype(np.uint8)

def process_and_split():
    """
    Iterates through raw SEN1-2 folders, pairs S1/S2, and moves them to processed.
    """
    print("🚀 Starting Systems Engineering Data Pipeline.")
    setup_directories()
    
    # Locate all Spring scenes (customize based on actual extraction)
    # Assuming user extracted SEN1-2 into ./ROIs1158_spring
    # Structure is: ROIs1158_spring/s1_X/ROIs1158_spring_s1_X_pY.png
    
    pairs = []
    
    # Walk through raw directory to find paired patches
    print("🔍 Indexing patch pairs...")
    # Adjusted glob pattern for PNGs nested in s1_X folders
    s1_files = sorted(list(RAW_DIR.rglob("s1_*/*_s1_*.png")))
    
    if not s1_files:
        print(f"❌ No S1 PNG files found in {RAW_DIR}. Please check the path and file structure.")
        return

    # Using tqdm with a limit
    for s1_path in tqdm(s1_files, desc="Finding pairs", total=min(len(s1_files), SUBSET_LIMIT)):
        if len(pairs) >= SUBSET_LIMIT:
            break
        # Infer S2 path (replace 's1' with 's2' in filename, and 's1_X' folder with 's2_X' folder)
        s2_path = s1_path.parent.parent / s1_path.parent.name.replace("s1_", "s2_") / s1_path.name.replace("_s1_", "_s2_")
        
        if s2_path.exists():
            pairs.append((s1_path, s2_path))
    
    print(f"✅ Found {len(pairs)} valid EO-SAR integrity pairs.")
    
    if not pairs:
        print("❌ No pairs found to process. Exiting.")
        return

    # Split: 70% Train, 15% Val, 15% Test
    np.random.seed(42) # for reproducibility
    np.random.shuffle(pairs)
    n = len(pairs)
    train_idx = int(0.7 * n)
    val_idx = int(0.85 * n)
    
    splits = {
        'train': pairs[:train_idx],
        'val': pairs[train_idx:val_idx],
        'test': pairs[val_idx:]
    }
    
    print("⚙️  Processing and Normalizing (This creates the Nominal Integrity Baseline)...")
    
    for split_name, split_pairs in splits.items():
        for s1_file, s2_file in tqdm(split_pairs, desc=f"Building {split_name}"):
            try:
                # Read SAR (Single Channel PNG, assume grayscale)
                sar_pil_img = Image.open(s1_file).convert('L') # 'L' mode for grayscale
                sar_data_array = np.array(sar_pil_img)
                sar_img_processed = Image.fromarray(normalize_sar(sar_data_array), mode='L')
                
                # Read Optical (RGB PNG)
                eo_pil_img = Image.open(s2_file).convert('RGB') # 'RGB' mode
                eo_data_array = np.array(eo_pil_img)
                eo_img_processed = Image.fromarray(normalize_eo(eo_data_array), mode='RGB')
                
                # Save
                # Construct filename like "ROIs1158_spring_0_p1.jpg"
                fname = s1_file.name.replace("_s1_", "_").replace(".png", ".jpg")
                sar_img_processed.save(PROCESSED_DIR / split_name / "sar" / fname)
                eo_img_processed.save(PROCESSED_DIR / split_name / "eo" / fname)
                
            except Exception as e:
                print(f"Skipped {s1_file.name}: {e}")
                # In Systems Engineering, we log failures but continue processing
                pass

    print("\n✅ Pipeline Complete.")
    print(f"   Train: {len(splits['train'])}")
    print(f"   Val:   {len(splits['val'])}")
    print(f"   Test:  {len(splits['test'])}")
    print(f"   Data ready at: {PROCESSED_DIR}")

if __name__ == "__main__":
    # --- MANUAL STEP ---
    # 1. Ensure "ROIs1158_spring" is in the same directory as this script.
    # 2. Run this script.
    process_and_split()
