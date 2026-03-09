import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SEN12Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.eo_dir = self.root_dir / 'eo'
        self.sar_dir = self.root_dir / 'sar'
        
        # Robustly find valid pairs
        eo_files = set(f.name for f in self.eo_dir.glob('*.jpg'))
        sar_files = set(f.name for f in self.sar_dir.glob('*.jpg'))
        self.filenames = sorted(list(eo_files.intersection(sar_files)))
        
        self.transform = transform
        
        # --- DIAGNOSTIC: Check Class Balance immediately ---
        print(f"   [Dataset Debug] Scanning {split} set for class balance...")
        urban_count = 0
        nature_count = 0
        
        # We sample the first 500 images to estimate the balance quickly
        sample_size = min(len(self.filenames), 500)
        for i in range(sample_size):
            fname = self.filenames[i]
            sar_path = self.sar_dir / fname
            try:
                # Load raw to check intensity
                sar_img = Image.open(sar_path).convert('L')
                if np.array(sar_img).mean() > 80: # Threshold: 80/255 (~30% brightness)
                    urban_count += 1
                else:
                    nature_count += 1
            except:
                pass
                
        print(f"   [Dataset Debug] Estimated Balance in {split}: {urban_count} Bright (Urban) / {nature_count} Dark (Nature)")
        if urban_count == 0 or nature_count == 0:
            print("   ⚠️ WARNING: Dataset is still unbalanced! Adjust the threshold (80) in dataset.py")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        eo_path = self.eo_dir / fname
        sar_path = self.sar_dir / fname
        
        eo_img = Image.open(eo_path).convert('RGB')
        sar_img = Image.open(sar_path).convert('L') 
        
        # --- PHYSICS-BASED LABELING ---
        # Calculate mean brightness of the Radar image (0-255)
        # Urban areas are typically bright (>80), Nature is dark (<80)
        sar_val = np.array(sar_img).mean()
        label = 1.0 if sar_val > 80 else 0.0
        
        if self.transform:
            eo_tensor = self.transform(eo_img)
            sar_tensor = self.transform(sar_img)
        else:
            to_tensor = transforms.ToTensor()
            eo_tensor = to_tensor(eo_img)
            sar_tensor = to_tensor(sar_img)
            
        return eo_tensor, sar_tensor, label