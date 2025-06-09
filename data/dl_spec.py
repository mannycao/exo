# Save as bulk_download_kepler.py
import os
import requests
import tarfile
import logging
from tqdm import tqdm  # Install with pip install tqdm if needed
import config

def setup_logging():
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def download_kepler_quarter(quarter=1, output_dir=None):
    """
    Download a quarter of Kepler data directly from MAST bulk download service.
    
    Args:
        quarter: Kepler quarter number (1-17)
        output_dir: Directory to save files
    """
    logger = logging.getLogger(__name__)
    
    if output_dir is None:
        output_dir = config.LIGHT_CURVE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for Kepler bulk downloads (quarter 1 example)
    base_url = "https://archive.stsci.edu/pub/kepler/lightcurves"
    
    # Quarter specific directory
    q_str = str(quarter).zfill(2)
    quarter_dir = f"q{q_str}"
    
    # List of subdirectories to try (Kepler IDs are divided by ranges)
    # Each subdirectory contains ~1000 light curves
    ranges_to_try = [
        "0000", "1000", "2000", "3000", "4000", "5000",
        "6000", "7000", "8000", "9000", "10000", "11000"
    ]
    
    downloaded_files = []
    
    for range_dir in ranges_to_try[:2]:  # Limit to first two ranges for testing
        # Construct the URL for this range
        url = f"{base_url}/{quarter_dir}/{range_dir}"
        logger.info(f"Checking data in {url}")
        
        try:
            # First, let's see what tar files are available in this directory
            index_url = f"{url}/kplr"
            
            # This is a basic way to list directory contents - may need to be adjusted
            # based on how the MAST archive is structured
            response = requests.get(index_url)
            
            if response.status_code != 200:
                logger.warning(f"Could not access {index_url}, skipping")
                continue
            
            # Find tar files in the response
            import re
            tar_files = re.findall(r'href="(kplr\d+.*?\.tar)"', response.text)
            
            if not tar_files:
                logger.warning(f"No tar files found in {url}")
                continue
            
            # Download a few tar files (limit for testing)
            for tar_file in tar_files[:5]:
                tar_url = f"{url}/{tar_file}"
                local_tar_path = os.path.join(output_dir, tar_file)
                
                logger.info(f"Downloading {tar_url}")
                
                # Download tar file
                with requests.get(tar_url, stream=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get('content-length', 0))
                    
                    with open(local_tar_path, 'wb') as f, tqdm(
                        desc=tar_file,
                        total=total,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)
                
                # Extract tar file
                logger.info(f"Extracting {local_tar_path}")
                with tarfile.open(local_tar_path) as tar:
                    tar.extractall(path=output_dir)
                
                # Find extracted light curve files
                extracted_files = [
                    os.path.join(output_dir, f) for f in os.listdir(output_dir)
                    if f.endswith('.fits') and 'llc.fits' in f  # Long cadence light curves
                ]
                downloaded_files.extend(extracted_files)
                
                # Remove tar file after extraction
                os.remove(local_tar_path)
        
        except Exception as e:
            logger.error(f"Error downloading from {url}: {e}")
    
    logger.info(f"Downloaded and extracted {len(downloaded_files)} light curve files")
    return downloaded_files

if __name__ == "__main__":
    setup_logging()
    # Download Quarter 1 Kepler data
    download_kepler_quarter(quarter=1)