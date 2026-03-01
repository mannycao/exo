import os
import shutil
import time
import glob
import re
import logging
import requests
from datetime import datetime
from typing import List, Generator, Any, Dict, Tuple

try:
    from astroquery.mast import Observations
    from astropy.time import Time
    # Set a longer timeout for MAST queries
    Observations.TIMEOUT = 1800 # 30 minutes
except ImportError:
    print("Warning: astroquery not found. Mocking Observations.")
    class MockObservations:
        def query_criteria(self, *args, **kwargs):
            return []
        def get_product_list(self, *args, **kwargs):
            return []
        def download_products(self, *args, **kwargs):
            return {"Status": "No Results"}
    Observations = MockObservations()
    class MockTime: # Mock astropy.time.Time as well
        def __init__(self, *args, **kwargs):
            pass
        @property
        def mjd(self):
            return 0
    Time = MockTime


from .registry import TargetRegistry
from .types import TargetConfig

# --- Helper for direct file download ---
def _download_file_direct(url: str, local_path: str, logger: logging.Logger):
    """Downloads a single file from a URL to a local path using requests."""
    try:
        if os.path.exists(local_path):
            logger.debug(f"File already exists, skipping: {os.path.basename(local_path)}")
            return True

        logger.debug(f"Attempting direct download of {url} to {os.path.basename(local_path)}")
        with requests.get(url, stream=True, timeout=300, verify=False) as r: # Increased timeout and added verify=False
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {os.path.basename(local_path)}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading {url}: {e}", exc_info=True)
        return False

class DiskSpaceManager:
    """Monitors and manages disk space for a given directory."""
    def __init__(self, download_dir: str, max_usage_pct: float = 85.0):
        self.download_dir = download_dir
        self.max_usage_pct = max_usage_pct
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def has_sufficient_space(self) -> bool:
        """Checks if disk usage is below the maximum percentage."""
        total, used, _ = shutil.disk_usage(self.download_dir)
        usage_pct = (used / total) * 100
        if usage_pct >= self.max_usage_pct:
            return False
        return True

class DataSentinel:
    """
    Manages querying external archives (MAST) for new data, downloading it,
    and scanning local directories for unprocessed files.
    """
    def __init__(self, registry: TargetRegistry, download_dir: str, max_disk_usage_pct: float = 85.0):
        self.registry = registry
        self.logger = self._setup_logger()
        self.disk_manager = DiskSpaceManager(download_dir, max_disk_usage_pct)
        self.download_dir = download_dir
        # Regex for parsing local filenames
        self.filename_parsers = {
            'TESS': re.compile(r'tess-(\d+)-s(\d+)\.fits'), # Format: tess-<TICID>-s<SECTOR>.fits
            'KEPLER': re.compile(r'kepler-(\d+)-q(\d+)\.fits'), # Format: kepler-<KICID>-q<QUARTER>.fits
            'K2': re.compile(r'k2-(\d+)-c(\d+)\.fits') # Format: k2-<EPICID>-c<CAMPAIGN>.fits
        }

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _get_unique_obs_identifier(self, obs: Any, mission: str) -> int:
        """
        Extracts a unique numerical identifier (sector, quarter, campaign) from an MAST observation.
        Returns 0 if not found.
        """
        if mission == "TESS":
            # Prefer 'sequence_number' for TESS, which often corresponds to the sector
            tess_id = obs.get('sequence_number')
            if tess_id is not None and tess_id != 0:
                return tess_id
            # Fallback to parsing from obs_id if 'sequence_number' is not directly available or is 0
            match = re.search(r'-s(\d+)', obs.get('obs_id', ''), re.IGNORECASE)
            return int(match.group(1)) if match else 0
        elif mission == "KEPLER":
            return obs.get('quarter', 0)
        elif mission == "K2":
            return obs.get('campaign', 0)
        return 0 # Default for other missions or if pattern not found

    def _get_fits_download_url(self, obsid: str, mission: str) -> str | None:
        """Helper to get the direct FITS light curve download URL from MAST."""
        products = []
        try:
            products = Observations.get_product_list(obsid)
        except Exception as e:
            self.logger.error(f"Error getting product list for obsid {obsid}: {e}")
            return None

        # Filter for light curve FITS files
        lc_fits_products = [
            p for p in products
            if p['productFilename'].lower().endswith('.fits') and ('lc.fits' in p['productFilename'].lower() or 'llc.fits' in p['productFilename'].lower())
        ]
        
        if not lc_fits_products:
            self.logger.debug(f"No suitable FITS light curve product found for obsid: {obsid}. Full product list: {products}")
            return None
        
        # Prioritize 'llc' (long cadence) or 'lc' if both exist
        # and prioritize specific mission light curves (e.g., kplr*.fits)
        def product_priority(p):
            score = 0
            filename = p['productFilename'].lower()
            if mission.lower() in filename: # e.g. 'kplr' in kplr_lc.fits
                score += 2
            if 'llc' in filename:
                score += 1
            return score

        lc_fits_products.sort(key=product_priority, reverse=True)
        return lc_fits_products[0]['dataURI']


    def _parse_filename(self, filename: str) -> tuple[str | None, int | None, str | None]:
        """
        Parses target_id, sector/campaign, and mission from a local FITS filename.
        Returns (target_id, sector, mission) or (None, None, None) if parsing fails.
        """
        # TESS: tess-<TICID>-s<SECTOR>.fits
        match = re.search(self.filename_parsers['TESS'], filename, re.IGNORECASE)
        if match:
            return str(match.group(1)), int(match.group(2)), "TESS"

        # KEPLER: kepler-<KICID>-q<QUARTER>.fits
        match = re.search(self.filename_parsers['KEPLER'], filename, re.IGNORECASE)
        if match:
            return str(match.group(1)), int(match.group(2)), "KEPLER"

        # K2: k2-<EPICID>-c<CAMPAIGN>.fits
        match = re.search(self.filename_parsers['K2'], filename, re.IGNORECASE)
        if match:
            return str(match.group(1)), int(match.group(2)), "K2"

        self.logger.warning(f"Could not parse mission, target_id, or sector from filename: {filename}")
        return None, None, None


    def poll_mast_and_download(self, missions: List[str], start_date: str, max_downloads: int = 100, tess_sector: int = None):
        """
        Polls MAST for new light curve data from specified missions since a given date
        and downloads it using a direct requests-based approach.
        """
        self.logger.info(f"Polling MAST for missions {missions} with data starting on or after {start_date}.")
        
        download_count = 0
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Convert to astropy Time objects and then to MJD
        start_mjd = Time(start_date_dt).mjd
        now_mjd = Time(datetime.now()).mjd

        for mission in missions:
            # Check overall download limit
            if download_count >= max_downloads:
                self.logger.info("Overall download limit reached. Stopping further downloads.")
                break
            
            self.logger.info(f"Querying {mission} data...")
            try:
                query_params = {
                    "obs_collection": mission,
                    "dataproduct_type": "timeseries",
                    "t_min": [start_mjd, now_mjd]
                }
                if mission == "TESS" and tess_sector is not None:
                    query_params["sequence_number"] = tess_sector

                obs_table = Observations.query_criteria(**query_params)

                if not obs_table:
                    self.logger.info(f"No new observations found for {mission} matching criteria.")
                    continue

                self.logger.info(f"Found {len(obs_table)} potential observations for {mission}.")
                
                # Sort observations by date to prioritize newer ones, or consistent behavior
                obs_table.sort('t_min')

                for idx, obs in enumerate(obs_table):
                    # Check individual download limit within this mission's loop too
                    if download_count >= max_downloads:
                        break

                    # --- Debug Logging for each observation ---
                    self.logger.debug(f"--- Processing obs {idx+1}/{len(obs_table)} for {mission} ---")
                    self.logger.debug(f"  obsid: {obs.get('obsid', 'N/A')}, target_name: {obs.get('target_name', 'N/A')}")
                    self.logger.debug(f"  Available obs keys: {obs.keys()}")
                    if mission == "TESS": # Added debug for TESS specific keys
                        self.logger.debug(f"  TESS obs_id: {obs.get('obs_id', 'N/A')}")
                        self.logger.debug(f"  TESS s_region: {obs.get('s_region', 'N/A')}")
                        self.logger.debug(f"  TESS sequence_number: {obs.get('sequence_number', 'N/A')}")
                    # --- End Debug Logging ---

                    target_id_from_mast = str(obs['target_name'])
                    if not target_id_from_mast.isdigit():
                        self.logger.debug(f"Skipping non-numeric target_name: {target_id_from_mast}")
                        continue
                    
                    # Extract unique mission-specific identifier (sector/quarter/campaign)
                    unique_obs_id = self._get_unique_obs_identifier(obs, mission)
                    self.logger.debug(f"  Extracted unique_obs_id: {unique_obs_id} (from {mission})")

                    if unique_obs_id == 0:
                        self.logger.warning(f"Could not determine unique mission ID from MAST obs {obs.get('obsid', 'N/A')} for {mission}. Skipping.")
                        continue

                    # Check if this target has already been processed (using the unique key)
                    is_processed_status = self.registry.is_processed(target_id_from_mast, unique_obs_id)
                    self.logger.debug(f"  Registry check for {target_id_from_mast}, {unique_obs_id}: {is_processed_status}")
                    if is_processed_status:
                        self.logger.debug(f"Skipping already processed target: {mission} {target_id_from_mast}, ID {unique_obs_id}")
                        continue

                    disk_space_status = self.disk_manager.has_sufficient_space()
                    self.logger.debug(f"  Disk has sufficient space: {disk_space_status}")
                    if not disk_space_status:
                        self.logger.warning("Disk space limit reached. Pausing all downloads for now.")
                        return # Stop all downloads
                    
                    # Get the direct download URL for the FITS light curve
                    download_url = self._get_fits_download_url(obs['obsid'], mission)
                    self.logger.debug(f"  Download URL found: {download_url}")

                    if download_url:
                        # Construct a clean filename using mission, target_id, and unique_obs_id
                        # Example: tess-12345678-s01.fits, kepler-9876543-q05.fits, k2-200000000-c04.fits
                        # Ensure 'q' for Kepler quarters and 'c' for K2 campaigns in filename
                        mission_prefix = mission.lower()
                        if mission == "KEPLER":
                            id_tag = "q"
                        elif mission == "K2":
                            id_tag = "c"
                        else: # TESS
                            id_tag = "s"

                        base_filename = f"{mission_prefix}-{target_id_from_mast}-{id_tag}{unique_obs_id:02d}.fits"
                        local_path = os.path.join(self.download_dir, base_filename)

                        self.logger.info(f"Downloading {mission} {target_id_from_mast}, ID {unique_obs_id} to {local_path}")
                        if _download_file_direct(download_url, local_path, self.logger):
                            download_count += 1
                        else:
                            self.logger.error(f"Failed to download FITS light curve for {mission} {target_id_from_mast}, {unique_obs_id} from {download_url}")
                    else:
                        self.logger.warning(f"No direct FITS download URL found for {mission} {target_id_from_mast}, {unique_obs_id}. Skipping download attempt.")


            except Exception as e:
                self.logger.error(f"Error polling {mission} data: {e}", exc_info=True)

    def _parse_kepler_sh_urls(self, kepler_sh_path: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Parses a local Kepler.sh script for wget commands that download FITS files
        and extracts Kepler IDs and their corresponding URLs.
        Returns a dictionary: {kic_id: [(output_filename, url), ...]}
        """
        self.logger.info(f"Parsing {kepler_sh_path} for wget commands...")
        wget_commands_by_kic = {}
        try:
            with open(kepler_sh_path, 'r') as f:
                for line in f:
                    # Regex to extract output filename (group 1), KIC ID (group 2), and full URL (group 3)
                    match = re.search(r"wget -O '(kplr(\d+)-.*?\.fits)' '(http://exoplanetarchive\.ipac\.caltech\.edu:80/data/ETSS//Kepler/\d+/\d+/\d+/kplr\2-.*?\.fits)'", line)
                    if match:
                        output_filename_with_ext = match.group(1)
                        kic_id_full_str = match.group(2)
                        full_url = match.group(3)

                        kic_id = str(int(kic_id_full_str)) # Convert to int and back to str to remove leading zeros if any

                        if kic_id not in wget_commands_by_kic:
                            wget_commands_by_kic[kic_id] = []
                        wget_commands_by_kic[kic_id].append((output_filename_with_ext, full_url))
        except FileNotFoundError:
            self.logger.error(f"Error: {kepler_sh_path} not found. Please ensure it's in the correct directory.")
            return {}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while parsing {kepler_sh_path}: {e}")
            return {}
        
        self.logger.info(f"Found {len(wget_commands_by_kic)} unique KIC IDs with FITS wget commands in {kepler_sh_path}.")
        return wget_commands_by_kic

    def poll_from_kepler_sh_and_download_kepler(self, kepler_sh_path: str, kepler_ids_to_download: List[str]):
        """
        Downloads Kepler FITS files from URLs specified in a local Kepler.sh script,
        filtered by a provided list of Kepler IDs.
        """
        self.logger.info(f"Downloading Kepler data using URLs from {kepler_sh_path}, filtered by provided KIC IDs.")
        
        wget_urls_by_kic = self._parse_kepler_sh_urls(kepler_sh_path)
        download_count = 0

        for kic_id in kepler_ids_to_download:
            if kic_id not in wget_urls_by_kic:
                self.logger.warning(f"KIC ID {kic_id} not found in {kepler_sh_path}. Skipping.")
                continue

            kic_download_dir = os.path.join(self.download_dir, kic_id)
            os.makedirs(kic_download_dir, exist_ok=True)

            for output_filename, file_url in wget_urls_by_kic[kic_id]:
                local_path = os.path.join(kic_download_dir, output_filename)
                unique_obs_id = 0 # Cannot derive quarter/sector/campaign from this specific URL structure

                # Check if already processed using the registry (KICID + 0 as unique_obs_id for now)
                if self.registry.is_processed(kic_id, unique_obs_id):
                    self.logger.debug(f"Skipping already processed file for KIC {kic_id}: {output_filename}")
                    continue

                if self.disk_manager.has_sufficient_space():
                    self.logger.info(f"Downloading {output_filename} for KIC {kic_id} from {file_url}")
                    if _download_file_direct(file_url, local_path, self.logger):
                        download_count += 1
                        self.registry.mark_processed(kic_id, unique_obs_id, "Downloaded", datetime.utcnow().isoformat())
                    else:
                        self.logger.error(f"Failed to download {output_filename} for KIC {kic_id}.")
                else:
                    self.logger.warning("Disk space limit reached. Stopping downloads.")
                    return
        
        self.logger.info(f"Finished downloading from {kepler_sh_path}. Total files downloaded: {download_count}")