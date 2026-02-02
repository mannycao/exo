"""
DataSentinel: Monitors for new TESS/Kepler data on the filesystem.

This module scans designated directories for new light curve FITS files,
parses metadata (TIC ID, Sector) from filenames, and yields TargetConfig
objects for the pipeline to process.
"""

import os
import re
import logging
from typing import List, Optional
from pathlib import Path

from .types import TargetConfig

logger = logging.getLogger(__name__)


class DataSentinel:
    """
    Watches for new FITS data files and yields targets for processing.
    
    Supports both local filesystem monitoring and (placeholder) MAST API queries.
    """
    
    # Regex patterns for common TESS/Kepler FITS filenames
    TESS_FILENAME_PATTERN = re.compile(
        r"tess(\d{4})(\d{2})(\d{2})-s(\d{4})"  # tess20240901-s0015...
        r".*?_(\d+)_lc\.fits",  # ...tic123456789_lc.fits
        re.IGNORECASE
    )
    
    # Alternative TESS pattern (simpler)
    TESS_SIMPLE_PATTERN = re.compile(
        r"tic_(\d+).*s(\d+).*\.fits",
        re.IGNORECASE
    )
    
    # Kepler pattern
    KEPLER_PATTERN = re.compile(
        r"kplr(\d+).*\.fits",
        re.IGNORECASE
    )
    
    def __init__(self, registry=None):
        """
        Initialize the sentinel.
        
        Args:
            registry: Optional TargetRegistry instance for de-duplication.
        """
        self.registry = registry
        logger.info("DataSentinel initialized.")
    
    def check_for_new_data(self, watch_folder: str) -> List[TargetConfig]:
        """
        Scan a directory for new FITS files and yield TargetConfig objects.
        
        This function:
        1. Lists all .fits files in the watch folder.
        2. Parses TIC ID and Sector from filenames.
        3. Consults registry (if available) to skip already-processed targets.
        4. Returns list of TargetConfig objects for new targets.
        
        Args:
            watch_folder: Path to the directory to scan.
        
        Returns:
            List of TargetConfig objects for new data.
        """
        new_targets = []
        
        if not os.path.isdir(watch_folder):
            logger.warning(f"Watch folder does not exist: {watch_folder}")
            return new_targets
        
        try:
            # Find all FITS files
            fits_files = list(Path(watch_folder).glob("**/*.fits"))
            logger.debug(f"Found {len(fits_files)} FITS files in {watch_folder}")
            
            for fits_path in fits_files:
                filename = fits_path.name
                
                # Try to parse the filename
                tic_id, sector = self._parse_filename(filename)
                
                if tic_id is None:
                    logger.debug(f"Could not parse TIC ID / Sector from {filename}")
                    continue
                
                # Check registry if available
                if self.registry and self.registry.is_processed(tic_id, sector):
                    logger.debug(f"TIC {tic_id} Sector {sector} already processed. Skipping.")
                    continue
                
                # Create a TargetConfig
                config = TargetConfig(
                    tic_id=tic_id,
                    mission="TESS" if tic_id > 10**8 else "Kepler",  # Heuristic
                    sector=str(sector)
                )
                new_targets.append(config)
                logger.info(f"Discovered new target: TIC {tic_id} Sector {sector}")
        
        except Exception as e:
            logger.error(f"Error scanning watch folder {watch_folder}: {e}", exc_info=True)
        
        return new_targets
    
    def _parse_filename(self, filename: str) -> tuple:
        """
        Extract TIC ID and Sector from a FITS filename.
        
        Tries multiple patterns (TESS, Kepler).
        
        Args:
            filename: The filename to parse.
        
        Returns:
            Tuple of (tic_id, sector) or (None, None) if parsing fails.
        """
        # Try TESS with full pattern first
        match = self.TESS_FILENAME_PATTERN.search(filename)
        if match:
            tic_id = int(match.group(5))
            sector = int(match.group(4))
            logger.debug(f"Parsed TESS (full): TIC {tic_id}, Sector {sector}")
            return tic_id, sector
        
        # Try TESS with simple pattern
        match = self.TESS_SIMPLE_PATTERN.search(filename)
        if match:
            tic_id = int(match.group(1))
            sector = int(match.group(2))
            logger.debug(f"Parsed TESS (simple): TIC {tic_id}, Sector {sector}")
            return tic_id, sector
        
        # Try Kepler
        match = self.KEPLER_PATTERN.search(filename)
        if match:
            kic_id = int(match.group(1))
            logger.debug(f"Parsed Kepler: KIC {kic_id}")
            # Kepler doesn't have sectors, use 0 as placeholder
            return kic_id, 0
        
        return None, None
    
    def poll_mast(self, sector: int, mission: str = "TESS") -> List[TargetConfig]:
        """
        Query NASA MAST for light curves in a specific sector.
        
        **PLACEHOLDER**: This is a stub for future integration with astroquery.
        
        Args:
            sector: The observation sector to query.
            mission: Mission name ("TESS" or "Kepler").
        
        Returns:
            List of TargetConfig objects for newly available targets.
        """
        logger.info(f"MAST poll requested for {mission} Sector {sector} (stub implementation)")
        
        # TODO: Future implementation using astroquery.mast.Observations
        # Example structure:
        # from astroquery.mast import Observations
        # obs_table = Observations.query_criteria(
        #     obs_collection=mission,
        #     obs_id=f"*{sector}*",
        #     dataproduct_type="timeseries"
        # )
        # for row in obs_table:
        #     tic_id = int(row['target_name'].split('-')[0])
        #     target = TargetConfig(tic_id=tic_id, mission=mission, sector=str(sector))
        #     yield target
        
        return []
    
    def verify_fits_integrity(self, fits_path: str) -> bool:
        """
        Check if a FITS file is readable and not corrupted.
        
        Args:
            fits_path: Path to the FITS file.
        
        Returns:
            True if file is readable, False otherwise.
        """
        try:
            from astropy.io import fits
            with fits.open(fits_path) as hdul:
                # Just check that we can read the header of the primary HDU
                _ = hdul[0].header
            logger.debug(f"FITS integrity check passed: {fits_path}")
            return True
        except Exception as e:
            logger.warning(f"FITS integrity check failed for {fits_path}: {e}")
            return False
