import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from astropy.io import fits

from .types import TargetConfig, TransitHypothesis, PipelineStatus
from data.provenance import DataProvenance
from detection.transit_detector import find_transits_bls
import config as lite_config

logger = logging.getLogger(__name__)

# --- Mission-Specific Data Profiles ---
MISSION_PROFILES = {
    "TESS": {
        "time_col": "TIME",
        "flux_col": "PDCSAP_FLUX",
        "quality_col": "QUALITY",
        "id_keyword": "TICID",
        "sector_keyword": "SECTOR",
    },
    "KEPLER": {
        "time_col": "TIME",
        "flux_col": "PDCSAP_FLUX",
        "quality_col": "SAP_QUALITY",
        "id_keyword": "KEPLERID",
        "sector_keyword": "QUARTER",
    },
    "K2": {
        "time_col": "TIME",
        "flux_col": "PDCSAP_FLUX",
        "quality_col": "SAP_QUALITY",
        "id_keyword": "KEPLERID", # EPIC ID is in a different place
        "sector_keyword": "CAMPAIGN",
    }
}

def _create_data_views(time: np.ndarray, flux: np.ndarray, signal: Dict) -> Dict[str, np.ndarray]:
    """
    **ARCHITECTURAL PLACEHOLDER**
    This function must be implemented to generate the data views expected by the Bayesian model.
    The current model expects a single feature vector of length 512.
    """
    logger.warning("Using placeholder function `_create_data_views`. True implementation is required.")
    feature_vector_length = 512
    return {'global_view': np.random.rand(feature_vector_length)}


class IngestionEngine:
    """
    Reads FITS files from various missions, hunts for transit signals, and
    packages them into mission-aware TransitHypothesis objects.
    """
    def __init__(self):
        logger.info("IngestionEngine initialized.")

    def scan_target(self, target: TargetConfig) -> List[TransitHypothesis]:
        """
        Processes a light curve from a FITS file, identifies potential transit
        signals, and creates a list of TransitHypothesis objects.
        """
        logger.info(f"Scanning target file: {target.source_file}")
        
        try:
            with fits.open(target.source_file, memmap=False) as hdul:
                primary_header = hdul[0].header
                lc_data = hdul[1].data
                
                mission = primary_header.get('TELESCOP', 'Unknown').upper()
                profile = MISSION_PROFILES.get(mission)

                if not profile:
                    logger.error(f"Unsupported mission '{mission}' in file {target.source_file}. No profile found.")
                    return []

                time_lc = lc_data[profile['time_col']]
                flux_lc = lc_data[profile['flux_col']]
                
                # --- Create DataProvenance Object ---
                provenance = DataProvenance(
                    source=mission,
                    instrument_id=str(primary_header.get(profile['id_keyword'], 'UnknownID')),
                    cadence_s=float(np.nanmedian(np.diff(time_lc)) * 24 * 3600 if len(time_lc) > 1 else -1.0),
                    mission_quarter=primary_header.get('QUARTER'),
                    mission_sector=primary_header.get('SECTOR'),
                    mission_campaign=primary_header.get('CAMPAIGN'),
                    injected='INJECT' in primary_header and primary_header['INJECT'],
                )
                # ---

        except Exception as e:
            logger.error(f"Failed to read or process FITS file {target.source_file}: {e}")
            return []

        # Basic data cleaning
        mask = np.isfinite(time_lc) & np.isfinite(flux_lc)
        time_lc, flux_lc = time_lc[mask], flux_lc[mask]
        
        if np.std(flux_lc) > 1e-9:
            flux_lc = (flux_lc - np.mean(flux_lc)) / np.std(flux_lc)
        else:
            flux_lc = np.zeros_like(flux_lc)
        
        # --- Detect Signals using BLS ---
        transit_info, periodicity_data = find_transits_bls(time_lc, flux_lc)
        
        if not transit_info or not transit_info.get('times'):
            logger.info(f"No significant signals found for {provenance.instrument_id}.")
            return []

        logger.info(f"Found {len(transit_info['times'])} potential signals for {provenance.instrument_id}.")

        # --- Process each signal into a TransitHypothesis ---
        hypotheses = []
        bls_period = periodicity_data.get('median_period')
        if not bls_period:
            logger.error("BLS did not provide a median_period. Cannot create hypotheses.")
            return []

        for i in range(len(transit_info['times'])):
            signal = {
                'period': bls_period,
                't0': transit_info['times'][i],
                'duration': transit_info['durations'][i],
                'depth': transit_info['depths'][i],
            }
            try:
                hypo_id = f"{provenance.instrument_id}_P{signal['period']:.5f}_T{signal['t0']:.2f}"
                data_views = _create_data_views(time_lc, flux_lc, signal)

                hypothesis = TransitHypothesis(
                    id=hypo_id,
                    mission=mission,
                    provenance=provenance,
                    signal_params=signal,
                    data_views=data_views,
                    status=PipelineStatus.DETECTED
                )
                hypothesis.log(f"Signal detected by IngestionEngine from {mission} data.")
                hypotheses.append(hypothesis)
            except Exception as e:
                logger.error(f"Error processing signal into hypothesis: {e}", exc_info=True)

        return hypotheses