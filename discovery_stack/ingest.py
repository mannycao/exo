import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from astropy.io import fits

# Import existing components from the 'lite' codebase
from data.enhanced_data_fetcher import download_kepler_light_curve
from detection.transit_detector import find_transits_bls
import config as lite_config

# Import shared types and provenance
from .types import TargetConfig, TransitHypothesis, PipelineStatus
from data.provenance import DataProvenance

logger = logging.getLogger(__name__)

def _create_data_views(time: np.ndarray, flux: np.ndarray, signal: Dict) -> Dict[str, np.ndarray]:
    """
    **ARCHITECTURAL PLACEHOLDER**

    This function is the critical missing link. It must take a raw light curve
    and a detected signal (period, t0, duration) and generate the data view
    expected by the Bayesian model.

    The KerasTransformerEncoder model expects a single input of shape (BATCH_SIZE, 512).
    Therefore, this function should return a single feature vector of length 512.

    Args:
        time (np.ndarray): The time array of the light curve.
        flux (np.ndarray): The flux array of the light curve.
        signal (Dict): A dictionary containing 'period', 't0', 'duration'.

    Returns:
        A dictionary containing a single 'global_view' as a numpy array of shape (512,).
    """
    logger.warning("Using placeholder function `_create_data_views`. True implementation is required.")
    
    # KerasTransformerEncoder expects a single input of shape (BATCH_SIZE, 512)
    feature_vector_length = 512 # Based on the error message build_config input_shape
    return {
        'global_view': np.random.rand(feature_vector_length),
    }


class IngestionEngine:
    """
    Hunts for transit signals in a target's light curve and packages them
    into TransitHypothesis objects.
    """
    def __init__(self):
        # This engine is stateless for now, but could hold BLS configs
        logger.info("IngestionEngine initialized.")

    def scan_target(self, target: TargetConfig, time_series_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[TransitHypothesis]:
        """
        Downloads, processes, and extracts potential transit signals from a target.
        If `time_series_data` is provided, it uses that directly instead of fetching.
        """
        logger.info(f"Scanning target: TIC {target.tic_id}")
        
        time_lc, flux_lc = None, None
        provenance = None

        if time_series_data is not None:
            time_lc, flux_lc = time_series_data
            # Create a simplified provenance for direct test data
            provenance = DataProvenance(
                source="Test/DirectInput",
                instrument_id=str(target.tic_id),
                cadence_s=np.nanmedian(np.diff(time_lc)) * 24 * 3600 if len(time_lc) > 1 else -1.0,
                injected=True, # Assume test data is always injected or synthetic
                injected_params={"test_data": "simulated_transit"},
            )
            logger.info(f"Using direct time series data for TIC {target.tic_id}")
        else:
            # 1. Fetch Data
            # NOTE: This fetcher is Kepler-only and uses KIC IDs.
            fits_path = download_kepler_light_curve(kic_id=str(target.tic_id))
            if not fits_path:
                logger.error(f"Failed to download light curve for KIC {target.tic_id}.")
                return []

            try:
                with fits.open(fits_path, memmap=False) as hdul:
                    primary_header = hdul[0].header
                    lc_data = hdul[1].data
                    time_lc = lc_data['TIME']
                    flux_lc = lc_data['PDCSAP_FLUX']

                    # --- Create DataProvenance Object ---
                    source = primary_header.get('TELESCOP', 'Unknown')
                    instrument_id = primary_header.get('KEPLERID') or primary_header.get('TICID', 'UnknownID')
                    cadence = np.nanmedian(np.diff(time_lc)) * 24 * 3600 if len(time_lc) > 1 else -1.0
                    provenance = DataProvenance(
                        source=source,
                        instrument_id=str(instrument_id),
                        cadence_s=float(cadence),
                        mission_quarter=primary_header.get('QUARTER'),
                        mission_sector=primary_header.get('SECTOR'),
                        injected='INJECT' in primary_header and primary_header['INJECT'],
                    )
                    # ---
            
            except Exception as e:
                logger.error(f"Failed to read or process FITS file {fits_path}: {e}")
                return []

        # Ensure time and flux are available
        if time_lc is None or flux_lc is None:
            logger.error(f"No time or flux data available for TIC {target.tic_id} after acquisition.")
            return []

        # Basic data cleaning
        mask = np.isfinite(time_lc) & np.isfinite(flux_lc)
        time_lc, flux_lc = time_lc[mask], flux_lc[mask]
        
        # New normalization: zero mean, unit variance
        if np.std(flux_lc) > 1e-8: # Avoid division by zero for constant flux
            flux_lc = (flux_lc - np.mean(flux_lc)) / np.std(flux_lc)
        else:
            flux_lc = np.zeros_like(flux_lc) # If flux is constant, set to zero
        
        logger.debug(f"Flux after normalization: min={flux_lc.min():.4f}, max={flux_lc.max():.4f}, median={np.nanmedian(flux_lc):.4f}")

        # 2. Detect Signals using existing BLS logic
        # The existing function returns a list of dictionaries, one per signal
        # We adapt this to the expected output of the prompt
        transit_info, periodicity_data = find_transits_bls(time_lc, flux_lc)
        
        if not transit_info or not transit_info.get('times'):
            logger.info(f"No significant signals found for KIC {target.tic_id}.")
            return []

        logger.info(f"Found {len(transit_info['times'])} potential signals for KIC {target.tic_id}.")

        # 3. Process each signal into a TransitHypothesis
        hypotheses = []
        # Assuming find_transits_bls returns info for one best signal
        # but let's structure it to handle multiple if the function is ever updated
        # For now, it will just loop once.
        
        # We need to create a list of signals from the bls output format
        num_signals = len(transit_info.get('times', []))
        signals = []
        # find_transits_bls returns `transit_info` which contains 'times', 'depths', 'durations'
        # and `periodicity_data` which contains 'median_period'
        # We need to combine these to form a signal dictionary.
        
        if num_signals > 0:
            # Assume for now, all detected transits correspond to the best period found
            # by BLS in periodicity_data
            bls_period = periodicity_data.get('median_period')
            if bls_period is None:
                logger.error("BLS did not provide a median_period. Cannot create signals.")
                return []

            for i in range(num_signals):
                 signals.append({
                     'period': bls_period,
                     't0': transit_info['times'][i],
                     'duration': transit_info['durations'][i],
                     'depth': transit_info['depths'][i],
                 })
        else:
            logger.error("No transit times found in transit_info, despite non-None transit_info. Cannot create signals.")
            return []


        for signal in signals:
            try:
                # Generate a unique ID for the hypothesis
                hypo_id = f"KIC{target.tic_id}_P{signal['period']:.5f}"

                # Create the data views for the model
                # This is the critical placeholder step
                data_views = _create_data_views(time_lc, flux_lc, signal)

                hypothesis = TransitHypothesis(
                    id=hypo_id,
                    provenance=provenance, # Pass the provenance object
                    signal_params={
                        'period': signal['period'],
                        't0': signal['t0'],
                        'duration': signal['duration'],
                        'depth': signal['depth'],
                    },
                    data_views=data_views,
                    status=PipelineStatus.DETECTED
                )
                hypothesis.log("Signal detected by IngestionEngine.")
                hypotheses.append(hypothesis)

            except Exception as e:
                logger.error(f"Error processing signal into hypothesis: {e}", exc_info=True)

        return hypotheses
