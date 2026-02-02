"""
MissionControl: Orchestrates the continuous monitoring service.

This module runs an infinite daemon loop that:
1. Monitors for new TESS/Kepler data (via DataSentinel).
2. Runs targets through the full discovery pipeline.
3. Generates human-readable decision rationales.
4. Maintains persistent state (via TargetRegistry).
5. Logs all decisions for audit and scientific review.
"""

import os
import sys
import logging
import time
import shutil
import traceback
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

# Local imports
from .registry import TargetRegistry
from .sentinel import DataSentinel
from .ingest import IngestionEngine
from .wrappers import BayesianWrapper
from .governance import DiscoveryGovernor
from .types import TargetConfig, TransitHypothesis, PipelineStatus

# Try to import config
try:
    import config as lite_config
except ImportError:
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import config as lite_config
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Could not import config module. Using defaults.")
        lite_config = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("discovery_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MissionControl:
    """
    Central orchestrator for the continuous discovery service.
    
    Manages the full pipeline:
      Data Sentinel → Ingestion → Bayesian ML → Governance → Registry
    """
    
    def __init__(
        self,
        registry: Optional[TargetRegistry] = None,
        sentinel: Optional[DataSentinel] = None,
        ingestion_engine: Optional[IngestionEngine] = None,
        bayesian_wrapper: Optional[BayesianWrapper] = None,
        governor: Optional[DiscoveryGovernor] = None,
        results_dir: str = "discovery_results"
    ):
        """
        Initialize MissionControl with components.
        
        Args:
            registry: TargetRegistry instance (created if None).
            sentinel: DataSentinel instance (created if None).
            ingestion_engine: IngestionEngine instance (created if None).
            bayesian_wrapper: BayesianWrapper instance (created if None).
            governor: DiscoveryGovernor instance (created if None).
            results_dir: Directory for dossiers and logs.
        """
        self.registry = registry or TargetRegistry()
        self.sentinel = sentinel or DataSentinel(registry=self.registry)
        self.ingestion_engine = ingestion_engine or IngestionEngine()
        self.bayesian_wrapper = bayesian_wrapper
        self.governor = governor or DiscoveryGovernor()
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"MissionControl initialized. Results directory: {self.results_dir}")
    
    def run_daemon(
        self,
        watch_folder: str,
        interval_seconds: int = 60,
        processed_folder: str = "data_files/processed",
        max_iterations: Optional[int] = None  # For testing; None = infinite
    ):
        """
        Run the continuous monitoring daemon.
        
        This is the main service loop. It runs indefinitely, polling for
        new data every `interval_seconds`, processing targets through the
        full pipeline, and logging decisions.
        
        Args:
            watch_folder: Directory to monitor for new FITS files.
            interval_seconds: Seconds to sleep between polls.
            processed_folder: Directory to move processed files to.
            max_iterations: Max loop iterations (None = infinite; for testing).
        """
        logger.info("=" * 80)
        logger.info("DISCOVERY SERVICE STARTING")
        logger.info("=" * 80)
        logger.info(f"Watch folder: {watch_folder}")
        logger.info(f"Poll interval: {interval_seconds}s")
        logger.info(f"Max iterations: {max_iterations or 'INFINITE'}")
        
        os.makedirs(processed_folder, exist_ok=True)
        
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            try:
                logger.info(f"\n[ITERATION {iteration}] Checking for new data...")
                
                # Poll for new targets
                new_targets = self.sentinel.check_for_new_data(watch_folder)
                
                if not new_targets:
                    logger.info(f"No new data. Sleeping for {interval_seconds}s...")
                    time.sleep(interval_seconds)
                    continue
                
                logger.info(f"Found {len(new_targets)} new target(s).")
                
                # Process each target
                for target in new_targets:
                    try:
                        self._process_target(target, watch_folder, processed_folder)
                    except Exception as e:
                        logger.error(
                            f"Error processing target TIC {target.tic_id}: {e}",
                            exc_info=True
                        )
                        # Still mark as processed to prevent retry loop
                        self.registry.mark_processed(
                            target.tic_id,
                            int(target.sector),
                            "ERROR",
                            rationale=f"Pipeline error: {str(e)[:100]}"
                        )
                
                logger.info(f"Sleeping for {interval_seconds}s...")
                time.sleep(interval_seconds)
            
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in daemon loop: {e}", exc_info=True)
                time.sleep(interval_seconds)
        
        logger.info("=" * 80)
        logger.info("DISCOVERY SERVICE STOPPED")
        logger.info("=" * 80)
    
    def _process_target(
        self,
        target: TargetConfig,
        watch_folder: str,
        processed_folder: str
    ):
        """
        Process a single target through the full discovery pipeline.
        
        Flow:
          1. Ingest: Extract transit signals from light curve.
          2. ML: Run Bayesian inference (if BayesianWrapper available).
          3. Governance: Apply physics and data quality gates.
          4. Registry: Mark as processed with rationale.
          5. Archive: Move FITS file to processed folder.
        
        Args:
            target: TargetConfig to process.
            watch_folder: Directory where original FITS file is located.
            processed_folder: Directory to move processed file to.
        """
        logger.info(f"\n>> PROCESSING: TIC {target.tic_id} Sector {target.sector}")
        
        # Step 1: Ingest (extract transit candidates)
        logger.debug(f"   [INGEST] Scanning target for transits...")
        try:
            hypotheses = self.ingestion_engine.scan_target(target)
        except Exception as e:
            logger.error(f"   [INGEST] FAILED: {e}")
            self.registry.mark_processed(
                target.tic_id,
                int(target.sector),
                "REJECTED",
                rationale=f"Ingestion failed: {str(e)[:80]}"
            )
            return
        
        if not hypotheses:
            logger.info(f"   [INGEST] No transit signals detected.")
            self.registry.mark_processed(
                target.tic_id,
                int(target.sector),
                "REJECTED",
                rationale="No transit signals detected by BLS."
            )
            self._archive_fits_file(target, watch_folder, processed_folder)
            return
        
        logger.info(f"   [INGEST] Found {len(hypotheses)} transit candidate(s).")
        
        # Step 2 & 3: ML + Governance for each hypothesis
        best_hypothesis = None
        best_status = PipelineStatus.REJECTED
        best_rationale = "No candidates passed governance."
        
        for idx, hyp in enumerate(hypotheses):
            logger.debug(f"   [HYPOTHESIS {idx+1}/{len(hypotheses)}] {hyp.id}")
            
            # ML step
            if self.bayesian_wrapper:
                logger.debug(f"      [ML] Running Bayesian inference...")
                try:
                    hyp = self.bayesian_wrapper.predict(hyp)
                except Exception as e:
                    logger.warning(f"      [ML] FAILED: {e}")
                    hyp.log(f"Bayesian prediction failed: {e}")
                    hyp.status = PipelineStatus.REJECTED
            else:
                logger.warning(f"      [ML] BayesianWrapper not configured. Skipping ML.")
                hyp.log("No ML model available. Skipping inference.")
            
            # Governance step
            logger.debug(f"      [AUDIT] Running governance checks...")
            try:
                hyp = self.governor.audit(hyp)
            except Exception as e:
                logger.warning(f"      [AUDIT] FAILED: {e}")
                hyp.log(f"Governance audit failed: {e}")
                hyp.status = PipelineStatus.REJECTED
            
            # Extract rationale from hypothesis logs
            rationale = " | ".join(hyp.logs[-5:]) if hyp.logs else "No logs"
            
            # Track best result
            if hyp.status in [PipelineStatus.ML_CANDIDATE, PipelineStatus.PHYSICS_CLEARED]:
                if best_hypothesis is None:
                    best_hypothesis = hyp
                    best_status = hyp.status
                    best_rationale = rationale
                elif hyp.inference.get('posterior_mean', 0) > best_hypothesis.inference.get('posterior_mean', 0):
                    best_hypothesis = hyp
                    best_status = hyp.status
                    best_rationale = rationale
            
            logger.info(f"      [RESULT] Status: {hyp.status.value}, Rationale: {rationale[:100]}...")
        
        # Step 4: Register the result
        status_str = best_status.value if best_hypothesis else "REJECTED"
        logger.info(f">> FINAL DECISION: {status_str}")
        logger.info(f">> RATIONALE: {best_rationale[:150]}...")
        
        self.registry.mark_processed(
            target.tic_id,
            int(target.sector),
            status_str,
            rationale=best_rationale
        )
        
        # Step 5: Generate dossier (detailed report)
        if best_hypothesis:
            self._generate_dossier(target, best_hypothesis, best_status, best_rationale)
        
        # Step 6: Archive the FITS file
        self._archive_fits_file(target, watch_folder, processed_folder)
    
    def _generate_dossier(
        self,
        target: TargetConfig,
        hypothesis: TransitHypothesis,
        status: PipelineStatus,
        rationale: str
    ):
        """
        Generate a detailed "Dossier" report for a candidate.
        
        The dossier is a human-readable document that summarizes:
        - Target metadata
        - Transit signal characteristics
        - ML inference results with CACL rationale
        - Governance decision and reasoning
        - Provenance and data quality information
        
        Args:
            target: The TargetConfig.
            hypothesis: The TransitHypothesis with inference results.
            status: The final PipelineStatus.
            rationale: The decision rationale string.
        """
        dossier_filename = f"dossier_TIC{target.tic_id}_S{target.sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        dossier_path = os.path.join(self.results_dir, dossier_filename)
        
        try:
            with open(dossier_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("EXOPLANET DISCOVERY DOSSIER\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Target: TIC {target.tic_id} (Mission: {target.mission})\n")
                f.write(f"Sector: {target.sector}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("TRANSIT SIGNAL\n")
                f.write("-" * 80 + "\n")
                f.write(f"Hypothesis ID: {hypothesis.id}\n")
                f.write(f"Signal Parameters: {hypothesis.signal_params}\n")
                f.write(f"Provenance: {hypothesis.provenance}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("ML INFERENCE RESULTS\n")
                f.write("-" * 80 + "\n")
                for key, value in hypothesis.inference.items():
                    if isinstance(value, (list, np.ndarray)):
                        f.write(f"{key}: [array of {len(value)} elements]\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                # CACL-specific rationale
                if 'rationale' in hypothesis.inference:
                    f.write(f"\nCACL Rationale: {hypothesis.inference['rationale']}\n")
                
                f.write(f"\n")
                
                f.write("-" * 80 + "\n")
                f.write("GOVERNANCE DECISION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Status: {status.value}\n")
                f.write(f"Decision Rationale:\n{rationale}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("PROCESSING LOG\n")
                f.write("-" * 80 + "\n")
                for log_entry in hypothesis.logs:
                    f.write(f"  • {log_entry}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF DOSSIER\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Dossier written: {dossier_path}")
        
        except Exception as e:
            logger.error(f"Failed to generate dossier: {e}", exc_info=True)
    
    def _archive_fits_file(
        self,
        target: TargetConfig,
        watch_folder: str,
        processed_folder: str
    ):
        """
        Move a processed FITS file from watch folder to processed folder.
        
        Args:
            target: The TargetConfig (used to identify the file).
            watch_folder: Original location.
            processed_folder: Destination.
        """
        try:
            # Find FITS files matching this target
            watch_path = Path(watch_folder)
            pattern = f"*tic{target.tic_id}*s{int(target.sector):04d}*.fits"
            
            matches = list(watch_path.glob(pattern))
            if not matches:
                # Try alternate pattern
                pattern = f"*{target.tic_id}*{target.sector}*.fits"
                matches = list(watch_path.glob(pattern))
            
            for fits_file in matches:
                try:
                    dest = os.path.join(processed_folder, fits_file.name)
                    shutil.move(str(fits_file), dest)
                    logger.info(f"Archived: {fits_file.name} → {processed_folder}/")
                except Exception as e:
                    logger.warning(f"Could not archive {fits_file.name}: {e}")
        
        except Exception as e:
            logger.warning(f"Error archiving FITS files for TIC {target.tic_id}: {e}")
    
    def get_registry_stats(self) -> Dict:
        """Retrieve current registry statistics."""
        return self.registry.get_stats()
    
    def print_status(self):
        """Print human-readable service status."""
        stats = self.get_registry_stats()
        logger.info("\n" + "=" * 80)
        logger.info("DISCOVERY SERVICE STATUS")
        logger.info("=" * 80)
        for key, value in stats.items():
            logger.info(f"{key.upper()}: {value}")
        logger.info("=" * 80)


# Add numpy import at module level for dossier generation
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available for dossier formatting.")
