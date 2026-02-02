#!/usr/bin/env python3
"""
quick_start.py: Boot the Continuous Discovery Service

Run this script to start monitoring for new TESS/Kepler data and
automatically processing targets through the full exoplanet detection pipeline.

Usage:
    python quick_start.py                    # Default: watch data_files/incoming
    python quick_start.py --watch /path/to/folder
    python quick_start.py --interval 120     # Poll every 2 minutes
    python quick_start.py --test --max-iter 5  # Test mode: 5 iterations max
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from discovery_stack.mission_control import MissionControl
from discovery_stack.registry import TargetRegistry
from discovery_stack.sentinel import DataSentinel
from discovery_stack.ingest import IngestionEngine
from discovery_stack.wrappers import BayesianWrapper
from discovery_stack.governance import DiscoveryGovernor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Continuous Discovery Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py
  python quick_start.py --watch /mnt/data_files/incoming --interval 60
  python quick_start.py --test --max-iter 5
  python quick_start.py --model /models/bayes_v2.h5
        """
    )
    
    parser.add_argument(
        "--watch",
        type=str,
        default="data_files/incoming",
        help="Directory to monitor for new FITS files (default: data_files/incoming)"
    )
    
    parser.add_argument(
        "--processed",
        type=str,
        default="data_files/processed",
        help="Directory to archive processed FITS files (default: data_files/processed)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to Bayesian model file (optional; service will skip ML if not provided)"
    )
    
    parser.add_argument(
        "--registry",
        type=str,
        default="processed_targets.json",
        help="Path to registry file (default: processed_targets.json)"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        default="discovery_results",
        help="Directory for dossiers and reports (default: discovery_results)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (reduces verbosity, useful for initial testing)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum iterations (None = infinite; useful for testing)"
    )
    
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.2,
        help="Reject if epistemic uncertainty > threshold (default: 0.2)"
    )
    
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.8,
        help="Accept if posterior probability > threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Validate directories exist
    if not os.path.isdir(args.watch):
        logger.error(f"Watch folder does not exist: {args.watch}")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs(args.processed, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CONTINUOUS DISCOVERY SERVICE — QUICK START")
    print("=" * 80 + "\n")
    
    print("Configuration:")
    print(f"  Watch folder:          {args.watch}")
    print(f"  Processed folder:      {args.processed}")
    print(f"  Registry:              {args.registry}")
    print(f"  Results directory:     {args.results}")
    print(f"  Poll interval:         {args.interval}s")
    print(f"  Bayesian model:        {args.model or '(DISABLED — ML will be skipped)'}")
    print(f"  Uncertainty threshold: {args.uncertainty_threshold}")
    print(f"  Probability threshold: {args.prob_threshold}")
    print(f"  Max iterations:        {args.max_iter or 'INFINITE'}")
    print(f"  Log level:             {args.log_level}")
    print(f"  Test mode:             {'YES' if args.test else 'NO'}")
    
    print("\n" + "-" * 80)
    print("Initializing components...\n")
    
    try:
        # Initialize registry
        registry = TargetRegistry(registry_path=args.registry)
        
        # Initialize sentinel
        sentinel = DataSentinel(registry=registry)
        
        # Initialize ingestion
        ingestion_engine = IngestionEngine()
        
        # Initialize Bayesian wrapper (optional)
        bayesian_wrapper = None
        if args.model:
            try:
                bayesian_wrapper = BayesianWrapper(model_path=args.model)
                print(f"✓ Bayesian model loaded: {args.model}")
            except Exception as e:
                logger.warning(f"Failed to load Bayesian model: {e}. Continuing without ML.")
        else:
            print("ℹ Bayesian model not specified. ML will be skipped.")
        
        # Initialize governance
        governor = DiscoveryGovernor(
            uncertainty_threshold=args.uncertainty_threshold,
            prob_threshold=args.prob_threshold
        )
        
        # Initialize mission control
        cms = MissionControl(
            registry=registry,
            sentinel=sentinel,
            ingestion_engine=ingestion_engine,
            bayesian_wrapper=bayesian_wrapper,
            governor=governor,
            results_dir=args.results
        )
        
        print("✓ Registry initialized")
        print("✓ Sentinel initialized")
        print("✓ Ingestion engine initialized")
        print("✓ Governance initialized")
        print("✓ Mission control ready\n")
        
        # Print initial stats
        stats = cms.get_registry_stats()
        print("Registry Status:")
        print(f"  Total processed:     {stats.get('total', 0)}")
        print(f"  Accepted:            {stats.get('accepted', 0)}")
        print(f"  Rejected:            {stats.get('rejected', 0)}")
        
        print("\n" + "-" * 80)
        print("Starting daemon...\n")
        print("Press Ctrl+C to stop gracefully.\n")
        
        # Run daemon
        cms.run_daemon(
            watch_folder=args.watch,
            interval_seconds=args.interval,
            processed_folder=args.processed,
            max_iterations=args.max_iter
        )
    
    except KeyboardInterrupt:
        print("\n\nShutdown signal received. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
