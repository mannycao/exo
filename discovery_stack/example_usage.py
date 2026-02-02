#!/usr/bin/env python3
"""
example_usage.py: Demonstrates the Continuous Discovery Service

This script shows:
1. Creating and using the registry
2. Creating and using the sentinel
3. Running a single target through the pipeline
4. Interpreting CACL rationales
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from discovery_stack.registry import TargetRegistry
from discovery_stack.sentinel import DataSentinel
from discovery_stack.types import TargetConfig, TransitHypothesis, PipelineStatus
import numpy as np


def example_1_registry():
    """Example 1: Using the Target Registry"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Target Registry")
    print("=" * 80 + "\n")
    
    # Create a temporary registry
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        registry_path = f.name
    
    print(f"Creating registry at: {registry_path}\n")
    
    # Initialize registry
    registry = TargetRegistry(registry_path=registry_path)
    
    # Mark some targets as processed
    print("Marking targets as processed:")
    registry.mark_processed(
        tic_id=123456789,
        sector=15,
        status="ACCEPTED",
        rationale="ML Confidence HIGH (0.92) | CACL Agreement STRONG (0.05)"
    )
    print("  ✓ TIC 123456789 Sector 15 → ACCEPTED")
    
    registry.mark_processed(
        tic_id=987654321,
        sector=15,
        status="REJECTED",
        rationale="No transit signals detected by BLS"
    )
    print("  ✓ TIC 987654321 Sector 15 → REJECTED")
    
    registry.mark_processed(
        tic_id=555555555,
        sector=20,
        status="PHYSICS_CLEARED",
        rationale="Passed physics gates but ML confidence LOW (0.62)"
    )
    print("  ✓ TIC 555555555 Sector 20 → PHYSICS_CLEARED\n")
    
    # Check if targets are processed
    print("Checking processed status:")
    for tic, sector in [(123456789, 15), (111111111, 25)]:
        is_proc = registry.is_processed(tic, sector)
        status = "Already processed ✓" if is_proc else "New target ✓"
        print(f"  TIC {tic} Sector {sector}: {status}")
    
    # Get statistics
    print("\nRegistry statistics:")
    stats = registry.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get all targets
    print("\nAll registered targets:")
    all_targets = registry.get_all_targets()
    for target in all_targets:
        print(f"  • TIC {target['tic_id']} S{target['sector']}: {target['status']}")
        print(f"    Rationale: {target['rationale'][:70]}...")
    
    # Get history for a specific TIC
    print("\nHistory for TIC 123456789:")
    history = registry.get_target_history(123456789)
    for entry in history:
        print(f"  • Sector {entry['sector']}: {entry['status']}")
    
    # Cleanup
    os.unlink(registry_path)
    print("\n✓ Example 1 complete\n")


def example_2_sentinel():
    """Example 2: Using the Data Sentinel"""
    print("=" * 80)
    print("EXAMPLE 2: Data Sentinel")
    print("=" * 80 + "\n")
    
    # Create a temporary directory with mock FITS files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating mock FITS files in: {tmpdir}\n")
        
        # Create mock FITS files
        fits_files = [
            "tess20240901-s0015-tic123456789_lc.fits",
            "tess20240915-s0016-tic987654321_lc.fits",
            "tic_555555555_s0020_lc.fits",
            "kplr111111111_lc.fits",
        ]
        
        for fname in fits_files:
            Path(tmpdir, fname).touch()
            print(f"  ✓ Created: {fname}")
        
        print("\nScanning for new data:")
        
        # Create sentinel
        registry = TargetRegistry(registry_path=os.path.join(tmpdir, "registry.json"))
        sentinel = DataSentinel(registry=registry)
        
        # Check for new data
        targets = sentinel.check_for_new_data(tmpdir)
        
        print(f"\nFound {len(targets)} new targets:\n")
        for target in targets:
            print(f"  • TIC {target.tic_id} (Mission: {target.mission}, Sector: {target.sector})")
        
        # Try again (should find none, as all are now registered)
        print("\n\nScanning again (should find 0, as all are now registered):")
        # First, mark all as processed
        for target in targets:
            registry.mark_processed(target.tic_id, int(target.sector), "ACCEPTED", "test")
        
        targets_2 = sentinel.check_for_new_data(tmpdir)
        print(f"Found {len(targets_2)} new targets")
    
    print("\n✓ Example 2 complete\n")


def example_3_cacl_rationale():
    """Example 3: Understanding CACL Rationales"""
    print("=" * 80)
    print("EXAMPLE 3: CACL Rationale Examples")
    print("=" * 80 + "\n")
    
    # Import the wrapper to access rationale generation
    from discovery_stack.wrappers import BayesianWrapper
    
    # Create a dummy wrapper instance (just for rationale generation)
    # We'll manually call the methods
    
    print("Understanding the CACL Rationale Components:\n")
    
    print("1. ML Confidence: Posterior mean probability")
    print("   • HIGH: posterior ≥ 0.8 (model thinks it's a planet)")
    print("   • LOW:  posterior < 0.8 (model less confident)\n")
    
    print("2. Epistemic Uncertainty: Model's own uncertainty")
    print("   • LOW:  epistemic ≤ 0.2 (model is confident in its uncertainty)")
    print("   • HIGH: epistemic > 0.2 (model is uncertain about uncertainty)\n")
    
    print("3. CACL Agreement: Disagreement between model views")
    print("   • STRONG: disagreement ≤ 0.15 (views agree)")
    print("   • WEAK:   disagreement > 0.15 (views disagree)\n")
    
    print("4. Predictions: Entropy of MC samples")
    print("   • SHARP:   entropy < 0.5 bits (concentrated)")
    print("   • DIFFUSE: entropy ≥ 0.5 bits (spread out)\n")
    
    print("Example Rationales:\n")
    
    scenarios = [
        {
            "name": "STRONG CANDIDATE (Acceptance likely)",
            "posterior": 0.95,
            "uncertainty": 0.05,
            "disagreement": 0.02,
            "entropy": 0.2
        },
        {
            "name": "WEAK CANDIDATE (Governance may reject)",
            "posterior": 0.72,
            "uncertainty": 0.25,
            "disagreement": 0.18,
            "entropy": 0.8
        },
        {
            "name": "MODERATE CANDIDATE (Close call)",
            "posterior": 0.82,
            "uncertainty": 0.18,
            "disagreement": 0.12,
            "entropy": 0.35
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  posterior={scenario['posterior']:.2f}, uncertainty={scenario['uncertainty']:.2f}, "
              f"disagreement={scenario['disagreement']:.2f}, entropy={scenario['entropy']:.2f}")
        
        # Build rationale components
        components = []
        
        # Confidence
        if scenario['posterior'] >= 0.8:
            components.append(f"ML Confidence HIGH ({scenario['posterior']:.3f} ≥ 0.8)")
        else:
            components.append(f"ML Confidence LOW ({scenario['posterior']:.3f} < 0.8)")
        
        # Uncertainty
        if scenario['uncertainty'] <= 0.2:
            components.append(f"Epistemic Uncertainty LOW ({scenario['uncertainty']:.3f} ≤ 0.2)")
        else:
            components.append(f"Epistemic Uncertainty HIGH ({scenario['uncertainty']:.3f} > 0.2)")
        
        # CACL
        if scenario['disagreement'] <= 0.15:
            components.append(f"CACL Agreement STRONG ({scenario['disagreement']:.3f} ≤ 0.15)")
        else:
            components.append(f"CACL Agreement WEAK ({scenario['disagreement']:.3f} > 0.15)")
        
        # Entropy
        if scenario['entropy'] < 0.5:
            components.append(f"Predictions SHARP ({scenario['entropy']:.2f} bits)")
        else:
            components.append(f"Predictions DIFFUSE ({scenario['entropy']:.2f} bits)")
        
        rationale = " | ".join(components)
        print(f"  Rationale: {rationale}")
    
    print("\n✓ Example 3 complete\n")


def example_4_pipeline_flow():
    """Example 4: Understanding the Pipeline Flow"""
    print("=" * 80)
    print("EXAMPLE 4: Pipeline Decision Flow")
    print("=" * 80 + "\n")
    
    print("Pipeline Step-by-Step:\n")
    
    print("1. SENTINEL DETECTS NEW FITS FILE")
    print("   ├─ Filename: tess20240901-s0015-tic123456789_lc.fits")
    print("   ├─ Parsed: TIC 123456789, Sector 15")
    print("   └─ Registry check: Not processed → CONTINUE\n")
    
    print("2. INGESTION ENGINE (BLS Signal Detection)")
    print("   ├─ Downloads light curve")
    print("   ├─ Runs BLS algorithm")
    print("   ├─ Finds: 1 transit signal (period=3.5 days)")
    print("   └─ Result: TransitHypothesis created\n")
    
    print("3. BAYESIAN ML MODEL (Inference)")
    print("   ├─ Input: Light curve features")
    print("   ├─ MC Sampling: 100 samples from posterior")
    print("   ├─ Outputs:")
    print("   │  • posterior_mean = 0.92")
    print("   │  • epistemic_uncertainty = 0.08")
    print("   │  • predictive_entropy = 0.28")
    print("   │  • cacl_disagreement = 0.05")
    print("   └─ Generates rationale: \"ML Confidence HIGH | Epistemic Uncertainty LOW...\" \n")
    
    print("4. GOVERNANCE GATES (Physics Checks)")
    print("   ├─ Transit Geometry Oracle")
    print("   │  └─ Q: Is this an eclipsing binary? A: No → PASS")
    print("   ├─ Orbital Stability Oracle")
    print("   │  └─ Q: Can this orbit exist? A: Yes → PASS")
    print("   ├─ Uncertainty Gate")
    print("   │  └─ Q: Is uncertainty ≤ 0.2? A: Yes (0.08) → PASS")
    print("   ├─ Confidence Gate")
    print("   │  └─ Q: Is posterior ≥ 0.8? A: Yes (0.92) → PASS")
    print("   └─ Result: Status = ML_CANDIDATE ✓\n")
    
    print("5. REGISTRY & DOSSIER")
    print("   ├─ mark_processed(123456789, 15, 'ML_CANDIDATE', rationale='...')")
    print("   ├─ Generate dossier_TIC123456789_S15_20250202_143522.txt")
    print("   │  └─ Contains: target metadata, signal params, ML results, CACL rationale, decision")
    print("   └─ Archive FITS to data_files/processed/\n")
    
    print("FINAL OUTPUT:")
    print("   ✓ Registry entry created")
    print("   ✓ Dossier generated (human-readable scientific report)")
    print("   ✓ FITS file archived")
    print("   ✓ Logs written to discovery_service.log\n")
    
    print("✓ Example 4 complete\n")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Continuous Discovery Service - Usage Examples".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        example_1_registry()
        example_2_sentinel()
        example_3_cacl_rationale()
        example_4_pipeline_flow()
        
        print("=" * 80)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Review CONTINUOUS_SERVICE_GUIDE.md for detailed operations")
        print("  2. Run: python discovery_stack/quick_start.py --help")
        print("  3. Start service: python discovery_stack/quick_start.py\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
