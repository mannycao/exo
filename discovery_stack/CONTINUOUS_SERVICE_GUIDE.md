# Continuous Monitoring Service (CMS) — Operations Guide

**Status:** Production-Ready SRE Implementation  
**Version:** 1.0  
**Last Updated:** February 2, 2026

---

## Executive Summary

The **Discovery Stack** has been upgraded from a batch script to a **Continuous Monitoring Service (CMS)**. This service runs indefinitely, watching for new TESS/Kepler data and automatically runs the complete exoplanet discovery pipeline:

```
Data Sentinel → Ingest (BLS) → ML (Bayesian) → Governance (Physics) → Registry
                                                                     ↓
                                                            Dossier + Archive
```

The system produces detailed **Dossiers** for each candidate with scientific rationales grounded in CACL (Contextual Abstract Co-Learning) metrics and Bayesian statistics.

---

## Architecture Overview

### Four Core Components

#### 1. **`registry.py` — The Memory** 
Persistent state management for processed targets.

**Key Classes:**
- `TargetRegistry`: Maintains `processed_targets.json`

**Key Methods:**
- `is_processed(tic_id, sector) → bool`: Check if target already handled
- `mark_processed(tic_id, sector, status, rationale, timestamp)`: Record result
- `get_stats() → dict`: Summary counts (accepted, rejected, total)
- `get_all_targets() → list`: Full history

**Thread-safe:** Uses locks for concurrent access.

---

#### 2. **`sentinel.py` — The Watchdog**
Monitors the filesystem for new data and yields targets for processing.

**Key Classes:**
- `DataSentinel`: Scans directories, parses FITS filenames

**Key Methods:**
- `check_for_new_data(watch_folder) → List[TargetConfig]`: 
  - Finds `.fits` files in `watch_folder`
  - Parses TIC ID and Sector from filenames (regex-based)
  - Consults `TargetRegistry` to skip already-processed targets
  - Returns list of new `TargetConfig` objects

- `poll_mast(sector, mission) → List[TargetConfig]`:
  - **Placeholder** for future NASA MAST API integration
  - Currently returns empty list

**Filename Patterns:**
- TESS: `tess2024...-s0015-tic123456789_lc.fits`
- Kepler: `kplr123456789_lc.fits`

---

#### 3. **`mission_control.py` — The Orchestrator**
Central service daemon that orchestrates the full pipeline.

**Key Classes:**
- `MissionControl`: Coordinates all components

**Key Methods:**
- `run_daemon(watch_folder, interval_seconds=60, processed_folder, max_iterations=None)`:
  - **Infinite Loop:**
    1. Poll for new targets
    2. For each target:
       - `IngestionEngine.scan_target()` → List[TransitHypothesis]
       - `BayesianWrapper.predict()` → Add ML scores and CACL rationale
       - `DiscoveryGovernor.audit()` → Apply physics gates
       - Register result with rationale
       - Generate Dossier
       - Archive FITS file

- `_process_target()`: Single target pipeline
- `_generate_dossier()`: Creates detailed scientific report
- `_archive_fits_file()`: Moves FITS from incoming → processed

---

#### 4. **`wrappers.py` (Updated)** — The ML Brain with CACL
Bayesian inference with integrated CACL rationale generation.

**Key Classes:**
- `BayesianWrapper`: Wraps Keras/TensorFlow Bayesian model

**New Methods:**
- `_calculate_cacl_disagreement(prediction_dist) → float`:
  - Calculates variance of MC samples as proxy for model disagreement
  - Returns score ∈ [0, 1] where:
    - 0 = perfect agreement between model views
    - 1 = maximum disagreement

- `_generate_rationale(...) → str`:
  - Combines:
    - ML Confidence (posterior_mean ≥ 0.8 → HIGH)
    - Epistemic Uncertainty (≤ 0.2 → LOW)
    - CACL Agreement (disagreement ≤ 0.15 → STRONG)
    - Predictive Entropy (< 0.5 bits → SHARP predictions)
  - Returns human-readable string like:
    ```
    ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW (0.08 ≤ 0.2) | 
    CACL Agreement STRONG (0.05 ≤ 0.15) | Predictions SHARP (0.3 bits)
    ```

**Inference Fields:**
```python
hypothesis.inference = {
    'posterior_mean': 0.92,
    'epistemic_uncertainty': 0.08,
    'predictive_entropy': 0.3,
    'cacl_disagreement_score': 0.05,
    'posterior_distribution': [0.88, 0.91, 0.94, ...],  # MC samples
    'rationale': "ML Confidence HIGH | Epistemic Uncertainty LOW | CACL Agreement STRONG | Predictions SHARP"
}
```

---

## Pipeline Decision Flow

```
INPUT: New FITS file (watch_folder)
  ↓
SENTINEL: Parse filename → TIC ID, Sector
  ↓
REGISTRY: Check is_processed(tic_id, sector)
  ├─ YES → Skip (log "Already processed")
  └─ NO  → Continue
  ↓
INGEST: BLS signal detection
  ├─ No signals found
  │  └─ mark_processed(status="REJECTED", reason="No transits")
  │     Archive & return
  └─ Signals found → Continue
  ↓
ML: Bayesian inference + CACL rationale
  ├─ Failure → mark_processed(status="REJECTED", reason="ML error")
  └─ Success → Continue
  ↓
GOVERNANCE: Physics + Data Quality Gates
  ├─ Transit Geometry Check
  │  └─ "Is this an eclipsing binary?" → VETO if yes
  ├─ Orbital Stability Check
  │  └─ "Can orbit exist?" → VETO if no
  ├─ Uncertainty Gate (epistemic ≤ 0.2?)
  │  └─ Reject if high uncertainty
  └─ Confidence Gate (posterior ≥ 0.8?)
     ├─ YES → Status = ML_CANDIDATE
     └─ NO  → Status = PHYSICS_CLEARED (passed physics, low ML conf)
  ↓
REGISTRY: mark_processed(status, rationale)
  ↓
DOSSIER: Generate scientific report
  ↓
ARCHIVE: Move FITS → processed_folder
  ↓
OUTPUT: discovery_results/dossier_TIC*.txt
```

---

## Running the Service

### Quick Start

```python
from discovery_stack.mission_control import MissionControl
from discovery_stack.registry import TargetRegistry
from discovery_stack.sentinel import DataSentinel
from discovery_stack.ingest import IngestionEngine
from discovery_stack.wrappers import BayesianWrapper
from discovery_stack.governance import DiscoveryGovernor

# Initialize components
registry = TargetRegistry(registry_path="processed_targets.json")
sentinel = DataSentinel(registry=registry)
ingest = IngestionEngine()
bayes = BayesianWrapper(model_path="path/to/model.h5")
governor = DiscoveryGovernor(uncertainty_threshold=0.2, prob_threshold=0.8)

# Create service
cms = MissionControl(
    registry=registry,
    sentinel=sentinel,
    ingestion_engine=ingest,
    bayesian_wrapper=bayes,
    governor=governor,
    results_dir="discovery_results"
)

# Run daemon indefinitely
cms.run_daemon(
    watch_folder="data_files/incoming",
    interval_seconds=60,
    processed_folder="data_files/processed"
)
```

### With Testing (Limited Iterations)

```python
cms.run_daemon(
    watch_folder="data_files/incoming",
    interval_seconds=2,
    processed_folder="data_files/processed",
    max_iterations=10  # Stop after 10 iterations for testing
)
```

### Background Execution (systemd)

Create `/etc/systemd/system/discovery-service.service`:

```ini
[Unit]
Description=Continuous Exoplanet Discovery Service
After=network.target

[Service]
Type=simple
User=science-user
WorkingDirectory=/Users/emmanuel/proj/phd
ExecStart=/usr/bin/python3 -c "from discovery_stack.mission_control import MissionControl; cms = MissionControl(); cms.run_daemon('data_files/incoming')"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl start discovery-service
sudo systemctl status discovery-service
sudo journalctl -u discovery-service -f  # Follow logs
```

---

## Output Artifacts

### 1. **Dossier (`discovery_results/dossier_TIC*.txt`)**

Scientific report for each hypothesis. Example:

```
================================================================================
EXOPLANET DISCOVERY DOSSIER
================================================================================

Generated: 2025-02-02T14:35:22.123456
Target: TIC 123456789 (Mission: TESS)
Sector: 15

--------------------------------------------------------------------------------
TRANSIT SIGNAL
--------------------------------------------------------------------------------
Hypothesis ID: KIC123456789_P3.5
Signal Parameters: {'period': 3.521, 't0': 2520.05, 'duration': 0.18}
Provenance: DataProvenance(...)

--------------------------------------------------------------------------------
ML INFERENCE RESULTS
--------------------------------------------------------------------------------
posterior_mean: 0.92
epistemic_uncertainty: 0.08
predictive_entropy: 0.28
cacl_disagreement_score: 0.05
posterior_distribution: [0.88, 0.91, 0.94, ...]

CACL Rationale: ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW 
(0.08 ≤ 0.2) | CACL Agreement STRONG (0.05 ≤ 0.15) | Predictions SHARP (0.28 bits)

--------------------------------------------------------------------------------
GOVERNANCE DECISION
--------------------------------------------------------------------------------
Status: ML_CANDIDATE
Decision Rationale:
  • PASS: High probability score (Posterior Mean=0.92 > 0.8)
  • PASS: Geometry check cleared (exoplanet-like transit shape)
  • PASS: Orbital stability confirmed
  • PASS: Low Epistemic Uncertainty (0.08 ≤ 0.2)

--------------------------------------------------------------------------------
PROCESSING LOG
--------------------------------------------------------------------------------
  • Inference complete: posterior_mean=0.9200, uncertainty=0.0800, CACL disagreement=0.0500
  • CACL Rationale: ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW (0.08 ≤ 0.2) ...
  • PASS: High probability score (Posterior Mean=0.92 > 0.8)

================================================================================
END OF DOSSIER
================================================================================
```

### 2. **Registry (`processed_targets.json`)**

Persistent state of all processed targets:

```json
{
  "targets": [
    {
      "tic_id": 123456789,
      "sector": 15,
      "status": "ACCEPTED",
      "timestamp": "2025-02-02T14:35:22",
      "rationale": "ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW (0.08 ≤ 0.2) ..."
    }
  ],
  "stats": {
    "accepted": 5,
    "rejected": 12,
    "processing": 0,
    "total": 17
  }
}
```

### 3. **Logs (`discovery_service.log`)**

```
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: ================================================================================
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: DISCOVERY SERVICE STARTING
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: ================================================================================
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: Watch folder: data_files/incoming
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: Poll interval: 60s
2025-02-02 14:32:10 [INFO] discovery_stack.mission_control: Max iterations: INFINITE

[ITERATION 1] Checking for new data...
2025-02-02 14:32:11 [INFO] discovery_stack.sentinel: Found 2 FITS files in data_files/incoming
2025-02-02 14:32:11 [INFO] discovery_stack.sentinel: Discovered new target: TIC 123456789 Sector 15
2025-02-02 14:32:11 [INFO] discovery_stack.mission_control: Found 1 new target(s).

>> PROCESSING: TIC 123456789 Sector 15
2025-02-02 14:32:12 [INFO] discovery_stack.mission_control:    [INGEST] Found 1 transit candidate(s).
2025-02-02 14:32:13 [INFO] discovery_stack.mission_control:    [ML] Bayesian inference complete
2025-02-02 14:32:14 [INFO] discovery_stack.mission_control:    [AUDIT] Governance passed
>> FINAL DECISION: ML_CANDIDATE
>> RATIONALE: ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW (0.08 ≤ 0.2) ...
```

---

## Configuration & Tuning

### Registry Path
```python
registry = TargetRegistry(registry_path="my_registry.json")
```

### Poll Interval
**Recommendation:** 60-300 seconds for typical data centers.
```python
cms.run_daemon(watch_folder="...", interval_seconds=120)
```

### Governance Thresholds
```python
governor = DiscoveryGovernor(
    uncertainty_threshold=0.2,   # Reject if epistemic unc. > 0.2
    prob_threshold=0.8           # Accept if posterior_mean > 0.8
)
```

### CACL Rationale Thresholds
Edit in `wrappers.py`:
```python
def _generate_rationale(self, ..., 
    conf_threshold=0.8,       # High confidence if posterior ≥ 0.8
    uncert_threshold=0.2,     # Low uncertainty if ≤ 0.2
    disagree_threshold=0.15   # Strong agreement if disagreement ≤ 0.15
):
```

---

## Error Handling & Recovery

### Corrupted Registry
The system auto-recovers by reinitializing `processed_targets.json`:
```python
# In registry.py
except json.JSONDecodeError:
    logger.warning(f"Registry file corrupted. Reinitializing.")
    self._initialize_registry()
```

### Missing BayesianWrapper
The system skips ML if model is unavailable:
```python
if self.bayesian_wrapper:
    hyp = self.bayesian_wrapper.predict(hyp)
else:
    logger.warning("BayesianWrapper not configured. Skipping ML.")
```

### Pipeline Errors
- Caught at target level; service continues
- Target marked as "ERROR" with error message
- Full traceback in logs

---

## Monitoring & Metrics

### Get Service Status
```python
stats = cms.get_registry_stats()
print(f"Total processed: {stats['total']}")
print(f"Accepted: {stats['accepted']}")
print(f"Rejected: {stats['rejected']}")
```

### Log Tailing
```bash
tail -f discovery_service.log
```

### Registry History
```python
registry = TargetRegistry()
tic_history = registry.get_target_history(tic_id=123456789)
for entry in tic_history:
    print(f"TIC {entry['tic_id']} S{entry['sector']}: {entry['status']} - {entry['rationale']}")
```

---

## SRE Runbook: Common Operations

### Stop Service (Graceful)
```bash
# If running interactively:
Ctrl+C  # Triggers KeyboardInterrupt

# If running via systemd:
sudo systemctl stop discovery-service
```

### Reset Registry (Clear History)
```bash
rm processed_targets.json
# Service will auto-create on next run
```

### Replay Unprocessed Data
```bash
# Move files from processed/ back to incoming/:
mv data_files/processed/*.fits data_files/incoming/

# Service will re-process on next poll
```

### Debug Single Target
```python
from discovery_stack.mission_control import MissionControl
from discovery_stack.types import TargetConfig

cms = MissionControl()
target = TargetConfig(tic_id=123456789, mission="TESS", sector="15")
cms._process_target(target, "data_files/incoming", "data_files/processed")
```

---

## Science Notes

### CACL Disagreement Score
- **Interpretation:** How much do different model "views" agree?
- **Calculation:** Variance of MC samples, normalized to [0, 1]
- **In Publication:** "We computed CACL disagreement scores to quantify view consistency..."

### Rationale String Example

**Accepted:**
```
ML Confidence HIGH (0.92 ≥ 0.8) | Epistemic Uncertainty LOW (0.08 ≤ 0.2) | 
CACL Agreement STRONG (0.05 ≤ 0.15) | Predictions SHARP (0.28 bits)
```

**Rejected:**
```
ML Confidence LOW (0.62 < 0.8) | Epistemic Uncertainty HIGH (0.35 > 0.2) | 
CACL Agreement WEAK (0.22 > 0.15) | Predictions DIFFUSE (1.8 bits)
```

---

## References

- **Types:** `discovery_stack/types.py` (TargetConfig, TransitHypothesis, PipelineStatus)
- **Ingestion:** `discovery_stack/ingest.py` (IngestionEngine, BLS detection)
- **Physics:** `validation/physics_oracles/` (Transit geometry, orbital stability)
- **Config:** `config.py` (Model paths, thresholds)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Service doesn't start | Check model path in BayesianWrapper, verify watch_folder exists |
| No data detected | Check FITS filename patterns match regex in `sentinel.py` |
| Registry corrupts | Automatic recovery, but check disk space |
| Memory leak over time | Monitor via `htop`; may need periodic service restart |
| Targets re-processed | Check registry.json not being deleted; verify is_processed() logic |

---

**Deployment Date:** February 2, 2026  
**Status:** Production Ready  
**Contact:** SRE Team (Data Pipeline)
