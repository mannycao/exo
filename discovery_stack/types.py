from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import numpy as np

# Import the new Provenance object
from data.provenance import DataProvenance


class PipelineStatus(Enum):
    """Represents the state of a transit hypothesis within the pipeline."""
    DETECTED = "DETECTED"
    ML_CANDIDATE = "ML_CANDIDATE"
    PHYSICS_CLEARED = "PHYSICS_CLEARED"
    REJECTED = "REJECTED"

@dataclass
class TargetConfig:
    """Configuration for a target to be processed, identified by mission-specific ID."""
    target_id: str
    sector: int
    source_file: str
    mission: str = "Unknown" # Will be determined from FITS header if not provided

@dataclass
class TransitHypothesis:
    """
    Core data artifact representing a potential transit signal.
    This object is created by the IngestionEngine and enriched as it
    passes through the discovery pipeline.
    """
    id: str  # Unique ID, e.g., "KIC1234567_P12.5"
    mission: str # Mission name, e.g., "TESS", "Kepler"
    provenance: DataProvenance
    signal_params: Dict[str, float]
    
    data_views: Dict[str, np.ndarray] = field(repr=False)
    
    # Renamed fields to be more scientifically precise
    inference: Dict[str, any] = field(default_factory=dict)
    
    status: PipelineStatus = PipelineStatus.DETECTED
    logs: List[str] = field(default_factory=list)

    def log(self, message: str):
        """Adds a message to the hypothesis's event log."""
        self.logs.append(message)
