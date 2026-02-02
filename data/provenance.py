from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataProvenance:
    """
    Metadata tracking the origin and processing history of a data sample.

    This object ensures that every signal and hypothesis can be traced back
    to its source, fulfilling a key requirement for a governed discovery pipeline.
    """
    source: str  # e.g., 'Kepler', 'TESS', 'Synthetic'
    instrument_id: str  # e.g., 'KIC-12345', 'TIC-67890'
    
    # Observation details
    cadence_s: float
    mission_quarter: Optional[int] = None
    mission_sector: Optional[int] = None

    # Processing details
    detrending_method: str = 'PDC-MAP'
    noise_model: Optional[str] = 'SimulatedGaussian'

    # For synthetic data
    injected: bool = False
    injected_params: dict = field(default_factory=dict)
