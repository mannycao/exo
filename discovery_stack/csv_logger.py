import csv
import os
from datetime import datetime
from .types import TransitHypothesis

class CSVLogger:
    """Logs pipeline results to a CSV file."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the CSV file with a header if it doesn't exist."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "tic_id",
                    "sector",
                    "hypothesis_id",
                    "status",
                    "rationale",
                    "ml_confidence",
                    "disagreement_score",
                    "transit_depth",
                    "physics_clean"
                ])

    def log_hypothesis(self, hypothesis: TransitHypothesis):
        """Appends the details of a processed hypothesis to the CSV file."""
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.utcnow().isoformat()
            
            inference = hypothesis.inference
            rationale = " | ".join(hypothesis.logs)

            writer.writerow([
                timestamp,
                hypothesis.tic_id,
                hypothesis.sector,
                hypothesis.id,
                hypothesis.status.value,
                rationale,
                inference.get("posterior_mean", ""),
                inference.get("epistemic_uncertainty", ""),
                inference.get("transit_depth", ""),
                inference.get("physics_clean", "")
            ])
