import logging
import os
import json
import sys
from datetime import datetime
from .types import TransitHypothesis, PipelineStatus

# Import the new, explicit physics oracles
from validation.physics_oracles.transit_geometry import check_transit_shape
from validation.physics_oracles.orbital_stability import check_orbital_stability

# --- DR. V FIX: Robust Import Strategy ---
try:
    import config as lite_config
except ImportError:
    try:
        from lite import config as lite_config
    except ImportError:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import config as lite_config
# -----------------------------------------

logger = logging.getLogger(__name__)

class DiscoveryGovernor:
    """
    Applies scientific and data-quality rules to vet ML candidates.
    This is the 'Judge' of the pipeline, now using explicit oracles.
    It persists all decisions to a JSON log file for auditability.
    """
    def __init__(self, uncertainty_threshold=0.2, prob_threshold=0.8):
        self.uncertainty_threshold = uncertainty_threshold
        self.prob_threshold = prob_threshold
        
        # Setup Persistence
        self.log_path = os.path.join(lite_config.RESULTS_DIR, "discovery_decision_log.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Initialize log if empty
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)
                
        logger.info(f"DiscoveryGovernor initialized. Decision log: {self.log_path}")

    def audit(self, hypothesis: TransitHypothesis) -> TransitHypothesis:
        """
        Audits a single hypothesis against a set of governance gates.
        Writes the result to disk immediately.

        Args:
            hypothesis (TransitHypothesis): The hypothesis to audit.

        Returns:
            The updated hypothesis with a new status and log entries.
        """
        # Prepare the decision record
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis_id": getattr(hypothesis, 'id', 'unknown'),
            "ml_score": hypothesis.inference.get('posterior_mean', 0.0),
            "uncertainty": hypothesis.inference.get('epistemic_uncertainty', 0.0),
            "original_status": str(hypothesis.status),
            "new_status": None,
            "reason": None,
            "gates_passed": []
        }

        # Don't audit a hypothesis that already failed upstream
        if hypothesis.status == PipelineStatus.REJECTED:
            decision_record['new_status'] = "REJECTED"
            decision_record['reason'] = "Rejected upstream"
            self._persist_decision(decision_record)
            return hypothesis

        # === Layer 4: Physical & System Constraints ===
        # These checks are authoritative and cannot be overridden by the model.
        
        # 1. Transit Geometry Oracle
        if not check_transit_shape(hypothesis):
            # The reason is logged inside the oracle too, but we capture the status change here
            hypothesis.log("VETO (Transit Geometry): Shape consistent with Eclipsing Binary.")
            hypothesis.status = PipelineStatus.REJECTED
            
            decision_record['new_status'] = "REJECTED"
            decision_record['reason'] = "VETO: Transit Geometry (Eclipsing Binary)"
            self._persist_decision(decision_record)
            return hypothesis
            
        decision_record['gates_passed'].append("Geometry")

        # 2. Orbital Stability Oracle
        if not check_orbital_stability(hypothesis):
            hypothesis.log("VETO (Orbital Stability): Proposed orbit is not stable.")
            hypothesis.status = PipelineStatus.REJECTED
            
            decision_record['new_status'] = "REJECTED"
            decision_record['reason'] = "VETO: Orbital Stability"
            self._persist_decision(decision_record)
            return hypothesis

        decision_record['gates_passed'].append("Stability")

        # === Layer 2 / 3: Inference & Consistency Gates ===

        # 3. Uncertainty Gate
        uncertainty = hypothesis.inference.get('epistemic_uncertainty', 1.0)
        if uncertainty > self.uncertainty_threshold:
            hypothesis.log(f"REJECT: High Epistemic Uncertainty (Uncert={uncertainty:.3f} > {self.uncertainty_threshold})")
            hypothesis.status = PipelineStatus.REJECTED
            
            decision_record['new_status'] = "REJECTED"
            decision_record['reason'] = f"High Uncertainty ({uncertainty:.3f})"
            self._persist_decision(decision_record)
            return hypothesis
            
        decision_record['gates_passed'].append("Uncertainty")

        # 4. Score Gate
        probability = hypothesis.inference.get('posterior_mean', 0.0)
        if probability > self.prob_threshold:
            hypothesis.log(f"PASS: High probability score (Posterior Mean={probability:.3f} > {self.prob_threshold})")
            hypothesis.status = PipelineStatus.ML_CANDIDATE
            
            decision_record['new_status'] = "ML_CANDIDATE"
            decision_record['reason'] = f"High Confidence ({probability:.3f})"
        else:
            # It passed physics but score is low
            hypothesis.log(f"CLEAR: Low probability score (Posterior Mean={probability:.3f} <= {self.prob_threshold})")
            hypothesis.status = PipelineStatus.PHYSICS_CLEARED
            
            decision_record['new_status'] = "PHYSICS_CLEARED"
            decision_record['reason'] = f"Low Confidence ({probability:.3f})"

        # Persist the final success state
        self._persist_decision(decision_record)
        return hypothesis

    def _persist_decision(self, record):
        """Helper to write the decision record to the JSON log safely."""
        try:
            # Read existing
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            # Append new
            data.append(record)

            # Write back
            with open(self.log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist decision to log: {e}")
