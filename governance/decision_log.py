import logging
import json
from discovery_stack.types import TransitHypothesis

logger = logging.getLogger(__name__)

class DecisionLog:
    """
    A simple logger to record every state transition and the evidence
    used to make the decision.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        # Clear the log file at the start of a run
        with open(self.log_path, 'w') as f:
            f.write("[]") 
        logger.info(f"Decision log initialized at {self.log_path}")

    def record(self, hypothesis: TransitHypothesis, reason: str):
        """
        Records a decision for a given hypothesis.
        """
        log_entry = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'hypothesis_id': hypothesis.id,
            'new_status': hypothesis.status.value,
            'reason': reason,
            'evidence': {
                'signal_params': hypothesis.signal_params,
                'inference': hypothesis.inference,
                'logs': hypothesis.logs
            }
        }
        
        # Read current log, append, and write back
        try:
            with open(self.log_path, 'r+') as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write to decision log: {e}")

def log_rejection(decision_logger: DecisionLog, hypothesis: TransitHypothesis):
    """Helper to log a rejection decision."""
    reason = "Rejected"
    if hypothesis.logs:
        # Get the last log message, which is usually the cause of rejection
        reason = hypothesis.logs[-1]
    decision_logger.record(hypothesis, reason)

def log_promotion(decision_logger: DecisionLog, hypothesis: TransitHypothesis):
    """Helper to log a promotion decision."""
    reason = "Promoted to ML Candidate"
    decision_logger.record(hypothesis, reason)