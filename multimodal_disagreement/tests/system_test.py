import unittest
from multimodal_disagreement.metrics import disagreement, apply_veto

class TestSystem(unittest.TestCase):
    def test_example_from_paper(self):
        # Simulated independent branch probabilities from paper example
        p_temporal = 0.91
        p_spatial = 0.12

        # Calculate the disagreement metric (delta)
        delta = disagreement(p_temporal, p_spatial)
        
        # Verify delta calculation
        self.assertAlmostEqual(delta, 0.79)
        
        # Apply the recommended Disagreement Veto threshold
        is_vetoed = apply_veto(delta, threshold=0.3)
        
        # Verify veto application
        self.assertTrue(is_vetoed)
        
        print(f"\n[Test] Inter-modal Disagreement: {delta}")
        print(f"[Test] Flagged as False Positive: {is_vetoed}")

    def test_consensus_case(self):
        p_temporal = 0.85
        p_spatial = 0.88
        delta = disagreement(p_temporal, p_spatial)
        self.assertAlmostEqual(delta, 0.03)
        self.assertFalse(apply_veto(delta, threshold=0.3))

if __name__ == '__main__':
    unittest.main()
