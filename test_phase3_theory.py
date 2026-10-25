import unittest

import numpy as np

from local_protocol import random_density
from phase3_theory import (
    block_temporal_dispersion,
    gram_matrix,
    mismatch_bias,
    mismatch_hs_bound,
    safe_dispersion_gate,
    synchronized_beats_temporal,
)


class PhaseBoundaryTests(unittest.TestCase):
    def test_boundary_excludes_heldout_dispersion(self):
        import pandas as pd
        from analyze_phase3_boundary import calibration_boundary_margin
        frame = pd.DataFrame(dict(cal_dispersion_sq=[0.3],
            cal_sync_noise_mse=[0.2], cal_temporal_noise_mse=[0.1],
            test_dispersion_sq=[999.0]))
        self.assertAlmostEqual(calibration_boundary_margin(frame).iloc[0], 0.2)
        frame.test_dispersion_sq = -999.0
        self.assertAlmostEqual(calibration_boundary_margin(frame).iloc[0], 0.2)

    def test_block_dispersion_matches_pairwise_definition(self):
        rng = np.random.default_rng(101)
        states = [random_density(4, rng, rank=r) for r in (1, 2, 4)]
        purity = np.asarray([np.real(np.trace(r @ r)) for r in states])
        gram = gram_matrix(states)
        labels = np.asarray([0, 1, 2, 0, 2, 1, 1, 0])
        observed = block_temporal_dispersion(labels, purity, gram, width=4)
        expected = 0.0
        for block in (labels[:4], labels[4:]):
            p = np.mean([purity[i] for i in block])
            q = np.mean(
                [
                    gram[label_i, label_j]
                    for position_i, label_i in enumerate(block)
                    for position_j, label_j in enumerate(block)
                    if position_i != position_j
                ]
            )
            expected += 0.5 * (p - q)
        self.assertAlmostEqual(observed, expected, places=12)
        self.assertGreaterEqual(observed, -1e-12)

    def test_boundary_decision(self):
        self.assertTrue(synchronized_beats_temporal(0.3, 0.10, 0.02))
        self.assertFalse(synchronized_beats_temporal(0.2, 0.10, 0.02))

    def test_safe_gate_uses_only_observable_inputs(self):
        estimate, selected = safe_dispersion_gate(
            np.array([0.9, 0.7]),
            np.array([0.5, 0.69]),
            np.array([0.01, 0.01]),
            multiplier=0.5,
        )
        self.assertTrue(np.array_equal(selected, np.array([True, False])))
        self.assertTrue(np.allclose(estimate, np.array([0.9, 0.69])))

    def test_mismatch_bound(self):
        rng = np.random.default_rng(102)
        for dimension in (2, 4, 8):
            for _ in range(20):
                rho = random_density(dimension, rng)
                sigma = random_density(dimension, rng)
                self.assertLessEqual(
                    abs(mismatch_bias(rho, sigma)),
                    mismatch_hs_bound(rho, sigma) + 1e-12,
                )


if __name__ == "__main__":
    unittest.main()
