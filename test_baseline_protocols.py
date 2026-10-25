import unittest
import numpy as np

from baseline_protocols import (
    bell_swap_probabilities,
    bell_swap_subset_kernel,
    global_rm_collision_from_purity,
    shadow_pair_kernel,
    windowed_shadow_u_statistics,
)


class BaselineProtocolTests(unittest.TestCase):
    def test_shadow_pair_local_values(self):
        s0 = np.array([0])
        s1 = np.array([1])
        b0 = np.array([0])
        b1 = np.array([1])
        self.assertEqual(shadow_pair_kernel(s0, b0, s0, b0, (0,)), 5.0)
        self.assertEqual(shadow_pair_kernel(s0, b0, s0, b1, (0,)), -4.0)
        self.assertEqual(shadow_pair_kernel(s0, b0, s1, b0, (0,)), 0.5)

    def test_windowed_u_statistic_matches_direct_pair_average(self):
        settings = np.array([[0], [0], [1], [2]], dtype=np.int8)
        bits = np.array([[0], [1], [0], [0]], dtype=np.int8)
        direct = []
        for i in range(4):
            for j in range(i + 1, 4):
                direct.append(
                    shadow_pair_kernel(settings[i], bits[i], settings[j], bits[j], (0,))
                )
        estimate = windowed_shadow_u_statistics(settings, bits, [(0,)], [4])[4][0]
        self.assertAlmostEqual(estimate, float(np.mean(direct)), places=12)

    def test_bell_swap_moments(self):
        purities = {(): 1.0, (0,): 0.5, (1,): 0.5, (0, 1): 1.0}
        signs, probabilities = bell_swap_probabilities(purities, 2)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        subsets = [(0,), (1,), (0, 1)]
        moments = sum(
            p * bell_swap_subset_kernel(s, subsets)
            for s, p in zip(signs, probabilities)
        )
        np.testing.assert_allclose(moments, [0.5, 0.5, 1.0], atol=1e-12)

    def test_global_rm_is_unbiased(self):
        purity = 0.73
        dimension = 2**3
        collision_probability = (1.0 + purity) / (dimension + 1.0)
        exact_expectation = (
            dimension * collision_probability - (1.0 - collision_probability)
        )
        self.assertAlmostEqual(exact_expectation, purity, places=14)
        rng = np.random.default_rng(801)
        self.assertIn(global_rm_collision_from_purity(purity, 3, rng), (-1.0, 8.0))


if __name__ == "__main__":
    unittest.main()
