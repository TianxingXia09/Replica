import unittest
import numpy as np

from local_protocol import (
    all_subsystems,
    local_overlap_kernel,
    partial_trace,
    probability_table,
    random_density,
    sample_pair,
    subsystem_purities,
    corrected_local_overlap_kernel,
)


class LocalProtocolTests(unittest.TestCase):
    def test_partial_trace_bell_state(self):
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        self.assertTrue(np.allclose(partial_trace(rho, (0,), 2), np.eye(2) / 2))
        self.assertTrue(np.allclose(partial_trace(rho, (1,), 2), np.eye(2) / 2))

    def test_local_kernel_estimates_overlap(self):
        rng = np.random.default_rng(19)
        n = 2
        rho = random_density(4, rng, rank=3)
        sigma = random_density(4, rng, rank=2)
        subsets = all_subsystems(n)
        trho, tsigma = probability_table(rho, n), probability_table(sigma, n)
        values = np.empty((120000, len(subsets)))
        for i in range(len(values)):
            a, b = sample_pair(trho, tsigma, n, rng)
            values[i] = local_overlap_kernel(a, b, subsets)
        expected = np.array([
            np.real(np.trace(partial_trace(rho, s, n) @ partial_trace(sigma, s, n)))
            for s in subsets
        ])
        self.assertTrue(np.all(np.abs(values.mean(axis=0) - expected) < 0.018))

    def test_purity_special_case(self):
        rng = np.random.default_rng(23)
        rho = random_density(8, rng, rank=4)
        subsets = all_subsystems(3, max_size=2)
        direct = subsystem_purities(rho, subsets, 3)
        expected = np.array([
            np.real(np.trace((r := partial_trace(rho, s, 3)) @ r)) for s in subsets
        ])
        self.assertTrue(np.allclose(direct, expected))

    def test_readout_correction_removes_bias(self):
        rng = np.random.default_rng(29)
        n = 2
        rho = random_density(4, rng, rank=2)
        subsets = all_subsystems(n)
        table = probability_table(rho, n)
        rates = np.array([0.04, 0.08])
        raw = np.zeros(len(subsets))
        corrected = np.zeros(len(subsets))
        shots = 180000
        for _ in range(shots):
            a, b = sample_pair(table, table, n, rng)
            a_noisy = np.bitwise_xor(a, rng.random(n) < rates)
            b_noisy = np.bitwise_xor(b, rng.random(n) < rates)
            raw += local_overlap_kernel(a_noisy, b_noisy, subsets)
            corrected += corrected_local_overlap_kernel(
                a_noisy, b_noisy, subsets, rates
            )
        truth = subsystem_purities(rho, subsets, n)
        raw /= shots
        corrected /= shots
        self.assertGreater(np.max(np.abs(raw - truth)), 0.025)
        self.assertLess(np.max(np.abs(corrected - truth)), 0.025)

    def test_exact_second_moment_identity(self):
        rng = np.random.default_rng(31)
        n = 2
        rho = random_density(4, rng, rank=3)
        table = probability_table(rho, n)
        subsets = [(0, 1)]
        shots = 180000
        values = np.empty(shots)
        for i in range(shots):
            a, b = sample_pair(table, table, n, rng)
            values[i] = local_overlap_kernel(a, b, subsets)[0]
        p0 = 1.0
        p1 = float(np.real(np.trace((r1 := partial_trace(rho, (0,), n)) @ r1)))
        p2 = float(np.real(np.trace((r2 := partial_trace(rho, (1,), n)) @ r2)))
        p12 = float(np.real(np.trace(rho @ rho)))
        exact = 4 * p0 + 2 * p1 + 2 * p2 + p12
        self.assertAlmostEqual(float(np.mean(values**2)), exact, delta=0.06)
        self.assertLessEqual(exact, 3**2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
