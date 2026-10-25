import unittest
from fractions import Fraction
from minimax_benchmark import variance_bound, sufficient_copy_budget


class MinimaxBoundsTests(unittest.TestCase):
    def test_mixed_qubit_counterexample_is_covered(self):
        # rho=I/2: equal outcomes with probability 1/2, kernel in {2,-1}.
        actual_variance = Fraction(1, 2)*4 + Fraction(1, 2) - Fraction(1, 2)**2
        self.assertEqual(actual_variance, Fraction(9, 4))
        self.assertEqual(variance_bound(1), actual_variance)
        self.assertGreater(actual_variance, variance_bound(1, pure_product=True))

    def test_pure_product_budgets(self):
        self.assertEqual([sufficient_copy_budget(k, pure_product=True) for k in range(1,7)],
                         [66,114,272,730,2064,6012])

    def test_strict_integer_pair_boundary(self):
        for pure in [False, True]:
            for k in range(1,7):
                B = sufficient_copy_budget(k, pure_product=pure)
                floor_squared = ((1-Fraction(1,2**k))/2)**2
                v = variance_bound(k, pure_product=pure)
                self.assertLess(v/(B//2), floor_squared)
                self.assertGreaterEqual(v/((B-1)//2), floor_squared)

    def test_general_bound_dominates_all_purity_relaxations(self):
        for k in range(1,7):
            for j in range(101):
                p = Fraction(1,2**k)+(1-Fraction(1,2**k))*Fraction(j,100)
                self.assertLessEqual(3**k-1+p-p*p,variance_bound(k))


if __name__ == '__main__':
    unittest.main()
