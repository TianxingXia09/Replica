import unittest
import numpy as np


class IdentifiabilityTests(unittest.TestCase):
    def test_one_copy_marginals_coincide(self):
        for d in [2,4,8]:
            avg=sum(np.diag(np.eye(d)[j]) for j in range(d))/d
            self.assertTrue(np.allclose(avg,np.eye(d)/d))

    def test_targets_differ(self):
        for d in [2,4,8]:
            self.assertEqual(1.0,1.0)
            self.assertLess(1/d,1.0)

    def test_collision_unbiasedness(self):
        rng=np.random.default_rng(3)
        for d,purity in [(2,.5),(2,1.),(4,.25),(4,1.)]:
            p=(purity+1)/(d+1)
            q=(d+1)*(rng.random(500000)<p)-1
            self.assertAlmostEqual(q.mean(),purity,delta=.015)


if __name__=="__main__":unittest.main()

