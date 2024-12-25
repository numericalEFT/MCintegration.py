import unittest
import numpy as np
import gvar
from utils import RAvg


class TestRAvg(unittest.TestCase):
    def setUp(self):
        # Initialize common variables for tests
        self.weighted_ravg = RAvg(weighted=True)
        self.unweighted_ravg = RAvg(weighted=False)
        self.test_results = [
            gvar.gvar(1.0, 0.1),
            gvar.gvar(2.0, 0.2),
            gvar.gvar(3.0, 0.3),
        ]
        _means = np.array([1.0, 2.0, 3.0])
        _sdevs = np.array([0.1, 0.2, 0.3])
        self.unweighted_sdev = np.sum(_sdevs**2) ** 0.5 / 3
        self.weighted_sdev = np.sum(1 / _sdevs**2) ** -0.5
        self.weighted_mean = np.sum(_means / _sdevs**2) * self.weighted_sdev**2

    def tearDown(self):
        # Clean up after tests
        pass

    def test_init_weighted(self):
        self.assertTrue(self.weighted_ravg.weighted)
        self.assertEqual(self.weighted_ravg._wlist, [])
        self.assertEqual(self.weighted_ravg._mlist, [])

    def test_init_unweighted(self):
        self.assertFalse(self.unweighted_ravg.weighted)
        self.assertEqual(self.unweighted_ravg._sum, 0.0)
        self.assertEqual(self.unweighted_ravg._varsum, 0.0)
        self.assertEqual(self.unweighted_ravg._count, 0)
        self.assertEqual(self.unweighted_ravg._mlist, [])

    def test_add_weighted(self):
        for res in self.test_results:
            self.weighted_ravg.add(res)
        self.assertEqual(len(self.weighted_ravg.itn_results), 3)
        self.assertEqual(len(self.weighted_ravg._wlist), 3)
        self.assertEqual(len(self.weighted_ravg._mlist), 3)
        self.assertAlmostEqual(self.weighted_ravg.mean, self.weighted_mean)
        self.assertAlmostEqual(self.weighted_ravg.sdev, self.weighted_sdev)

    def test_add_unweighted(self):
        for res in self.test_results:
            self.unweighted_ravg.add(res)
        self.assertEqual(len(self.unweighted_ravg.itn_results), 3)
        self.assertEqual(self.unweighted_ravg.mean, 2.0)
        self.assertAlmostEqual(self.unweighted_ravg.sdev, self.unweighted_sdev)
        self.assertEqual(self.unweighted_ravg._sum, 6.0)
        self.assertEqual(self.unweighted_ravg._varsum, 0.14)
        self.assertEqual(self.unweighted_ravg._count, 3)

    def test_update(self):
        self.weighted_ravg.update(1.0, 0.1, last_neval=10)
        self.assertEqual(len(self.weighted_ravg.itn_results), 1)
        self.assertEqual(self.weighted_ravg.sum_neval, 10)
        self.weighted_ravg.update(2.0, 0.2, last_neval=20)
        self.assertEqual(len(self.weighted_ravg.itn_results), 2)
        self.assertEqual(self.weighted_ravg.sum_neval, 30)

        self.unweighted_ravg.update(1.0, 0.1, last_neval=10)
        self.assertEqual(len(self.unweighted_ravg.itn_results), 1)
        self.assertEqual(self.unweighted_ravg.sum_neval, 10)
        self.unweighted_ravg.update(2.0, 0.2, last_neval=20)
        self.assertEqual(len(self.unweighted_ravg.itn_results), 2)
        self.assertEqual(self.unweighted_ravg.sum_neval, 30)

    def test_extend(self):
        ravg = RAvg(weighted=True)
        for res in self.test_results:
            ravg.add(res)
        ravg.sum_neval = 30
        self.weighted_ravg.extend(ravg)
        self.assertEqual(len(self.weighted_ravg.itn_results), 3)
        self.assertEqual(self.weighted_ravg.sum_neval, 30)

    def test_reduce_ex(self):
        self.weighted_ravg.sum_neval = 10
        reduced = self.weighted_ravg.__reduce_ex__(2)
        self.assertEqual(reduced[0], RAvg)
        self.assertTrue(reduced[1][0])
        self.assertEqual(reduced[1][2], 10)

    def test_remove_gvars(self):
        self.weighted_ravg.sum_neval = 10
        ravg = self.weighted_ravg._remove_gvars([])
        self.assertEqual(ravg.weighted, self.weighted_ravg.weighted)
        self.assertEqual(ravg.sum_neval, self.weighted_ravg.sum_neval)

    def test_distribute_gvars(self):
        self.weighted_ravg.sum_neval = 10
        ravg = self.weighted_ravg._distribute_gvars([])
        self.assertEqual(ravg.weighted, self.weighted_ravg.weighted)
        self.assertEqual(ravg.sum_neval, self.weighted_ravg.sum_neval)

    def test_chi2(self):
        for _ in range(3):
            self.weighted_ravg.add(self.test_results[0])
            self.unweighted_ravg.add(self.test_results[0])
        self.assertTrue(np.isclose(self.weighted_ravg.chi2, 0.0))
        self.assertTrue(np.isclose(self.unweighted_ravg.chi2, 0.0))

    def test_dof(self):
        for res in self.test_results:
            self.weighted_ravg.add(res)
        dof = self.weighted_ravg.dof
        self.assertEqual(dof, 2)

    def test_nitn(self):
        for res in self.test_results:
            self.weighted_ravg.add(res)
        nitn = self.weighted_ravg.nitn
        self.assertEqual(nitn, 3)

    def test_Q(self):
        self.weighted_ravg.add(self.test_results[0])
        Q = self.weighted_ravg.Q
        self.assertTrue(np.isnan(Q))
        self.weighted_ravg.add(self.test_results[0])
        self.assertIsInstance(Q, float)

    def test_avg_neval(self):
        nevals = [10, 20, 30]
        for i, res in enumerate(self.test_results):
            self.weighted_ravg.update(res.mean, res.var, nevals[i])
        self.assertEqual(self.weighted_ravg.sum_neval, 60)
        self.assertEqual(self.weighted_ravg.avg_neval, 20)

    def test_summary(self):
        for res in self.test_results:
            self.weighted_ravg.add(res)
        summary = self.weighted_ravg.summary()
        self.assertIsInstance(summary, str)

    def test_converged(self):
        self.weighted_ravg.add(gvar.gvar(1.0, 0.01))
        self.assertTrue(self.weighted_ravg.converged(0.1, 0.1))


if __name__ == "__main__":
    unittest.main()
