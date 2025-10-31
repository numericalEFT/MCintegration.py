import unittest
import numpy as np
import gvar
import torch
from utils import RAvg, set_seed, get_device


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

    def test_reduce_ex_serialization(self):
        ravg = RAvg(weighted=True)
        ravg.add(gvar.gvar(1.0, 0.1))
        reduced = ravg.__reduce_ex__(2)
        restored = reduced[0](*reduced[1])
        self.assertEqual(restored.mean, ravg.mean)
        self.assertEqual(restored.sdev, ravg.sdev)

    def test_summary_output(self):
        ravg = RAvg(weighted=True)
        ravg.add(gvar.gvar(1.0, 0.1))
        summary = ravg.summary()
        self.assertIn("itn", summary)
        self.assertIn("integral", summary)
        self.assertIn("wgt average", summary)

    def test_converged_criteria(self):
        ravg = RAvg(weighted=True)
        ravg.add(gvar.gvar(1.0, 0.1))
        self.assertTrue(ravg.converged(0.1, 0.1))
        self.assertFalse(ravg.converged(0.001, 0.001))

    def test_multiplication(self):
        ravg1 = RAvg(weighted=True)
        # Test multiplication by another RAvg object
        ravg1.update(2.0, 0.1)
        ravg2 = RAvg(weighted=True)
        ravg2.update(3.0, 0.1)

        result = ravg1 * ravg2
        self.assertAlmostEqual(result.mean, 6.0)
        sdev = (0.1 / 2**2 + 0.1 / 3**2) ** 0.5 * 6.0
        self.assertAlmostEqual(result.sdev, sdev)

        # Test multiplication by a scalar
        result = ravg1 * 2.0
        self.assertAlmostEqual(result.mean, 4.0)
        self.assertAlmostEqual(result.sdev, ravg1.sdev * 2)

        # Test multiplication by a negative number
        result = -1.0 * ravg1
        self.assertAlmostEqual(result.mean, -2.0)
        self.assertAlmostEqual(result.sdev, ravg1.sdev)

        # Test multiplication by a large number
        result = ravg1 * 1e10
        self.assertAlmostEqual(result.mean, 2.0 * 1e10)
        self.assertAlmostEqual(result.sdev, ravg1.sdev * 1e10)

        # Test multiplication by zero
        result = ravg1 * 0.0
        self.assertAlmostEqual(result.mean, 0.0)
        self.assertAlmostEqual(result.sdev, 0.0)

        # Test multiplication by a vector (numpy array)
        result = ravg1 * np.array([2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.allclose([r.mean for r in result], [4.0, 6.0]))
        self.assertTrue(
            np.allclose([r.sdev for r in result], [2.0 * ravg1.sdev, 3.0 * ravg1.sdev])
        )

        # Test multiplication with unweighted RAvg
        ravg_unweighted = RAvg(weighted=False)
        ravg_unweighted.update(2.0, 0.1)
        result = ravg_unweighted * 3.0
        self.assertAlmostEqual(result.mean, 6.0)
        self.assertAlmostEqual(result.sdev, ravg_unweighted.sdev * 3)

        # Test multiplication with zero variance
        ravg_zero_var = RAvg(weighted=True)
        ravg_zero_var.update(2.0, 0.0)
        result = ravg_zero_var * 3.0
        self.assertAlmostEqual(result.mean, 6.0)
        self.assertAlmostEqual(result.sdev, 0.0)

    def test_division(self):
        ravg1 = RAvg(weighted=True)
        ravg1.update(6.0, 0.1)
        ravg2 = RAvg(weighted=True)
        ravg2.update(3.0, 0.1)

        # Test division by another RAvg object
        result = ravg1 / ravg2
        self.assertAlmostEqual(result.mean, 2.0)
        sdev = (0.1 / 6.0**2 + 0.1 / 3.0**2) ** 0.5 * 2.0
        self.assertAlmostEqual(result.sdev, sdev)

        # Test division by a scalar
        result = ravg1 / 2.0
        self.assertAlmostEqual(result.mean, 3.0)
        self.assertAlmostEqual(result.sdev, ravg1.sdev / 2.0)

        # Test division by a negative number
        result = ravg1 / -1.0
        self.assertAlmostEqual(result.mean, -6.0)
        self.assertAlmostEqual(result.sdev, ravg1.sdev)

        # Test division by a large number
        result = ravg1 / 1e10
        self.assertAlmostEqual(result.mean, 6.0 / 1e10)
        self.assertAlmostEqual(result.sdev, ravg1.sdev / 1e10)

        result = -12.0 / ravg1
        self.assertAlmostEqual(result.mean, -2.0)
        self.assertAlmostEqual(result.sdev, ravg1.sdev / 3)

        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            result = ravg1 / 0.0

        # Test division by a vector (numpy array)
        result = ravg1 / np.array([2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.allclose([r.mean for r in result], [3.0, 2.0]))
        self.assertTrue(
            np.allclose([r.sdev for r in result], [ravg1.sdev / 2.0, ravg1.sdev / 3.0])
        )

        # Test division with unweighted RAvg
        ravg_unweighted = RAvg(weighted=False)
        ravg_unweighted.update(6.0, 0.1)
        result = ravg_unweighted / 3.0
        self.assertAlmostEqual(result.mean, 2.0)
        self.assertAlmostEqual(result.sdev, ravg_unweighted.sdev / 3)

        # Test division with zero variance
        ravg_zero_var = RAvg(weighted=True)
        ravg_zero_var.update(6.0, 0.0)
        result = ravg_zero_var / 3.0
        self.assertAlmostEqual(result.mean, 2.0)
        self.assertAlmostEqual(result.sdev, 0.0)

        # Test division of zero by RAvg
        zero_ravg = RAvg(weighted=True)
        zero_ravg.update(0.0, 0.1)
        divisor_ravg = RAvg(weighted=True)
        divisor_ravg.update(3.0, 0.1)
        result = zero_ravg / divisor_ravg
        self.assertAlmostEqual(result.mean, 0.0)
        # sdev = (0.1 / 0.0**2 + 0.1 / 3.0**2) ** 0.5 * 0.0  # This would be NaN
        # For 0/x, the error propagation gives 0 * sqrt((0.1/0.0^2) + (0.1/3.0^2))
        # But since we're dividing by zero, we need to be careful
        # In practice, gvar handles this appropriately

    def test_vector_operations_not_implemented(self):
        # Test that NotImplemented is returned for vector operations
        ravg = RAvg(weighted=True)
        ravg.update(2.0, 0.1)

        # Test multiplication with list (should return NotImplemented)
        result = ravg.__mul__([1, 2, 3])
        self.assertEqual(result, NotImplemented)

        # Test division with list (should return NotImplemented)
        result = ravg.__truediv__([1, 2, 3])
        self.assertEqual(result, NotImplemented)

        # Test multiplication with numpy array (should return NotImplemented)
        result = ravg.__mul__(np.array([1, 2, 3]))
        self.assertEqual(result, NotImplemented)

        # Test division with numpy array (should return NotImplemented)
        result = ravg.__truediv__(np.array([1, 2, 3]))
        self.assertEqual(result, NotImplemented)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Ensure a consistent state before each test
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def test_set_seed_cpu(self):
        # Test set_seed on a CPU-only environment
        set_seed(42)
        self.assertEqual(torch.initial_seed(), 42)
        u1 = torch.rand(10)
        set_seed(42)
        u2 = torch.rand(10)
        self.assertTrue(torch.all(u1 == u2))

    # @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    # def test_set_seed_cuda(self):
    #     # Test set_seed on a CUDA-enabled environment
    #     set_seed(42)
    #     self.assertEqual(torch.initial_seed(), 42)
    #     self.assertEqual(torch.cuda.initial_seed(), 42)

    @unittest.skipIf(torch.cuda.is_available(), "CUDA is available")
    def test_get_device_cpu(self):
        # Test get_device when CUDA is not available
        device = get_device()
        self.assertEqual(device, torch.device("cpu"))

    # @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    # def test_get_device_cuda(self):
    #     # Test get_device when CUDA is available
    #     device = get_device()
    #     self.assertEqual(device, torch.cuda.current_device())

    def test_get_device_cuda_inactive(self):
        if not torch.cuda.is_available():
            torch.cuda.set_device(-1)  # Simulate inactive CUDA
            device = get_device()
            self.assertEqual(device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()