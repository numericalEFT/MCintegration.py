import unittest
import torch
import numpy as np
from base import BaseDistribution, Uniform


class TestBaseDistribution(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.bounds_list = [[0.0, 1.0], [2.0, 3.0]]
        self.bounds_tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        self.device = "cpu"
        self.dtype = torch.float64

    def test_init_with_list(self):
        base_dist = BaseDistribution(self.bounds_list, self.device, self.dtype)
        self.assertEqual(base_dist.bounds.tolist(), self.bounds_list)
        self.assertEqual(base_dist.dim, 2)
        self.assertEqual(base_dist.device, self.device)
        self.assertEqual(base_dist.dtype, self.dtype)

    def test_init_with_tensor(self):
        base_dist = BaseDistribution(self.bounds_tensor, self.device, self.dtype)
        self.assertTrue(torch.equal(base_dist.bounds, self.bounds_tensor))
        self.assertEqual(base_dist.dim, 2)
        self.assertEqual(base_dist.device, self.device)
        self.assertEqual(base_dist.dtype, self.dtype)

    def test_init_with_invalid_bounds(self):
        with self.assertRaises(ValueError):
            BaseDistribution("invalid_bounds", self.device, self.dtype)

    def test_sample_not_implemented(self):
        base_dist = BaseDistribution(self.bounds_list, self.device, self.dtype)
        with self.assertRaises(NotImplementedError):
            base_dist.sample()

    def tearDown(self):
        # Common teardown for all tests
        pass


class TestUniform(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.bounds_list = [[0.0, 1.0], [2.0, 3.0]]
        self.bounds_tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        self.device = "cpu"
        self.dtype = torch.float64
        self.uniform_dist = Uniform(self.bounds_list, self.device, self.dtype)

    def test_init_with_list(self):
        self.assertEqual(self.uniform_dist.bounds.tolist(), self.bounds_list)
        self.assertEqual(self.uniform_dist.dim, 2)
        self.assertEqual(self.uniform_dist.device, self.device)
        self.assertEqual(self.uniform_dist.dtype, self.dtype)

    def test_init_with_tensor(self):
        uniform_dist = Uniform(self.bounds_tensor, self.device, self.dtype)
        self.assertTrue(torch.equal(uniform_dist.bounds, self.bounds_tensor))
        self.assertEqual(uniform_dist.dim, 2)
        self.assertEqual(uniform_dist.device, self.device)
        self.assertEqual(uniform_dist.dtype, self.dtype)

    def test_sample_within_bounds(self):
        nsamples = 1000
        samples, log_detJ = self.uniform_dist.sample(nsamples)
        self.assertEqual(samples.shape, (nsamples, 2))
        self.assertTrue(torch.all(samples[:, 0] >= 0.0))
        self.assertTrue(torch.all(samples[:, 0] <= 1.0))
        self.assertTrue(torch.all(samples[:, 1] >= 2.0))
        self.assertTrue(torch.all(samples[:, 1] <= 3.0))
        self.assertEqual(log_detJ.shape, (nsamples,))
        self.assertTrue(
            torch.allclose(
                log_detJ, torch.tensor([np.log(1.0) + np.log(1.0)] * nsamples)
            )
        )

    def test_sample_with_single_sample(self):
        samples, log_detJ = self.uniform_dist.sample(1)
        self.assertEqual(samples.shape, (1, 2))
        self.assertTrue(torch.all(samples[:, 0] >= 0.0))
        self.assertTrue(torch.all(samples[:, 0] <= 1.0))
        self.assertTrue(torch.all(samples[:, 1] >= 2.0))
        self.assertTrue(torch.all(samples[:, 1] <= 3.0))
        self.assertEqual(log_detJ.shape, (1,))
        self.assertTrue(
            torch.allclose(log_detJ, torch.tensor([np.log(1.0) + np.log(1.0)]))
        )

    def test_sample_with_zero_samples(self):
        samples, log_detJ = self.uniform_dist.sample(0)
        self.assertEqual(samples.shape, (0, 2))
        self.assertEqual(log_detJ.shape, (0,))

    def tearDown(self):
        # Common teardown for all tests
        pass


if __name__ == "__main__":
    unittest.main()
