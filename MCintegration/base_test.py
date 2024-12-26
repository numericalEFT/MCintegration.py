# base_test.py
import unittest
import torch
import numpy as np
from base import BaseDistribution, Uniform, LinearMap


class TestBaseDistribution(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.dim = 2
        self.device = "cpu"
        self.dtype = torch.float32

    def test_init(self):
        base_dist = BaseDistribution(self.dim, self.device, self.dtype)
        self.assertEqual(base_dist.dim, self.dim)
        self.assertEqual(base_dist.device, self.device)
        self.assertEqual(base_dist.dtype, self.dtype)

    def test_sample_not_implemented(self):
        base_dist = BaseDistribution(self.dim, self.device, self.dtype)
        with self.assertRaises(NotImplementedError):
            base_dist.sample()

    def tearDown(self):
        # Common teardown for all tests
        pass


class TestUniform(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.dim = 2
        self.device = "cpu"
        self.dtype = torch.float32
        self.uniform_dist = Uniform(self.dim, self.device, self.dtype)

    def test_init(self):
        self.assertEqual(self.uniform_dist.dim, self.dim)
        self.assertEqual(self.uniform_dist.device, self.device)
        self.assertEqual(self.uniform_dist.dtype, self.dtype)

    def test_sample_within_bounds(self):
        batch_size = 1000
        samples, log_detJ = self.uniform_dist.sample(batch_size)
        self.assertEqual(samples.shape, (batch_size, self.dim))
        self.assertTrue(torch.all(samples >= 0.0))
        self.assertTrue(torch.all(samples <= 1.0))
        self.assertEqual(log_detJ.shape, (batch_size,))
        self.assertTrue(torch.allclose(log_detJ, torch.tensor([0.0] * batch_size)))

    def test_sample_with_single_sample(self):
        samples, log_detJ = self.uniform_dist.sample(1)
        self.assertEqual(samples.shape, (1, self.dim))
        self.assertTrue(torch.all(samples >= 0.0))
        self.assertTrue(torch.all(samples <= 1.0))
        self.assertEqual(log_detJ.shape, (1,))
        self.assertTrue(torch.allclose(log_detJ, torch.tensor([0.0])))

    def test_sample_with_zero_samples(self):
        samples, log_detJ = self.uniform_dist.sample(0)
        self.assertEqual(samples.shape, (0, self.dim))
        self.assertEqual(log_detJ.shape, (0,))

    def tearDown(self):
        # Common teardown for all tests
        pass


class TestLinearMap(unittest.TestCase):
    def setUp(self):
        _A = torch.tensor([2.0, 3.0], dtype=torch.float32)
        _b = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.linear_map = LinearMap(_A, _b, dtype=torch.float32)
        self.A = self.linear_map.A
        self.b = self.linear_map.b
        self.device = self.linear_map.device

    def test_forward(self):
        u = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32, device=self.device
        )
        expected_x = torch.tensor(
            [[3.0, 5.0], [5.0, 8.0]], dtype=torch.float32, device=self.device
        )
        expected_detJ = torch.tensor(
            [np.log(6.0), np.log(6.0)], dtype=torch.float32, device=self.device
        )

        x, detJ = self.linear_map.forward(u)

        self.assertTrue(torch.allclose(x, expected_x))
        self.assertTrue(torch.allclose(detJ, expected_detJ))

    def test_forward_with_detJ(self):
        u = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32, device=self.device
        )
        expected_x = torch.tensor(
            [[3.0, 5.0], [5.0, 8.0]], dtype=torch.float32, device=self.device
        )
        expected_detJ = torch.tensor(
            [6.0, 6.0], dtype=torch.float32, device=self.device
        )

        x, detJ = self.linear_map.forward_with_detJ(u)

        self.assertTrue(torch.allclose(x, expected_x))
        self.assertTrue(torch.allclose(detJ, expected_detJ))

    def test_inverse(self):
        x = torch.tensor(
            [[3.0, 5.0], [5.0, 8.0]], dtype=torch.float32, device=self.device
        )
        expected_u = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32, device=self.device
        )
        expected_detJ = torch.tensor(
            [np.log(6.0), np.log(6.0)], dtype=torch.float32, device=self.device
        )

        u, detJ = self.linear_map.inverse(x)

        self.assertTrue(torch.allclose(u, expected_u))
        self.assertTrue(torch.allclose(detJ, expected_detJ))

    def test_init_with_list(self):
        A_list = [2.0, 3.0]
        b_list = [1.0, 2.0]
        linear_map = LinearMap(A_list, b_list, dtype=torch.float32)

        self.assertTrue(torch.allclose(linear_map.A, self.A))
        self.assertTrue(torch.allclose(linear_map.b, self.b))

    def test_init_with_numpy_array(self):
        A_np = np.array([2.0, 3.0])
        b_np = np.array([1.0, 2.0])
        linear_map = LinearMap(A_np, b_np, dtype=torch.float32)

        self.assertTrue(torch.allclose(linear_map.A, self.A))
        self.assertTrue(torch.allclose(linear_map.b, self.b))


if __name__ == "__main__":
    unittest.main()
