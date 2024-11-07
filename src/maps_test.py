import unittest
import torch

# import numpy as np
from maps import Map, CompositeMap, Linear, Vegas


class TestMap(unittest.TestCase):
    def setUp(self):
        self.bounds = [[0, 1], [2, 3]]
        self.device = "cpu"
        self.dtype = torch.float64
        self.map = Map(self.bounds, self.device, self.dtype)

    def test_init_with_list(self):
        self.assertEqual(self.map.bounds.tolist(), self.bounds)
        self.assertEqual(self.map.dim, 2)
        self.assertEqual(self.map.device, self.device)
        self.assertEqual(self.map.dtype, self.dtype)

    def test_init_with_tensor(self):
        bounds_tensor = torch.tensor(self.bounds, dtype=self.dtype, device=self.device)
        map_tensor = Map(bounds_tensor, self.device, self.dtype)
        self.assertTrue(torch.equal(map_tensor.bounds, bounds_tensor))
        self.assertEqual(map_tensor.dim, 2)
        self.assertEqual(map_tensor.device, self.device)
        self.assertEqual(map_tensor.dtype, self.dtype)

    def test_init_with_invalid_bounds(self):
        with self.assertRaises(ValueError):
            Map("invalid_bounds", self.device, self.dtype)

    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.map.forward(torch.tensor([0.5, 0.5], dtype=self.dtype))

    def test_inverse_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.map.inverse(torch.tensor([0.5, 0.5], dtype=self.dtype))


class TestCompositeMap(unittest.TestCase):
    def setUp(self):
        self.bounds1 = [[0, 1], [2, 3]]
        self.bounds2 = [[1, 2], [3, 4]]
        self.map1 = Linear(self.bounds1)
        self.map2 = Linear(self.bounds2)
        self.composite_map = CompositeMap([self.map1, self.map2])

    def test_init_with_empty_maps(self):
        with self.assertRaises(ValueError):
            CompositeMap([])

    def test_forward(self):
        u = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        expected_output = torch.tensor([[1.5, 5.5]], dtype=torch.float64)
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64)
        output, log_detJ = self.composite_map.forward(u)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))

    def test_inverse(self):
        x = torch.tensor([[1.5, 5.5]], dtype=torch.float64)
        expected_output = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64)
        output, log_detJ = self.composite_map.inverse(x)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.bounds = [[0, 1], [2, 3]]
        self.linear_map = Linear(self.bounds)

    def test_forward(self):
        u = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        expected_output = torch.tensor([[0.5, 2.5]], dtype=torch.float64)
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64)
        output, log_detJ = self.linear_map.forward(u)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))

    def test_inverse(self):
        x = torch.tensor([[0.5, 2.5]], dtype=torch.float64)
        expected_output = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64)
        output, log_detJ = self.linear_map.inverse(x)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))


class TestVegas(unittest.TestCase):
    def setUp(self):
        # Setup common parameters for tests
        self.bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        self.ninc = 10
        self.alpha = 0.5
        self.device = "cpu"
        self.dtype = torch.float64
        self.vegas = Vegas(
            self.bounds,
            ninc=self.ninc,
            alpha=self.alpha,
            device=self.device,
            dtype=self.dtype,
        )
        grid0 = torch.linspace(0, 1, 11, dtype=self.dtype)
        self.init_grid = torch.stack([grid0, grid0])

    def tearDown(self):
        # Teardown after each test
        del self.vegas

    def test_initialization(self):
        # Test initialization of the Vegas class
        self.assertEqual(self.vegas.dim, 2)
        self.assertEqual(self.vegas.alpha, self.alpha)
        self.assertEqual(self.vegas.ninc.tolist(), [self.ninc, self.ninc])
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertTrue(torch.equal(self.vegas.grid, self.init_grid))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))

    def test_add_training_data(self):
        # Test adding training data
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        fval = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.vegas.add_training_data(u, fval)
        self.assertIsNotNone(self.vegas.sum_f)
        self.assertIsNotNone(self.vegas.n_f)

    def test_adapt(self):
        # Test grid adaptation
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        fval = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.vegas.add_training_data(u, fval)
        self.vegas.adapt(alpha=self.alpha)
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))

    def test_extract_grid(self):
        # Test extracting the grid
        grid = self.vegas.extract_grid()
        self.assertEqual(len(grid), 2)
        self.assertEqual(len(grid[0]), self.ninc + 1)
        self.assertEqual(len(grid[1]), self.ninc + 1)

    def test_clear(self):
        # Test clearing accumulated data
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        fval = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.vegas.add_training_data(u, fval)
        self.vegas.clear()
        self.assertIsNone(self.vegas.sum_f)
        self.assertIsNone(self.vegas.n_f)

    def test_forward(self):
        # Test forward transformation
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        x, log_jac = self.vegas.forward(u)
        self.assertEqual(x.shape, u.shape)
        self.assertEqual(log_jac.shape, (u.shape[0],))

    def test_inverse(self):
        # Test inverse transformation
        x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        u, log_jac = self.vegas.inverse(x)
        self.assertEqual(u.shape, x.shape)
        self.assertEqual(log_jac.shape, (x.shape[0],))

    def test_train(self):
        # Test training the Vegas class
        def f(x, fx):
            fx[:, 0] = torch.sum(x, dim=1)
            return fx[:, 0]

        nsamples = 100
        epoch = 5
        self.vegas.train(nsamples, f, epoch=epoch, alpha=self.alpha)
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))

    def test_make_uniform(self):
        # Test the make_uniform method
        self.vegas.make_uniform()
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))
        self.assertTrue(torch.equal(self.vegas.grid, self.init_grid))
        self.assertIsNone(self.vegas.sum_f)
        self.assertIsNone(self.vegas.n_f)

    def test_edge_cases(self):
        # Test edge cases
        # Test with ninc as a list
        ninc_list = [5, 15]
        vegas_list = Vegas(
            self.bounds,
            ninc=ninc_list,
            alpha=self.alpha,
            device=self.device,
            dtype=self.dtype,
        )
        self.assertEqual(vegas_list.ninc.tolist(), ninc_list)
        del vegas_list

        # Test with alpha < 0
        vegas_neg_alpha = Vegas(
            self.bounds,
            ninc=self.ninc,
            alpha=-0.5,
            device=self.device,
            dtype=self.dtype,
        )
        self.assertEqual(vegas_neg_alpha.alpha, -0.5)
        del vegas_neg_alpha

        # Test with different device
        if torch.cuda.is_available():
            device_cuda = "cuda"
            vegas_cuda = Vegas(
                self.bounds,
                ninc=self.ninc,
                alpha=self.alpha,
                device=device_cuda,
                dtype=self.dtype,
            )
            self.assertEqual(vegas_cuda.device, device_cuda)
            del vegas_cuda


if __name__ == "__main__":
    unittest.main()
