import unittest
import torch
import numpy as np
from maps import Map, CompositeMap, Vegas, Configuration
from base import LinearMap


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.dim = 3
        self.f_dim = 2
        self.device = "cpu"
        self.dtype = torch.float64

    def test_configuration_initialization(self):
        config = Configuration(
            batch_size=self.batch_size,
            dim=self.dim,
            f_dim=self.f_dim,
            device=self.device,
            dtype=self.dtype,
        )

        self.assertEqual(config.batch_size, self.batch_size)
        self.assertEqual(config.dim, self.dim)
        self.assertEqual(config.f_dim, self.f_dim)
        self.assertEqual(config.device, self.device)

        self.assertEqual(config.u.shape, (self.batch_size, self.dim))
        self.assertEqual(config.x.shape, (self.batch_size, self.dim))
        self.assertEqual(config.fx.shape, (self.batch_size, self.f_dim))
        self.assertEqual(config.weight.shape, (self.batch_size,))
        self.assertEqual(config.detJ.shape, (self.batch_size,))

        self.assertEqual(config.u.dtype, self.dtype)
        self.assertEqual(config.x.dtype, self.dtype)
        self.assertEqual(config.fx.dtype, self.dtype)
        self.assertEqual(config.weight.dtype, self.dtype)
        self.assertEqual(config.detJ.dtype, self.dtype)


class TestMap(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float64
        self.map = Map(self.device, self.dtype)

    def test_init_with_list(self):
        self.assertEqual(self.map.device, self.device)
        self.assertEqual(self.map.dtype, self.dtype)

    def test_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.map.forward(torch.tensor([0.5, 0.5], dtype=self.dtype))

    def test_inverse_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.map.inverse(torch.tensor([0.5, 0.5], dtype=self.dtype))

    def test_forward_with_detJ(self):
        # Create a simple linear map for testing: x = u * A + b
        # With A=[1, 1] and b=[0, 0], we have x = u
        linear_map = LinearMap([1, 1], [0, 0], device=self.device)

        # Test forward_with_detJ method
        u = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=self.device)
        x, detJ = linear_map.forward_with_detJ(u)

        # Since it's a linear map from [0,0] to [1,1], x should equal u
        self.assertTrue(torch.allclose(x, u))

        # Determinant of Jacobian should be 1 for linear map with slope 1
        # forward_with_detJ returns actual determinant, not log
        self.assertAlmostEqual(detJ.item(), 1.0)

        # Test with a different linear map: x = u * [2, 3] + [1, 1]
        # So u = [0.5, 0.5] should give x = [0.5*2+1, 0.5*3+1] = [2, 2.5]
        linear_map2 = LinearMap([2, 3], [1, 1], device=self.device)
        u2 = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=self.device)
        x2, detJ2 = linear_map2.forward_with_detJ(u2)
        expected_x2 = torch.tensor(
            [[2.0, 2.5]], dtype=torch.float64, device=self.device
        )
        self.assertTrue(torch.allclose(x2, expected_x2))

        # Determinant should be 2 * 3 = 6
        self.assertAlmostEqual(detJ2.item(), 6.0)


class TestCompositeMap(unittest.TestCase):
    def setUp(self):
        # self.bounds1 = [[0, 1], [2, 3]]
        # self.bounds2 = [[1, 2], [3, 4]]
        self.map1 = LinearMap([1, 1], [0, 2])
        self.map2 = LinearMap([1, 1], [1, 3])
        self.composite_map = CompositeMap([self.map1, self.map2])
        self.device = self.composite_map.device

    def test_init_with_empty_maps(self):
        with self.assertRaises(ValueError):
            CompositeMap([])

    def test_forward(self):
        u = torch.tensor([[0.5, 0.5]], dtype=torch.float64, device=self.device)
        expected_output = torch.tensor(
            [[1.5, 5.5]], dtype=torch.float64, device=self.device
        )
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        output, log_detJ = self.composite_map.forward(u)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))

    def test_inverse(self):
        x = torch.tensor([[1.5, 5.5]], dtype=torch.float64, device=self.device)
        expected_output = torch.tensor(
            [[0.5, 0.5]], dtype=torch.float64, device=self.device
        )
        expected_log_detJ = torch.tensor([0.0], dtype=torch.float64, device=self.device)
        output, log_detJ = self.composite_map.inverse(x)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(log_detJ, expected_log_detJ))


class TestVegas(unittest.TestCase):
    def setUp(self):
        # Setup common parameters for tests
        self.dim = 2
        self.ninc = 10
        self.alpha = 0.5
        self.device = "cpu"
        self.dtype = torch.float64
        self.vegas = Vegas(
            self.dim,
            ninc=self.ninc,
            device=self.device,
            dtype=self.dtype,
        )
        grid0 = torch.linspace(0, 1, 11, dtype=self.dtype)
        self.init_grid = torch.stack([grid0, grid0])

        # u = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float64)
        # fval = torch.tensor([[1.0], [2.0], [-3.5]], dtype=torch.float64)
        self.sample = Configuration(
            batch_size=3, dim=2, f_dim=1, device=self.device, dtype=self.dtype
        )
        self.sample.u.uniform_(0, 1)
        self.sample.x[:] = self.sample.u
        self.sample.fx.uniform_(0, 1)
        self.sample.weight.fill_(1.0)
        self.sample.detJ.fill_(1.0)

    def tearDown(self):
        # Teardown after each test
        del self.vegas

    def test_initialization(self):
        # Test initialization of the Vegas class
        self.assertEqual(self.vegas.dim, 2)
        self.assertEqual(self.vegas.ninc.tolist(), [self.ninc, self.ninc])
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertTrue(torch.equal(self.vegas.grid, self.init_grid))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))

    def test_ninc_initialization_types(self):
        # Test ninc initialization with int
        vegas_int = Vegas(self.dim, ninc=5)
        self.assertEqual(vegas_int.ninc.tolist(), [5, 5])

        # Test ninc initialization with list
        vegas_list = Vegas(self.dim, ninc=[5, 10])
        self.assertEqual(vegas_list.ninc.tolist(), [5, 10])

        # Test ninc initialization with numpy array
        vegas_np = Vegas(self.dim, ninc=np.array([3, 7]))
        self.assertEqual(vegas_np.ninc.tolist(), [3, 7])

        # Test ninc initialization with torch tensor
        vegas_tensor = Vegas(self.dim, ninc=torch.tensor([4, 6]))
        self.assertEqual(vegas_tensor.ninc.tolist(), [4, 6])

        # Test ninc initialization with invalid type
        with self.assertRaises(ValueError):
            Vegas(self.dim, ninc="invalid")

    def test_ninc_shape_validation(self):
        # Test ninc shape validation
        with self.assertRaises(ValueError):
            Vegas(self.dim, ninc=[1, 2, 3])  # Wrong length

    def test_add_training_data(self):
        # Test adding training data
        self.vegas.add_training_data(self.sample)
        self.assertIsNotNone(self.vegas.sum_f)
        self.assertIsNotNone(self.vegas.n_f)

    def test_adapt(self):
        # Test grid adaptation
        self.vegas.add_training_data(self.sample)
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
        self.vegas.add_training_data(self.sample)
        self.vegas.clear()
        # self.assertIsNone(self.vegas.sum_f)
        # self.assertIsNone(self.vegas.n_f)
        self.assertTrue(torch.all(self.vegas.sum_f == 0).item())
        self.assertTrue(torch.all(self.vegas.sum_f == 0).item())
        # self.assertEqual(self.vegas.sum_f, torch.zeros_like(self.vegas.sum_f))
        # self.assertEqual(self.vegas.n_f, torch.zeros_like(self.vegas.n_f))

    def test_forward(self):
        # Test forward transformation
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        x, log_jac = self.vegas.forward(u)
        self.assertEqual(x.shape, u.shape)
        self.assertEqual(log_jac.shape, (u.shape[0],))

    def test_forward_with_detJ(self):
        # Test forward_with_detJ transformation
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        x, det_jac = self.vegas.forward_with_detJ(u)
        self.assertEqual(x.shape, u.shape)
        self.assertEqual(det_jac.shape, (u.shape[0],))

        # Determinant should be positive
        self.assertTrue(torch.all(det_jac > 0))

    def test_forward_out_of_bounds(self):
        # Test forward transformation with out-of-bounds u values
        u = torch.tensor(
            [[1.5, 0.5], [-0.1, 0.5]], dtype=torch.float64
        )  # Out-of-bounds values
        x, log_jac = self.vegas.forward(u)

        # Check that out-of-bounds x values are clamped to grid boundaries
        self.assertTrue(torch.all(x >= 0.0))
        self.assertTrue(torch.all(x <= 1.0))

        # Check log determinant adjustment for out-of-bounds cases
        self.assertEqual(log_jac.shape, (u.shape[0],))

    def test_inverse(self):
        # Test inverse transformation
        x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        u, log_jac = self.vegas.inverse(x)
        self.assertEqual(u.shape, x.shape)
        self.assertEqual(log_jac.shape, (x.shape[0],))

    def test_inverse_out_of_bounds(self):
        # Test inverse transformation with out-of-bounds x values
        x = torch.tensor(
            [[1.5, 0.5], [-0.1, 0.5]], dtype=torch.float64
        )  # Out-of-bounds values
        u, log_jac = self.vegas.inverse(x)

        # Check that out-of-bounds u values are clamped to [0, 1]
        self.assertTrue(torch.all(u >= 0.0))
        self.assertTrue(torch.all(u <= 1.0))

        # Check log determinant adjustment for out-of-bounds cases
        self.assertEqual(log_jac.shape, (x.shape[0],))

    def test_train(self):
        # Test training the Vegas class
        def f(x, fx):
            fx[:, 0] = torch.sum(x, dim=1)
            return fx[:, 0]

        batch_size = 100
        epoch = 5
        self.vegas.adaptive_training(batch_size, f, epoch=epoch, alpha=self.alpha)
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))

    def test_make_uniform(self):
        # Test the make_uniform method
        self.vegas.make_uniform()
        self.assertEqual(self.vegas.grid.shape, (2, self.ninc + 1))
        self.assertEqual(self.vegas.inc.shape, (2, self.ninc))
        self.assertTrue(torch.equal(self.vegas.grid, self.init_grid))
        self.assertTrue(torch.all(self.vegas.sum_f == 0).item())
        self.assertTrue(torch.all(self.vegas.sum_f == 0).item())

    def test_edge_cases(self):
        # Test edge cases
        # Test with ninc as a list
        ninc_list = [5, 15]
        vegas_list = Vegas(
            self.dim,
            ninc=ninc_list,
            device=self.device,
            dtype=self.dtype,
        )
        self.assertEqual(vegas_list.ninc.tolist(), ninc_list)
        del vegas_list

        # Test with different device
        if torch.cuda.is_available():
            device_cuda = "cuda"
            vegas_cuda = Vegas(
                self.dim,
                ninc=self.ninc,
                device=device_cuda,
                dtype=self.dtype,
            )
            self.assertEqual(vegas_cuda.device, device_cuda)
            del vegas_cuda


if __name__ == "__main__":
    unittest.main()
