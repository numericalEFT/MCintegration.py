import unittest
import torch
import numpy as np
from integrators import Integrator, MonteCarlo, MarkovChainMonteCarlo
from utils import RAvg


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.bounds = [[0, 1], [2, 3]]
        self.device = "cpu"
        self.dtype = torch.float64
        self.neval = 1000
        self.batch_size = 100

    def test_init_with_bounds(self):
        integrator = Integrator(
            bounds=self.bounds, device=self.device, dtype=self.dtype
        )
        self.assertEqual(integrator.bounds.tolist(), self.bounds)
        self.assertEqual(integrator.dim, 2)
        self.assertEqual(integrator.neval, self.neval)
        self.assertEqual(integrator.batch_size, self.neval)
        self.assertEqual(integrator.device, self.device)
        self.assertEqual(integrator.dtype, self.dtype)

    def test_init_with_maps(self):
        class MockMaps:
            def __init__(self, bounds, dtype):
                self.bounds = torch.tensor(bounds, dtype=dtype)
                self.dtype = dtype

        maps = MockMaps(bounds=self.bounds, dtype=self.dtype)
        integrator = Integrator(maps=maps, device=self.device, dtype=self.dtype)
        self.assertEqual(integrator.bounds.tolist(), self.bounds)
        self.assertEqual(integrator.dim, 2)
        self.assertEqual(integrator.neval, self.neval)
        self.assertEqual(integrator.batch_size, self.neval)
        self.assertEqual(integrator.device, self.device)
        self.assertEqual(integrator.dtype, self.dtype)

    def test_init_with_mismatched_dtype(self):
        class MockMaps:
            def __init__(self, bounds, dtype):
                self.bounds = bounds
                self.dtype = dtype

        maps = MockMaps(bounds=self.bounds, dtype=torch.float32)
        with self.assertRaises(ValueError):
            Integrator(maps=maps, device=self.device, dtype=self.dtype)

    def test_init_with_invalid_bounds(self):
        with self.assertRaises(TypeError):
            Integrator(bounds=123, device=self.device, dtype=self.dtype)

    def test_sample_without_maps(self):
        integrator = Integrator(
            bounds=self.bounds, device=self.device, dtype=self.dtype
        )
        u, log_detJ = integrator.sample(100)
        self.assertEqual(u.shape, (100, 2))
        self.assertEqual(log_detJ.shape, (100,))

    def test_sample_with_maps(self):
        class MockMaps:
            def __init__(self, bounds, dtype):
                self.bounds = bounds
                self.dtype = dtype

            def forward(self, u):
                return u, torch.zeros(u.shape[0], dtype=self.dtype)

        maps = MockMaps(bounds=self.bounds, dtype=self.dtype)
        integrator = Integrator(maps=maps, device=self.device, dtype=self.dtype)
        u, log_detJ = integrator.sample(100)
        self.assertEqual(u.shape, (100, 2))
        self.assertEqual(log_detJ.shape, (100,))


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        # Setup common parameters for tests
        self.maps = None
        self.bounds = [[0, 1]]
        self.q0 = None
        self.neval = 1000
        self.batch_size = 100
        self.device = "cpu"
        self.dtype = torch.float64
        self.monte_carlo = MonteCarlo(
            self.maps,
            self.bounds,
            self.q0,
            self.neval,
            self.batch_size,
            self.device,
            self.dtype,
        )

    def f(self, x, fx):
        fx[:, 0] = x.sum(dim=1)
        return fx[:, 0]

    def f2(self, x, fx):
        fx[:, 0] = x.sum(dim=1)
        fx[:, 1] = x.prod(dim=1)
        return fx.mean(dim=1)

    def tearDown(self):
        # Teardown after tests
        pass

    def test_initialization(self):
        # Test if the MonteCarlo class initializes correctly
        self.assertIsInstance(self.monte_carlo, MonteCarlo)
        self.assertEqual(self.monte_carlo.neval, self.neval)
        self.assertEqual(self.monte_carlo.batch_size, self.batch_size)
        self.assertEqual(self.monte_carlo.device, self.device)
        self.assertEqual(self.monte_carlo.dtype, self.dtype)
        self.assertTrue(
            torch.equal(
                self.monte_carlo.bounds,
                torch.tensor(self.bounds, dtype=self.dtype, device=self.device),
            )
        )
        self.assertEqual(self.monte_carlo.dim, 1)

    def test_call_single_tensor_output(self):
        # Test the __call__ method with a function that returns a single tensor
        result = self.monte_carlo(self.f)
        self.assertIsInstance(result, RAvg)
        self.assertEqual(result.shape, ())
        self.assertIsInstance(result.mean, float)
        self.assertIsInstance(result.sdev, float)

    def test_call_multiple_tensor_output(self):
        # Test the __call__ method with a function that returns a list of tensors
        result = self.monte_carlo(self.f2, f_dim=2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, RAvg)

    def test_call_invalid_output_type(self):
        # Test the __call__ method with a function that returns an invalid type
        def f(x):
            return "invalid"

        with self.assertRaises(TypeError):
            self.monte_carlo(f)

    def test_sample_method(self):
        # Test the sample method to ensure it returns the correct shape
        x, log_detJ = self.monte_carlo.sample(self.batch_size)
        self.assertEqual(x.shape, (self.batch_size, 1))
        self.assertEqual(log_detJ.shape, (self.batch_size,))

    def test_call_with_cuda(self):
        # Test the __call__ method with device = "cuda" if available
        if torch.cuda.is_available():
            monte_carlo_cuda = MonteCarlo(
                self.maps,
                self.bounds,
                self.q0,
                self.neval,
                self.batch_size,
                "cuda",
                self.dtype,
            )
            result_cuda = monte_carlo_cuda(self.f)
            self.assertIsInstance(result_cuda, RAvg)
            self.assertIsInstance(result_cuda.mean, float)
            self.assertIsInstance(result_cuda.sdev, float)
            self.assertEqual(result_cuda.shape, ())

    def test_call_with_different_dtype(self):
        # Test the __call__ method with different dtype values
        # Test with dtype = torch.float32
        monte_carlo_float32 = MonteCarlo(
            self.maps,
            self.bounds,
            self.q0,
            self.neval,
            self.batch_size,
            self.device,
            torch.float32,
        )
        result_float32 = monte_carlo_float32(self.f)
        self.assertIsInstance(result_float32, RAvg)
        self.assertIsInstance(result_float32.mean, float)
        self.assertIsInstance(result_float32.sdev, float)
        self.assertEqual(result_float32.shape, ())

    def test_call_with_different_bounds(self):
        # Test the __call__ method with different bounds values
        # Test with bounds = [0, 2]
        monte_carlo_bounds_0_2 = MonteCarlo(
            self.maps,
            [[0, 2]],
            self.q0,
            self.neval,
            self.batch_size,
            self.device,
            self.dtype,
        )
        result_bounds_0_2 = monte_carlo_bounds_0_2(self.f)
        self.assertIsInstance(result_bounds_0_2, RAvg)
        self.assertEqual(result_bounds_0_2.shape, ())

        # Test with bounds = [-1, 1]
        monte_carlo_bounds_minus1_1 = MonteCarlo(
            self.maps,
            [(-1, 1)],
            self.q0,
            self.neval,
            self.batch_size,
            self.device,
            self.dtype,
        )
        result_bounds_minus1_1 = monte_carlo_bounds_minus1_1(self.f)
        self.assertIsInstance(result_bounds_minus1_1, RAvg)
        self.assertEqual(result_bounds_minus1_1.shape, ())


class TestMCMC(unittest.TestCase):
    def setUp(self):
        # Setup common parameters for tests
        self.maps = None
        self.bounds = [(-1.2, 0.6)]
        self.q0 = None
        self.neval = 10000
        self.batch_size = 100
        self.nburnin = 500
        self.device = "cpu"
        self.dtype = torch.float64
        self.mcmc = MarkovChainMonteCarlo(
            self.maps,
            self.bounds,
            self.q0,
            self.neval,
            self.batch_size,
            self.nburnin,
            self.device,
            self.dtype,
        )

    def f(self, x, fx):
        fx[:, 0] = x.sum(dim=1)
        return fx[:, 0]

    def f2(self, x, fx):
        fx[:, 0] = x.sum(dim=1)
        fx[:, 1] = x.prod(dim=1)
        return fx.mean(dim=1)

    def tearDown(self):
        # Teardown after tests
        pass

    def test_initialization(self):
        # Test if the MarkovChainMonteCarlo class initializes correctly
        self.assertIsInstance(self.mcmc, MarkovChainMonteCarlo)
        self.assertEqual(self.mcmc.neval, self.neval)
        self.assertEqual(self.mcmc.batch_size, self.batch_size)
        self.assertEqual(self.mcmc.nburnin, self.nburnin)
        self.assertEqual(self.mcmc.device, self.device)
        self.assertEqual(self.mcmc.dtype, self.dtype)
        self.assertTrue(
            torch.equal(
                self.mcmc.bounds,
                torch.tensor(self.bounds, dtype=self.dtype, device=self.device),
            )
        )
        self.assertEqual(self.mcmc.dim, 1)

    def test_call_single_tensor_output(self):
        # Test the __call__ method with a function that returns a single tensor
        result = self.mcmc(self.f)
        self.assertIsInstance(result, RAvg)
        self.assertEqual(result.shape, ())
        self.assertIsInstance(result.mean, float)
        self.assertIsInstance(result.sdev, float)

    def test_call_multiple_tensor_output(self):
        # Test the __call__ method with a function that returns a list of tensors
        result = self.mcmc(self.f2, f_dim=2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, RAvg)

    def test_call_invalid_output_type(self):
        # Test the __call__ method with a function that returns an invalid type
        def f(x):
            return "invalid"

        with self.assertRaises(TypeError):
            self.mcmc(f)

    def test_call_with_different_device(self):
        # Test the __call__ method with device = "cuda" if available
        if torch.cuda.is_available():
            mcmc_cuda = MarkovChainMonteCarlo(
                self.maps,
                self.bounds,
                self.q0,
                self.neval,
                self.batch_size,
                self.nburnin,
                "cuda",
                self.dtype,
            )
            result_cuda = mcmc_cuda(self.f)
            self.assertIsInstance(result_cuda, RAvg)
            self.assertEqual(result_cuda.shape, ())

    def test_call_with_different_dtype(self):
        # Test the __call__ method with different dtype values
        # Test with dtype = torch.float32
        mcmc_float32 = MarkovChainMonteCarlo(
            self.maps,
            self.bounds,
            self.q0,
            self.neval,
            self.batch_size,
            self.nburnin,
            self.device,
            torch.float32,
        )
        result_float32 = mcmc_float32(self.f)
        self.assertIsInstance(result_float32, RAvg)
        self.assertEqual(result_float32.shape, ())
        self.assertIsInstance(result_float32.mean, float)
        self.assertIsInstance(result_float32.sdev, float)

    def test_call_with_different_bounds(self):
        # Test the __call__ method with different bounds values
        # Test with bounds = [0, 2]
        mcmc_bounds = MarkovChainMonteCarlo(
            self.maps,
            [(3, 5.2), (-1.1, 0.2)],
            self.q0,
            self.neval,
            self.batch_size,
            self.nburnin,
            self.device,
            self.dtype,
        )
        result_bounds = mcmc_bounds(self.f)
        self.assertIsInstance(result_bounds, RAvg)
        self.assertEqual(result_bounds.shape, ())

    def test_call_with_different_proposal_dist(self):
        # Test the __call__ method with different proposal distributions
        from integrators import random_walk, gaussian

        # Test with uniform proposal distribution
        result_rw = self.mcmc(self.f, proposal_dist=random_walk)
        self.assertIsInstance(result_rw, RAvg)
        self.assertEqual(result_rw.shape, ())

        # Test with normal proposal distribution
        result_normal = self.mcmc(self.f, proposal_dist=gaussian)
        self.assertIsInstance(result_normal, RAvg)
        self.assertEqual(result_normal.shape, ())


if __name__ == "__main__":
    unittest.main()
