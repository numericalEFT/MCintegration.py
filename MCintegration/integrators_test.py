import unittest
import torch
import numpy as np
from integrators import Integrator, MonteCarlo, MarkovChainMonteCarlo

# from base import LinearMap, Uniform
from maps import Configuration


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.bounds = torch.tensor([[0.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
        self.f = lambda x, fx: torch.sum(x**2, dim=1, keepdim=True)
        self.batch_size = 1000

    def test_initialization(self):
        integrator = Integrator(
            bounds=self.bounds, f=self.f, batch_size=self.batch_size
        )
        self.assertEqual(integrator.dim, 2)
        self.assertEqual(integrator.batch_size, 1000)
        self.assertEqual(integrator.f_dim, 1)
        # Check map type and properties instead of direct instance check
        self.assertTrue(hasattr(integrator.maps, "forward_with_detJ"))
        self.assertTrue(hasattr(integrator.maps, "device"))
        self.assertTrue(hasattr(integrator.maps, "dtype"))

    def test_bounds_conversion(self):
        # Test various input types
        test_cases = [
            (np.array([[0.0, 1.0], [-1.0, 1.0]]), "numpy array"),
            ([[0.0, 1.0], [-1.0, 1.0]], "list"),
            (
                torch.tensor([[0.0, 1.0], [-1.0, 1.0]], dtype=torch.float32),
                "float32 tensor",
            ),
            (
                torch.tensor([[0.0, 1.0], [-1.0, 1.0]], dtype=torch.float64),
                "float64 tensor",
            ),
        ]

        for bounds, desc in test_cases:
            with self.subTest(desc=desc):
                integrator = Integrator(bounds=bounds, f=self.f)
                self.assertIsInstance(integrator.bounds, torch.Tensor)
                self.assertEqual(
                    integrator.bounds.dtype, torch.float64
                )  # Check dtype conversion
                self.assertEqual(
                    integrator.bounds.shape, (2, 2)
                )  # Check shape preservation

    def test_invalid_bounds(self):
        invalid_cases = [
            ("invalid", TypeError, "string bounds"),
            (123, TypeError, "integer bounds"),
            ([[1, 2], [3]], ValueError, "inconsistent dimensions"),
            (
                np.array([[1.0]]),
                AssertionError,
                "invalid shape",
            ),  # Changed from single value
            (np.array([]), IndexError, "empty bounds"),  # Changed from empty list
        ]

        for bounds, error_type, desc in invalid_cases:
            with self.subTest(desc=desc):
                with self.assertRaises(error_type):
                    Integrator(bounds=bounds, f=self.f)

    def test_device_handling(self):
        if torch.cuda.is_available():
            integrator = Integrator(bounds=self.bounds, f=self.f, device="cuda")
            self.assertTrue(integrator.bounds.is_cuda)
            self.assertTrue(integrator.maps.device == "cuda")

    def test_dtype_handling(self):
        dtypes = [torch.float32, torch.float64]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                integrator = Integrator(bounds=self.bounds, f=self.f, dtype=dtype)
                self.assertEqual(integrator.bounds.dtype, dtype)
                self.assertEqual(integrator.maps.dtype, dtype)


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.bounds = torch.tensor([[-1.0, 1.0], [-1, 1]], dtype=torch.float64)

        def simple_integral(x, fx):
            fx[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double() / torch.pi
            return fx[:, 0]

        self.simple_integral = simple_integral
        self.mc = MonteCarlo(
            bounds=self.bounds, f=self.simple_integral, batch_size=10000
        )

    def test_simple_integration(self):
        # Test with different numbers of evaluations
        test_cases = [
            (10000, 1.0, 0.1),
            (100000, 1.0, 0.05),
            (1000000, 1.0, 0.01),
        ]

        for neval, expected, tolerance in test_cases:
            with self.subTest(neval=neval):
                result = self.mc(neval=neval)
                if hasattr(result, "mean"):
                    value = result.mean
                else:
                    value = result
                self.assertAlmostEqual(float(value), expected, delta=tolerance)

    def test_multidimensional_integration(self):
        # Test integration over higher dimensions
        bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)

        def volume_integral(x, fx):
            fx[:] = torch.ones_like(fx)
            return fx[:, 0]

        mc = MonteCarlo(bounds=bounds, f=volume_integral, batch_size=10000)
        result = mc(neval=100000)
        expected_volume = 1.0  # Unit square

        if hasattr(result, "mean"):
            value = result.mean
        else:
            value = result
        self.assertAlmostEqual(float(value), expected_volume, places=1)

    def test_convergence(self):
        # Test that uncertainty decreases with more samples
        results = []
        nevals = [10000, 100000, 1000000]

        for neval in nevals:
            result = self.mc(neval=neval)
            if hasattr(result, "mean"):
                value = result.mean
                uncertainty = result.sdev  # Changed from error to sdev
            else:
                value = result
                uncertainty = abs(value - 1.0)
            results.append(uncertainty)

        # Check that uncertainties decrease with more samples
        for i in range(len(results) - 1):
            self.assertGreater(results[i], results[i + 1])

    def test_batch_size_handling(self):
        test_cases = [
            (100, 32),  # Small neval
            (1000, 16),  # Medium neval
            (10000, 8),  # Large neval
        ]

        for neval, nblock in test_cases:
            with self.subTest(neval=neval, nblock=nblock):
                if neval < nblock * self.mc.batch_size:
                    with self.assertWarns(UserWarning):
                        self.mc(neval=neval, nblock=nblock)
                else:
                    # Should not raise warning
                    self.mc(neval=neval, nblock=nblock)


class TestMarkovChainMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.bounds = torch.tensor([[-1.0, 1.0], [-1, 1]], dtype=torch.float64)

        def simple_integral(x, fx):
            fx[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double() / torch.pi
            return fx[:, 0]

        self.simple_integral = simple_integral
        self.mcmc = MarkovChainMonteCarlo(
            bounds=self.bounds, f=self.simple_integral, batch_size=1000, nburnin=5
        )

    def test_proposal_distribution(self):
        # Test different proposal distributions
        def custom_proposal(dim, device, dtype, u, **kwargs):
            return torch.rand_like(u) * 0.5  # Restricted range

        test_cases = [
            (None, "default uniform"),
            (custom_proposal, "custom proposal"),
        ]

        for proposal_dist, desc in test_cases:
            with self.subTest(desc=desc):
                if proposal_dist:
                    mcmc = MarkovChainMonteCarlo(
                        bounds=self.bounds,
                        f=self.simple_integral,
                        proposal_dist=proposal_dist,
                    )
                else:
                    mcmc = self.mcmc

                config = Configuration(
                    mcmc.batch_size,
                    mcmc.dim,
                    mcmc.f_dim,
                    mcmc.device,
                    mcmc.dtype,
                )
                config.u, config.detJ = mcmc.q0.sample_with_detJ(mcmc.batch_size)
                new_u = mcmc.proposal_dist(mcmc.dim, mcmc.device, mcmc.dtype, config.u)
                self.assertEqual(new_u.shape, config.u.shape)
                self.assertTrue(torch.all(new_u >= 0) and torch.all(new_u <= 1))

    def test_burnin_effect(self):
        # Test different burnin values
        test_cases = [
            (0, 0.2),  # No burnin
            (5, 0.15),  # Short burnin
            (20, 0.1),  # Long burnin
        ]

        for nburnin, tolerance in test_cases:
            with self.subTest(nburnin=nburnin):
                mcmc = MarkovChainMonteCarlo(
                    bounds=self.bounds,
                    f=self.simple_integral,
                    batch_size=1000,
                    nburnin=nburnin,
                )
                result = mcmc(neval=50000, mix_rate=0.5, nblock=10)
                if hasattr(result, "mean"):
                    value = result.mean
                else:
                    value = result
                self.assertAlmostEqual(float(value), 1.0, delta=tolerance)

    # def test_mix_rate_sensitivity(self):
    #     # Modified mix rate test to be more robust
    #     mix_rates = [0.0, 0.5, 1.0]
    #     results = []

    #     for mix_rate in mix_rates:
    #         accumulated_error = 0
    #         n_trials = 3  # Run multiple trials for each mix_rate

    #         for _ in range(n_trials):
    #             result = self.mcmc(neval=50000, mix_rate=mix_rate, nblock=10)
    #             if hasattr(result, "mean"):
    #                 value = result.mean
    #                 error = result.sdev
    #             else:
    #                 value = result
    #                 error = abs(float(value) - 1.0)
    #             accumulated_error += error

    #         results.append(accumulated_error / n_trials)

    #     # We expect moderate mix rates to have lower average error
    #     self.assertLess(results[1], max(results[0], results[2]))


class TestDistributedFunctionality(unittest.TestCase):
    @unittest.skipIf(not torch.distributed.is_available(), "Distributed not available")
    def test_distributed_initialization(self):
        bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        f = lambda x, fx: torch.ones_like(x)
        integrator = Integrator(bounds=bounds, f=f)
        self.assertEqual(integrator.rank, 0)
        self.assertEqual(integrator.world_size, 1)

    @unittest.skipIf(not torch.distributed.is_available(), "Distributed not available")
    def test_multi_gpu_consistency(self):
        if torch.cuda.device_count() >= 2:
            bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
            f = lambda x, fx: torch.ones_like(x)

            # Create two integrators on different devices
            integrator1 = Integrator(bounds=bounds, f=f, device="cuda:0")
            integrator2 = Integrator(bounds=bounds, f=f, device="cuda:1")

            # Results should be consistent across devices
            result1 = integrator1(neval=10000)
            result2 = integrator2(neval=10000)

            if hasattr(result1, "mean"):
                value1, value2 = result1.mean, result2.mean
            else:
                value1, value2 = result1, result2

            self.assertAlmostEqual(float(value1), float(value2), places=1)


if __name__ == "__main__":
    unittest.main()
