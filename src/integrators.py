from typing import Callable, Union, Dict, List, Tuple, Optional
import numpy as np
import torch
from .statistics import Statistics
from .utils import RAvg, to_tensor, from_tensor, discretize
from .mcmc import MCMCIntegrator, VegasMCMCIntegrator

class Integrator:
    """
    Base class for Monte Carlo integrators.
    """

    def __init__(self,
                 ndim: int,
                 integrand: Callable,
                 bounds: Union[List[Tuple[float, float]], np.ndarray],
                 use_gpu: bool = False,
                 num_cpus: int = 1,
                 discrete_dims: Optional[List[int]] = None):
        """
        Initialize the integrator.

        Args:
            ndim (int): Number of dimensions of the integral.
            integrand (Callable): The function to integrate. Should accept a tensor of shape (batch_size, ndim).
            bounds (Union[List[Tuple[float, float]], np.ndarray]): Integration bounds. Can be a list of tuples or a numpy array of shape (ndim, 2).
            use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
            num_cpus (int, optional): Number of CPU cores to use for parallel computation. Ignored if use_gpu is True. Defaults to 1.
            device (torch.device, optional): The device to use for computation. Defaults to torch.device("cuda" if use_gpu else "cpu").
            statistics (Statistics, optional): The statistics object to use for storing integration results and error estimates. Defaults to Statistics().
        """
        self.ndim = ndim
        self.integrand = integrand
        self.bounds = bounds
        self.use_gpu = use_gpu
        self.num_cpus = num_cpus
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.discrete_dims = discrete_dims or []
        self.statistics = Statistics()

    def _wrap_integrand(self, func: Callable) -> Callable:
        def wrapped(x):
            x_tensor = to_tensor(x, self.device)
            x_discrete = discretize(x_tensor, self.discrete_dims, self.bounds)
            result = func(x_discrete)
            return from_tensor(result)
        return wrapped

    # def integrate(self, neval: int) -> Union[float, np.ndarray, Dict]:
    def integrate(self, *args, **kwargs) -> Union[float, np.ndarray, Dict]:
        """
        Perform the integration.

        Args:
            neval (int): Number of function evaluations.

        Returns:
            Union[float, np.ndarray, Dict]: The estimated integral value. Could be a scalar, array, or dictionary.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_sample_points(self, num_points: int) -> torch.Tensor:
        """
        Generate sample points for the integration.

        Args:
            num_points (int): Number of sample points to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_points, ndim) containing the sample points.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_statistics(self) -> Statistics:
        """
        Get the statistics of the integration.

        Returns:
            Statistics: The statistics object containing integration results and error estimates.
        """
        return self.statistics

class PlainMCIntegrator(Integrator):
    """
    Plain Monte Carlo integrator.
    """
    
    def integrate(self, neval: int) -> Union[float, np.ndarray, Dict]:
        samples = self.get_sample_points(neval)
        values = self.integrand(samples)
        
        ravg = RAvg(values)
        result = ravg.mean()
        error = ravg.error()

        self.statistics.update(result, error, neval)
        return result

    def get_sample_points(self, num_points: int) -> torch.Tensor:
        return torch.rand(num_points, self.ndim, device=self.device)

class MCMCIntegratorWrapper(Integrator):
    def __init__(self,
                 ndim: int,
                 integrand: Callable,
                 prior: Callable,
                 proposal: Callable,
                 bounds: Union[List[Tuple[float, float]], np.ndarray],
                 discrete_dims: Optional[List[int]] = None,
                 use_gpu: bool = False,
                 num_cpus: int = 1):
        super().__init__(ndim, integrand, bounds, use_gpu, num_cpus, discrete_dims)
        self.mcmc_integrator = MCMCIntegrator(ndim, self._wrap_integrand(integrand), prior, proposal, bounds, discrete_dims, use_gpu)

    def integrate(self, nsteps: int, burnin: int = 1000) -> Union[float, np.ndarray, Dict]:
        result = self.mcmc_integrator.integrate(nsteps, burnin)
        error_estimate = self._estimate_error(result)
        self.statistics.update(result, error_estimate, nsteps)
        return from_tensor(result)

    def _estimate_error(self, chain: torch.Tensor) -> torch.Tensor:
        # Use batch means for error estimate
        batch_size = min(100, len(chain) // 64) 
        num_batches = len(chain) // batch_size
        batches = chain[:num_batches * batch_size].reshape(num_batches, batch_size, -1)
        batch_means = torch.mean(batches, dim=1)
        return torch.std(batch_means, dim=0) / np.sqrt(num_batches)

class VegasMCMCIntegratorWrapper(Integrator):
    def __init__(self,
                 ndim: int,
                 integrand: Callable,
                 prior: Callable,
                 bounds: Union[List[Tuple[float, float]], np.ndarray],
                 discrete_dims: Optional[List[int]] = None,
                 use_gpu: bool = False,
                 num_cpus: int = 1):
        super().__init__(ndim, integrand, bounds, use_gpu, num_cpus, discrete_dims)
        self.vegas_mcmc_integrator = VegasMCMCIntegrator(ndim, self._wrap_integrand(integrand), prior, bounds, discrete_dims, use_gpu)

    def integrate(self, nsteps: int, burnin: int = 1000, adapt_iters: int = 5, adapt_evals: int = 10000) -> Union[float, np.ndarray, Dict]:
        self.vegas_mcmc_integrator.adapt(adapt_evals, adapt_iters)
        result = self.vegas_mcmc_integrator.integrate(nsteps, burnin)
        error_estimate = self._estimate_error(result)
        self.statistics.update(result, error_estimate, nsteps)
        return from_tensor(result)

    def _estimate_error(self, chain: torch.Tensor) -> torch.Tensor:
        return MCMCIntegratorWrapper._estimate_error(self, chain)

def integrate(integrand: Callable,
              ndim: int,
              bounds: Union[List[Tuple[float, float]], np.ndarray],
              method: str = 'plain',
              neval: int = 10000,
              use_gpu: bool = False,
              num_cpus: int = 1,
              discrete_dims: Optional[List[int]] = None,
              **kwargs) -> Union[float, np.ndarray, Dict]:
    """
    Convenience function to perform integration using the specified method.

    Args:
        integrand (Callable): The function to integrate. Should accept a tensor of shape (batch_size, ndim).
        ndim (int): Number of dimensions of the integral.
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Integration bounds. Can be a list of tuples or a numpy array of shape (ndim, 2).
        method (str, optional): Integration method to use. Options: 'plain', 'stratified', 'vegas'. Defaults to 'plain'.
        neval (int, optional): Number of function evaluations. Defaults to 10000.
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.
        num_cpus (int, optional): Number of CPU cores to use for parallel computation. Ignored if use_gpu is True. Defaults to 1.
        **kwargs: Additional keyword arguments for the specific integration method.

    Returns:
        Union[float, np.ndarray, Dict]: The estimated integral value. Could be a scalar, array, or dictionary.
    """
    if method == 'plainMC':
        integrator = PlainMCIntegrator(ndim, integrand, bounds, use_gpu, num_cpus)
    elif method == 'mcmc':
        prior = kwargs.get('prior', lambda x: torch.ones_like(x[..., 0]))
        proposal = kwargs.get('proposal', lambda x: x + torch.randn_like(x) * 0.1)
        integrator = MCMCIntegratorWrapper(ndim, integrand, prior, proposal, bounds, discrete_dims, use_gpu, num_cpus)
        return integrator.integrate(neval, kwargs.get('burnin', 1000))
    elif method == 'vegas_mcmc':
        prior = kwargs.get('prior', lambda x: torch.ones_like(x[..., 0]))
        integrator = VegasMCMCIntegratorWrapper(ndim, integrand, prior, bounds, discrete_dims, use_gpu, num_cpus)
        return integrator.integrate(neval, kwargs.get('burnin', 1000), kwargs.get('adapt_iters', 5), kwargs.get('adapt_evals', 10000))
    else:
        raise ValueError(f"Unknown integration method: {method}")