import torch
from torch import nn
import numpy as np
import sys

MINVAL = 10 ** (sys.float_info.min_10_exp + 50)
MAXVAL = 10 ** (sys.float_info.max_10_exp - 50)
EPSILON = 1e-16


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, bounds, device="cpu", dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        if isinstance(bounds, (list, np.ndarray)):
            self.bounds = torch.tensor(bounds, dtype=dtype, device=device)
        elif isinstance(bounds, torch.Tensor):
            self.bounds = bounds.to(dtype=dtype, device=device)
        else:
            raise ValueError("'bounds' must be a list, numpy array, or torch tensor.")

        self.dim = self.bounds.shape[0]
        self.device = device

    def sample(self, nsamples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        raise NotImplementedError


class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, bounds, device="cpu", dtype=torch.float64):
        super().__init__(bounds, device, dtype)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def sample(self, nsamples=1, **kwargs):
        # torch.manual_seed(0) # test seed
        u = (
            torch.rand((nsamples, self.dim), device=self.device, dtype=self.dtype)
            * self._rangebounds
            + self.bounds[:, 0]
        )
        log_detJ = torch.log(self._rangebounds).sum().repeat(nsamples)
        return u, log_detJ


class Linear(nn.Module):
    def __init__(self, bounds, device="cpu", dtype=torch.float64):
        super().__init__()
        if isinstance(bounds, (list, np.ndarray)):
            self.bounds = torch.tensor(bounds, dtype=dtype, device=device)
        elif isinstance(bounds, torch.Tensor):
            self.bounds = bounds.to(dtype=dtype, device=device)
        else:
            raise ValueError("'bounds' must be a list, numpy array, or torch tensor.")

        self.dim = self.bounds.shape[0]
        self.device = device
        self.dtype = dtype
        self._A = self.bounds[:, 1] - self.bounds[:, 0]
        self._jac1 = torch.prod(self._A)

    def forward(self, u):
        return u * self._A + self.bounds[:, 0], torch.log(self._jac1.repeat(u.shape[0]))

    def inverse(self, x):
        return (x - self.bounds[:, 0]) / self._A, torch.log(
            self._jac1.repeat(x.shape[0])
        )
