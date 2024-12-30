import torch
from torch import nn
import numpy as np
import sys
from MCintegration.utils import get_device

MINVAL = 10 ** (sys.float_info.min_10_exp + 50)
MAXVAL = 10 ** (sys.float_info.max_10_exp - 50)
EPSILON = 1e-16  # Small value to ensure numerical stability


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, dim, device="cpu", dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.dim = dim
        self.device = device

    def sample(self, batch_size=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        raise NotImplementedError

    def sample_with_detJ(self, batch_size=1, **kwargs):
        u, detJ = self.sample(batch_size, **kwargs)
        detJ.exp_()
        return u, detJ


class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, dim, device="cpu", dtype=torch.float32):
        super().__init__(dim, device, dtype)

    def sample(self, batch_size=1, **kwargs):
        # torch.manual_seed(0) # test seed
        u = torch.rand((batch_size, self.dim), device=self.device, dtype=self.dtype)
        log_detJ = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        return u, log_detJ


class LinearMap(nn.Module):
    def __init__(self, A, b, device=None, dtype=torch.float32):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.dtype = dtype

        assert len(A) == len(b), "A and b must have the same dimension."
        if isinstance(A, (list, np.ndarray)):
            self.A = torch.tensor(A, dtype=self.dtype, device=self.device)
        elif isinstance(A, torch.Tensor):
            self.A = A.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError("'A' must be a list, numpy array, or torch tensor.")

        if isinstance(b, (list, np.ndarray)):
            self.b = torch.tensor(b, dtype=self.dtype, device=self.device)
        elif isinstance(b, torch.Tensor):
            self.b = b.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError("'b' must be a list, numpy array, or torch tensor.")

        self._detJ = torch.prod(self.A)

    def forward(self, u):
        return u * self.A + self.b, torch.log(self._detJ.repeat(u.shape[0]))

    def forward_with_detJ(self, u):
        u, detJ = self.forward(u)
        detJ.exp_()
        return u, detJ

    def inverse(self, x):
        return (x - self.b) / self.A, torch.log(self._detJ.repeat(x.shape[0]))
