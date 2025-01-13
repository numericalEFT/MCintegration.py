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

    def __init__(self, dim, batch_size, device="cpu", dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self.register_buffer(
            "u", torch.empty((batch_size, dim), device=device, dtype=dtype)
        )
        self.register_buffer(
            "log_detJ", torch.empty(batch_size, device=device, dtype=dtype)
        )

    def sample(self, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        raise NotImplementedError

    def sample_with_detJ(self, **kwargs):
        self.sample(**kwargs)
        self.log_detJ.exp_()
        return self.u, self.log_detJ


class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, dim, batch_size, device="cpu", dtype=torch.float32):
        super().__init__(dim, batch_size, device, dtype)

    def sample(self, **kwargs):
        # torch.manual_seed(0) # test seed
        self.u.uniform_()
        self.log_detJ.zero_()
        return self.u, self.log_detJ


class LinearMap(nn.Module):
    def __init__(self, dim, batch_size, A, b, device=None, dtype=torch.float32):
        super().__init__()
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
        self.register_buffer(
            "u", torch.empty((batch_size, dim), device=device, dtype=dtype)
        )
        self.register_buffer("log_detJ", torch.log(self._detJ.repeat(batch_size)))
        self.register_buffer("detJ", self._detJ.repeat(batch_size))

    def forward(self, u):
        self.u = u * self.A + self.b
        return self.u, self.log_detJ

    def forward_with_detJ(self, u):
        self.forward(u)
        # detJ.exp_()
        return self.u, self.detJ

    def inverse(self, x):
        self.u = (x - self.b) / self.A
        return self.u, self.log_detJ
