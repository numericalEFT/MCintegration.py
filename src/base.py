import torch
from torch import nn
import numpy as np
import sys

MINVAL = 10 ** (sys.float_info.min_10_exp + 50)
MAXVAL = 10 ** (sys.float_info.max_10_exp - 50)
EPSILON = 1e-16  # Small value to ensure numerical stability


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, dim, device="cpu", dtype=torch.float64):
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

    def __init__(self, dim, device="cpu", dtype=torch.float64):
        super().__init__(dim, device, dtype)

    def sample(self, batch_size=1, **kwargs):
        # torch.manual_seed(0) # test seed
        u = torch.rand((batch_size, self.dim), device=self.device, dtype=self.dtype)
        log_detJ = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        return u, log_detJ
