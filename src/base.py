import torch
from torch import nn
import numpy as np


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, bounds, device="cpu"):
        super().__init__()
        # self.bounds = bounds
        if isinstance(bounds, (list, np.ndarray)):
            self.bounds = torch.tensor(bounds, dtype=torch.float64, device=device)
        else:
            raise ValueError("Unsupported map specification")
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

    def __init__(self, bounds, device="cpu"):
        super().__init__(bounds, device)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def sample(self, nsamples=1, **kwargs):
        u = (
            torch.rand((nsamples, self.dim), device=self.device) * self._rangebounds
            + self.bounds[:, 0]
        )
        log_detJ = torch.log(self._rangebounds).sum().repeat(nsamples)
        return u, log_detJ
