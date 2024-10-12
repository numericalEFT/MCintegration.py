import torch
from torch import nn


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, bounds, device="cpu"):
        super().__init__()
        self.bounds = bounds
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

    def sample(self, nsamples=1, **kwargs):
        dim = len(self.bounds)
        u = torch.rand((nsamples, dim), device=self.device)
        log_detJ = torch.zeros(nsamples, device=self.device)
        for i, bound in enumerate(self.bounds):
            u[:, i] = (bound[1] - bound[0]) * u[:, i] + bound[0]
            log_detJ += -torch.log(torch.tensor(bound[1] - bound[0]))
        return u, log_detJ
