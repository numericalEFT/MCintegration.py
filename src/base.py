import torch
from torch import nn
class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
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

    def __init__(self, bounds):
        super().__init__(bounds)
