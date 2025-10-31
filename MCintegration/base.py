# base.py
# This file contains the base distribution classes for Monte Carlo integration methods
# It defines foundational classes for sampling distributions and transformations

import torch
from torch import nn
import numpy as np
import sys
from MCintegration.utils import get_device

# Constants for numerical stability
EPSILON = 1e-16  # Small value to ensure numerical stability
# EPSILON = sys.float_info.epsilon * 1e4  # Small value to ensure numerical stability


class BaseDistribution(nn.Module):
    """
    Base distribution class for flow-based models.
    This is an abstract base class that provides structure for probability distributions
    used in Monte Carlo integration. Parameters do not depend on target variables
    (unlike a VAE encoder).
    """

    def __init__(self, dim, device="cpu", dtype=torch.float32):
        """
        Initialize BaseDistribution.

        Args:
            dim (int): Dimensionality of the distribution
            device (str or torch.device): Device to use for computation
            dtype (torch.dtype): Data type for computations
        """
        super().__init__()
        self.dtype = dtype
        self.dim = dim
        self.device = device

    def sample(self, batch_size=1, **kwargs):
        """
        Sample from the base distribution.

        Args:
            batch_size (int): Number of samples to draw
            **kwargs: Additional arguments

        Returns:
            tuple: (samples, log_det_jacobian)

        Raises:
            NotImplementedError: This is an abstract method
        """
        raise NotImplementedError

    def sample_with_detJ(self, batch_size=1, **kwargs):
        """
        Sample from base distribution with Jacobian determinant (not log).

        Args:
            batch_size (int): Number of samples to draw
            **kwargs: Additional arguments

        Returns:
            tuple: (samples, det_jacobian)
        """
        u, detJ = self.sample(batch_size, **kwargs)
        detJ.exp_()  # Convert log_det to det
        return u, detJ


class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution over [0,1]^dim.
    Samples from a uniform distribution in the hypercube [0,1]^dim.
    """

    def __init__(self, dim, device="cpu", dtype=torch.float32):
        """
        Initialize Uniform distribution.

        Args:
            dim (int): Dimensionality of the distribution
            device (str or torch.device): Device to use for computation
            dtype (torch.dtype): Data type for computations
        """
        super().__init__(dim, device, dtype)

    def sample(self, batch_size=1, **kwargs):
        """
        Sample from uniform distribution over [0,1]^dim.

        Args:
            batch_size (int): Number of samples to draw
            **kwargs: Additional arguments

        Returns:
            tuple: (uniform samples, log_det_jacobian=0)
        """
        # torch.manual_seed(0) # test seed
        u = torch.rand((batch_size, self.dim), device=self.device, dtype=self.dtype)
        log_detJ = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        return u, log_detJ


class LinearMap(nn.Module):
    """
    Linear transformation map of the form x = u * A + b.
    Maps points from one space to another using a linear transformation.
    """

    def __init__(self, A, b, device=None, dtype=torch.float32):
        """
        Initialize LinearMap with scaling A and offset b.

        Args:
            A (list, numpy.ndarray, torch.Tensor): Scaling factors
            b (list, numpy.ndarray, torch.Tensor): Offset values
            device (str or torch.device): Device to use for computation
            dtype (torch.dtype): Data type for computations
        """
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

        # Pre-compute determinant of Jacobian for efficiency
        self._detJ = torch.prod(self.A)

    def forward(self, u):
        """
        Apply forward transformation: x = u * A + b.

        Args:
            u (torch.Tensor): Input points

        Returns:
            tuple: (transformed points, log_det_jacobian)
        """
        return u * self.A + self.b, torch.log(self._detJ.repeat(u.shape[0]))

    def forward_with_detJ(self, u):
        """
        Apply forward transformation with Jacobian determinant (not log).

        Args:
            u (torch.Tensor): Input points

        Returns:
            tuple: (transformed points, det_jacobian)
        """
        u, detJ = self.forward(u)
        detJ.exp_()  # Convert log_det to det
        return u, detJ

    def inverse(self, x):
        """
        Apply inverse transformation: u = (x - b) / A.

        Args:
            x (torch.Tensor): Input points

        Returns:
            tuple: (transformed points, log_det_jacobian)
        """
        return (x - self.b) / self.A, torch.log(self._detJ.repeat(x.shape[0]))
