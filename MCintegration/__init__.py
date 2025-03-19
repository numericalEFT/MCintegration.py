# MCintegration/__init__.py
#
# Monte Carlo Integration Package
#
# This package provides tools for numerical integration using various Monte Carlo methods.
# It includes plain Monte Carlo, Markov Chain Monte Carlo (MCMC), and VEGAS algorithms
# for efficient multi-dimensional integration.
#
# The package provides:
#   - Base distributions for sampling
#   - Transformation maps for importance sampling
#   - Various integration algorithms
#   - Utilities for running averages and error estimation
#   - Multi-CPU/GPU support through torch.distributed
#

from .integrators import MonteCarlo, MarkovChainMonteCarlo, setup
from .maps import Vegas
from .utils import RAvg, set_seed, get_device
