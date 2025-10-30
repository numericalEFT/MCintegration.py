# MCintegration
[![alpha](https://img.shields.io/badge/docs-alpha-blue.svg)](https://numericaleft.github.io/MCintegration.py/)
[![Build Status](https://github.com/numericalEFT/MCIntegration.py/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.py/actions)
[![codecov](https://codecov.io/gh/numericalEFT/MCintegration.py/graph/badge.svg?token=851N2CNOTN)](https://codecov.io/gh/numericalEFT/MCintegration.py)
A Python library for Monte Carlo integration with support for multi-CPU and GPU computations.

## Overview

MCintegration is a specialized library designed for numerical integration using Monte Carlo methods. It provides efficient implementations of various integration algorithms with focus on applications in computational physics and effective field theories (EFT).

The library offers:
- Multiple Monte Carlo integration algorithms
- Support for multi-CPU parallelization
- GPU acceleration capabilities
- Integration with PyTorch for tensor-based computations

## Installation

```bash
pip install mcintegration
```

Or install from source:

```bash
python setup.py install
```

## Usage

### Example 1: Unit Circle Integration

This example demonstrates different Monte Carlo methods for integrating functions over [-1,1]×[-1,1]:

```python
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas
import torch

# Define integrand function
def unit_circle(x, f):
    r2 = x[:, 0]**2 + x[:, 1]**2
    f[:, 0] = (r2 <= 1).float()
    return f.mean(dim=-1)

# Set up integration parameters
dim = 2
bounds = [(-1, 1)] * dim
n_eval = 6400000
batch_size = 10000
n_therm = 100

# Create integrator instances
mc = MonteCarlo(f=unit_circle, bounds=bounds, batch_size=batch_size)
mcmc = MarkovChainMonteCarlo(f=unit_circle, bounds=bounds, batch_size=batch_size, nburnin=n_therm)

# Perform integration
result_mc = mc(n_eval)
result_mcmc = mcmc(n_eval)
```

### Example 2: Singular Function Integration

This example shows integration of a function with a singularity at x=0:

```python
# Integrate log(x)/sqrt(x) which has a singularity at x=0
def singular_func(x, f):
    f[:, 0] = torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
    return f[:, 0]

# Set up integration parameters
dim = 1
bounds = [(0, 1)]
n_eval = 6400000
batch_size = 10000
n_therm = 100

# Use VEGAS algorithm which adapts to the singular structure
vegas_map = Vegas(dim, ninc=1000)
vegas_map.adaptive_training(batch_size, singular_func)

# Create integrator instances using the adapted vegas map
vegas_mc = MonteCarlo(f=singular_func, bounds=bounds, batch_size=batch_size, maps=vegas_map)
vegas_mcmc = MarkovChainMonteCarlo(f=singular_func, bounds=bounds, batch_size=batch_size, nburnin=n_therm, maps=vegas_map)

# Perform integration
result_vegas = vegas_mc(n_eval)
result_vegas_mcmc = vegas_mcmc(n_eval)
```

### Example 3: Multiple Sharp Peak Integrands in Higher Dimensions

This example demonstrates integration of a sharp Gaussian peak and its moments in 4D space:

```python
# Define a sharp peak and its moments integrands
# This represents a Gaussian peak centered at (0.5, 0.5, 0.5, 0.5)
def sharp_integrands(x, f):
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)  # Distance from center
    f[:, 0] *= -200                           # Scale by width parameter
    f[:, 0].exp_()                            # Exponentiate to create Gaussian
    f[:, 1] = f[:, 0] * x[:, 0]                # First moment
    f[:, 2] = f[:, 0] * x[:, 0] ** 2          # Second moment
    return f.mean(dim=-1)

# Set up 4D integration with sharp peak
dim = 4
bounds = [(0, 1)] * dim
n_eval = 6400000
batch_size = 10000
n_therm = 100

# Use VEGAS algorithm which adapts to the peak structure
vegas_map = Vegas(dim, ninc=1000)
vegas_map.adaptive_training(batch_size, sharp_integrands, f_dim=3)

# Create integrator instances using the adapted vegas map
vegas_mc = MonteCarlo(f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size, maps=vegas_map)
vegas_mcmc = MarkovChainMonteCarlo(f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size, nburnin=n_therm, maps=vegas_map)

# Perform integration
result_vegas = vegas_mc(n_eval)
result_vegas_mcmc = vegas_mcmc(n_eval)
```

## Features

- **Base integration methods**: Core Monte Carlo algorithms in `MCintegration/base.py`
- **Integrator implementations**: Various MC integration strategies in `MCintegration/integrators.py`
- **Variable transformations**: Coordinate mapping utilities in `MCintegration/maps.py`
- **Utility functions**: Helper functions for numerical computations in `MCintegration/utils.py`
- **Multi-CPU support**: Parallel processing capabilities demonstrated in `MCintegration/mc_multicpu_test.py`
- **GPU acceleration**: CUDA-enabled functions through PyTorch in the examples directory


## Requirements

- Python 3.7+
- NumPy
- PyTorch 
- gvar

## Acknowledgements and Related Packages
The development of `MCIntegration.py` has been greatly inspired and influenced by `vegas` package. We would like to express our appreciation to the following:
- [vegas](https://github.com/gplepage/vegas) A Python package offering Monte Carlo estimations of multidimensional integrals, with notable improvements on the original Vegas algorithm. It's been a valuable reference for us. Learn more from the vegas [documentation](https://vegas.readthedocs.io/). **Reference: G. P. Lepage, J. Comput. Phys. 27, 192 (1978) and G. P. Lepage, J. Comput. Phys. 439, 110386 (2021) [arXiv:2009.05112](https://arxiv.org/abs/2009.05112)**. 