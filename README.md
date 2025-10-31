# MCintegration: Monte Carlo Integration in Python
[![alpha](https://img.shields.io/badge/docs-alpha-blue.svg)](https://numericaleft.github.io/MCintegration.py/)
[![Build Status](https://github.com/numericalEFT/MCIntegration.py/workflows/CI/badge.svg)](https://github.com/numericalEFT/MCIntegration.py/actions)
[![codecov](https://codecov.io/gh/numericalEFT/MCintegration.py/graph/badge.svg?token=851N2CNOTN)](https://codecov.io/gh/numericalEFT/MCintegration.py)

## Why Choose MCintegration?

MCintegration is a comprehensive Python package designed to handle both regular and singular high-dimensional integrals with ease. Its implementation of robust Monte Carlo integration methods makes it a versatile tool in various scientific domains, including high-energy physics, material science, computational chemistry, financial mathematics, and machine learning.

The package leverages **PyTorch** for efficient tensor computations, enabling:
- **GPU acceleration** for massive performance gains on compatible hardware
- **Batched computations** for efficient parallel processing
- **Multi-CPU parallelization** for distributed workloads

Whether you're dealing with simple low-dimensional integrals or complex high-dimensional integrals with sharp peaks and singularities, MCintegration provides the tools you need with a simple, intuitive API.

## Installation

Install via pip:

```bash
pip install mcintegration
```

Or install from source:

```bash
git clone https://github.com/your-repo/mcintegration
cd mcintegration
python setup.py install
```

## Requirements

- Python 3.7+
- NumPy
- PyTorch 
- gvar

## Quick Start

MCintegration simplifies complex integral calculations. Here are two examples to get you started.

### Example: Estimating π

Let's estimate π by computing the area of a unit circle:

```python
import torch
from MCintegration import MonteCarlo

# Define integrand: indicator function for unit circle
def circle(x, f):
    r2 = x[:, 0]**2 + x[:, 1]**2
    f[:, 0] = (r2 <= 1).float()
    return f.mean(dim=-1)

# Integrate over [-1,1] × [-1,1]
mc = MonteCarlo(f=circle, bounds=[(-1, 1), (-1, 1)], batch_size=10000)
result = mc(1000000)
pi_estimate = result * 4  # Circle area / square area * 4
print(f"π ≈ {pi_estimate}")
```

## Understanding the Integrand Function

All integrand functions in MCintegration follow a specific signature:

```python
def integrand(x, f) -> result:
    """
    Args:
        x: PyTorch tensor of shape (batch_size, dim) - input sample points
        f: PyTorch tensor of shape (batch_size, f_dim) - output buffer
    
    Returns:
        Reduced result, typically f.mean(dim=-1) for single output
        or f for multiple outputs
    """
    # Your computation here
    f[:, 0] = ...  # Store result(s)
    return f.mean(dim=-1)  # or return f for multiple outputs
```

**Key points:**
- **x**: Contains batches of random points sampled from the integration domain
- **f**: Pre-allocated tensor where you store function values
- **Return value**: Usually `f.mean(dim=-1)` for variance reduction with single outputs, or `f` directly for multiple outputs
- **Batched computation**: All operations should be vectorized using PyTorch for efficiency

## Core Integration Methods

MCintegration provides several Monte Carlo integration algorithms:

### 1. Standard Monte Carlo (`MonteCarlo`)

The classic Monte Carlo algorithm uses uniform random sampling. It's efficient for low-dimensional integrals and well-behaved functions.

```python
from MCintegration import MonteCarlo

mc = MonteCarlo(
    f=integrand_function,
    bounds=[(min, max), ...],
    batch_size=10000,
    f_dim=1  # Number of output dimensions
)
result = mc(n_eval)  # n_eval total function evaluations
```


### 2. Markov Chain Monte Carlo (`MarkovChainMonteCarlo`)

MCMC uses the Metropolis-Hastings algorithm to generate correlated samples. This can be more efficient for certain types of integrands.

```python
from MCintegration import MarkovChainMonteCarlo

mcmc = MarkovChainMonteCarlo(
    f=integrand_function,
    bounds=[(min, max), ...],
    batch_size=10000,
    nburnin=100,  # Thermalization/burn-in steps
    f_dim=1
)
result = mcmc(n_eval)
```


**Important parameters:**
- **nburnin**: Number of initial steps to discard while the chain reaches equilibrium
- Higher nburnin values may be needed for complex distributions

### 3. VEGAS Algorithm with Adaptive Importance Sampling

VEGAS uses **adaptive importance sampling** to concentrate samples in regions where the integrand is large, dramatically improving accuracy for difficult integrals.

```python
from MCintegration import Vegas, MonteCarlo

# Create and train the VEGAS map
vegas_map = Vegas(dim, ninc=1000)
vegas_map.adaptive_training(batch_size, integrand_function, f_dim=1)

# Use the adapted map with any integrator
mc = MonteCarlo(
    f=integrand_function,
    bounds=bounds,
    batch_size=batch_size,
    maps=vegas_map
)
result = mc(n_eval)
```

**When to use VEGAS:**
- Functions with singularities or near-singularities
- Sharply peaked functions (e.g., Gaussians with small width)
- Functions where most contribution comes from small regions

**VEGAS parameters:**
- **ninc**: Number of increments in the adaptive grid (more = finer adaptation)
- Higher values allow better adaptation but increase memory usage
- Typical values: 100-1000 for low dimensions, 50-100 for high dimensions

## Detailed Examples

### Example1: Singular Function Integration

Handle a function with a singularity at x=0 using VEGAS adaptive sampling.

```python
import torch
from MCintegration import Vegas, MonteCarlo, MarkovChainMonteCarlo

# Define singular function: log(x)/√x
def singular_func(x, f):
    """
    Integrand with singularity at x=0.
    The integral ∫₀¹ log(x)/√x dx = -4 (analytical result)
    """
    x_safe = torch.clamp(x[:, 0], min=1e-32)
    f[:, 0] = torch.log(x_safe) / torch.sqrt(x_safe)
    return f[:, 0]

# Integration parameters
dim = 1
bounds = [(0, 1)]          # Note: includes singular point x=0
n_eval = 6400000
batch_size = 10000
n_therm = 100

# VEGAS adaptive training
vegas_map = Vegas(dim, ninc=1000)
vegas_map.adaptive_training(batch_size, singular_func)

# Standard MC with VEGAS map
vegas_mc = MonteCarlo(
    f=singular_func, 
    bounds=bounds, 
    batch_size=batch_size, 
    maps=vegas_map
)
result_vegas = vegas_mc(n_eval)
print(f"VEGAS-MC: {result_vegas} (expected: -4)")

# MCMC with VEGAS map
vegas_mcmc = MarkovChainMonteCarlo(
    f=singular_func, 
    bounds=bounds, 
    batch_size=batch_size, 
    nburnin=n_therm, 
    maps=vegas_map
)
result_vegas_mcmc = vegas_mcmc(n_eval)
print(f"VEGAS-MCMC: {result_vegas_mcmc} (expected: -4)")
```

**Why VEGAS helps:**
- Standard MC samples uniformly, missing the singularity structure
- VEGAS learns to concentrate samples near x=0 where |log(x)/√x| is large
- This dramatically reduces the variance and improves accuracy

**The VEGAS workflow:**
1. **Create map**: `Vegas(dim, ninc)` initializes the adaptive grid
2. **Train map**: `adaptive_training()` learns the integrand structure
3. **Use map**: Pass `maps=vegas_map` to sample from adapted distribution

---

### Example 2: Sharp Gaussian Peak in High Dimensions

Compute multiple related quantities (normalization and moments) for a sharp 4D Gaussian.

```python
import torch
from MCintegration import Vegas, MonteCarlo, MarkovChainMonteCarlo

# Define sharp Gaussian and its moments
def sharp_integrands(x, f):
    """
    Computes three related integrals:
    1. Normalization: ∫ exp(-200·|x-0.5|²) dx
    2. First moment: ∫ x₀·exp(-200·|x-0.5|²) dx
    3. Second moment: ∫ x₀²·exp(-200·|x-0.5|²) dx
    
    The peak is centered at (0.5, 0.5, 0.5, 0.5) with width σ ≈ 0.05
    """
    # Squared distance from center
    dist2 = torch.sum((x - 0.5) ** 2, dim=-1)
    
    # Sharp Gaussian: exp(-200·dist²)
    gaussian = torch.exp(-200 * dist2)
    
    # Store three outputs
    f[:, 0] = gaussian                    # Normalization
    f[:, 1] = gaussian * x[:, 0]          # First moment
    f[:, 2] = gaussian * x[:, 0] ** 2     # Second moment
    
    return f.mean(dim=-1)

# Integration parameters
dim = 4                    # 4D integration
bounds = [(0, 1)] * dim    # Unit hypercube [0,1]⁴
n_eval = 6400000
batch_size = 10000
n_therm = 100

# VEGAS adaptive training for all three outputs
vegas_map = Vegas(dim, ninc=1000)
vegas_map.adaptive_training(batch_size, sharp_integrands, f_dim=3)

# Integration with VEGAS map
vegas_mc = MonteCarlo(
    f=sharp_integrands, 
    f_dim=3,              # Three simultaneous outputs
    bounds=bounds, 
    batch_size=batch_size, 
    maps=vegas_map
)
result_vegas = vegas_mc(n_eval)
print(f"Results: {result_vegas}")
print(f"  Normalization: {result_vegas[0]}")
print(f"  First moment: {result_vegas[1]}")
print(f"  Second moment: {result_vegas[2]}")

# MCMC alternative
vegas_mcmc = MarkovChainMonteCarlo(
    f=sharp_integrands, 
    f_dim=3,
    bounds=bounds, 
    batch_size=batch_size, 
    nburnin=n_therm, 
    maps=vegas_map
)
result_vegas_mcmc = vegas_mcmc(n_eval)
print(f"MCMC Results: {result_vegas_mcmc}")
```

**Multi-output integration:**
- **f_dim=3**: Tells the integrator to expect 3 outputs
- All three integrals use the **same sample points**, improving efficiency
- The samples are correlated, which is beneficial for computing related quantities
- Must specify `f_dim` in both `adaptive_training()` and integrator initialization

**High-dimensional integration:**
As dimension increases:
- Volume grows exponentially (curse of dimensionality)
- Uniform sampling becomes exponentially inefficient
- Sharp features become impossible to capture without adaptation
- VEGAS becomes essential for accurate results

---

## Citation

If you use MCintegration in your research, please cite:

```bibtex
@software{mcintegration,
  title = {MCintegration: Monte Carlo Integration in Python},
  author = {[Pengcheng Hou, Tao Wang, Caiyu Fan, Kun Chen]},
  year = {2024},
  url = {https://github.com/numericalEFT/MCintegration.py}
}
```

## Acknowledgements and Related Packages

The development of `MCintegration.py` has been greatly inspired and influenced by several significant works in the field of numerical integration. We would like to express our appreciation to the following:

- **[vegas](https://github.com/gplepage/vegas)**: A Python package offering Monte Carlo estimations of multidimensional integrals, with notable improvements on the original Vegas algorithm. It's been a valuable reference for us. Learn more from the vegas [documentation](https://vegas.readthedocs.io/). **Reference: G. P. Lepage, J. Comput. Phys. 27, 192 (1978) and G. P. Lepage, J. Comput. Phys. 439, 110386 (2021) [arXiv:2009.05112](https://arxiv.org/abs/2009.05112)**.

- **[MCIntegration.jl](https://github.com/numericalEFT/MCIntegration.jl)**: A comprehensive Julia package for Monte Carlo integration that has influenced our API design and algorithm choices. See their [documentation](https://numericaleft.github.io/MCIntegration.jl/dev/) for the Julia implementation.

- **[Cuba](https://feynarts.de/cuba/)** and **[Cuba.jl](https://github.com/giordano/Cuba.jl)**: The Cuba library offers numerous Monte Carlo algorithms for multidimensional numerical integration. **Reference: T. Hahn, Comput. Phys. Commun. 168, 78 (2005) [arXiv:hep-ph/0404043](https://arxiv.org/abs/hep-ph/0404043)**.

These groundbreaking efforts have paved the way for our project. We extend our deepest thanks to their creators and maintainers.
