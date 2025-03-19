# Example 2: Sharp Peak Integrand in Higher Dimensions
#
# This example demonstrates the effectiveness of different Monte Carlo integration methods
# for a challenging integrand with a sharp peak in a higher-dimensional space (4D).
# The integrand has a Gaussian peak centered at (0.5, 0.5, 0.5, 0.5) with a width of
# sqrt(1/200), making it challenging for uniform sampling methods.
#
# The example compares:
# - Plain Monte Carlo integration
# - Markov Chain Monte Carlo (MCMC)
# - VEGAS algorithm (adaptive importance sampling)
# - VEGAS with MCMC
#
# The integrand contains three components (f_dim=3) to demonstrate multi-dimensional outputs.

import torch
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas, set_seed, get_device

# Set random seed for reproducibility and get computation device
set_seed(42)
device = get_device()


def sharp_integrands(x, f):
    """
    A function with a sharp Gaussian peak centered at (0.5, 0.5, 0.5, 0.5).
    The function returns three components:
    1. exp(-200*sum((x_i - 0.5)²))
    2. x_0 * exp(-200*sum((x_i - 0.5)²))
    3. x_0² * exp(-200*sum((x_i - 0.5)²))

    Args:
        x (torch.Tensor): Batch of points in [0,1]⁴
        f (torch.Tensor): Output tensor to store function values

    Returns:
        torch.Tensor: Mean of the three components for each point
    """
    # Compute distance from the center point (0.5, 0.5, 0.5, 0.5)
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
    # Scale by -200 to create a sharp peak
    f[:, 0] *= -200
    # Apply exponential to get Gaussian function
    f[:, 0].exp_()
    # Second component: first coordinate times the Gaussian
    f[:, 1] = f[:, 0] * x[:, 0]
    # Third component: square of first coordinate times the Gaussian
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f.mean(dim=-1)


# Set parameters
dim = 4  # 4D integration
bounds = [(0, 1)] * dim  # Integration bounds (unit hypercube)
n_eval = 500000  # Number of function evaluations
batch_size = 10000  # Batch size for sampling
n_therm = 20  # Number of thermalization steps for MCMC

# Initialize VEGAS map with finer grid for better adaptation to the sharp peak
vegas_map = Vegas(dim, device=device, ninc=1000)

# Initialize plain MC and MCMC integrators
# Note that f_dim=3 to handle the three components of the integrand
mc_integrator = MonteCarlo(
    f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size, nburnin=n_therm
)

print("Sharp Peak Integration Results:")
print("Plain MC:", mc_integrator(n_eval))
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

# Train VEGAS map to adapt to the sharp peak
# Using alpha=2.0 for more aggressive grid adaptation
vegas_map.adaptive_training(
    batch_size, sharp_integrands, f_dim=3, epoch=10, alpha=2.0)

# Initialize integrators that use the trained VEGAS map
vegas_integrator = MonteCarlo(
    bounds, f=sharp_integrands, f_dim=3, maps=vegas_map, batch_size=batch_size
)
vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    f=sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)

print("VEGAS:", vegas_integrator(n_eval))
print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))

# Expected outcome: VEGAS should significantly outperform plain MC
# since it adapts the sampling grid to concentrate points near the peak.
