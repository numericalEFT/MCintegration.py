# Example 1: Unit Circle and Half-Sphere Integrands Comparison
#
# This example demonstrates how different Monte Carlo integration methods perform
# with two different integrand functions:
# 1. Unit Circle: Integrating the indicator function of a unit circle (area = π)
# 2. Half-Sphere: Integrating a function representing a half-sphere (volume = 2π/3)
#
# The example compares:
# - Plain Monte Carlo integration
# - Markov Chain Monte Carlo (MCMC)
# - VEGAS algorithm (adaptive importance sampling)
# - VEGAS with MCMC
#
# Both integrands are defined over the square [-1,1]×[-1,1]

import torch
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas, set_seed, get_device

# Set random seed for reproducibility and get computation device
set_seed(42)
device = get_device()


def unit_circle_integrand(x, f):
    """
    Indicator function for the unit circle: 1 if point is inside the circle, 0 otherwise.
    The true integral value is π (the area of the unit circle).

    Args:
        x (torch.Tensor): Batch of points in [-1,1]×[-1,1]
        f (torch.Tensor): Output tensor to store function values

    Returns:
        torch.Tensor: Indicator values (0 or 1) for each point
    """
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return f[:, 0]


def half_sphere_integrand(x, f):
    """
    Function representing a half-sphere with radius 1.
    The true integral value is 2π/3 (the volume of the half-sphere).

    Args:
        x (torch.Tensor): Batch of points in [-1,1]×[-1,1]
        f (torch.Tensor): Output tensor to store function values

    Returns:
        torch.Tensor: Height of the half-sphere at each point
    """
    f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
    return f[:, 0]


# Set parameters
dim = 2  # 2D integration
bounds = [(-1, 1), (-1, 1)]  # Integration bounds
n_eval = 6400000  # Number of function evaluations
batch_size = 10000  # Batch size for sampling
n_therm = 20  # Number of thermalization steps for MCMC

# Initialize VEGAS map for adaptive importance sampling
vegas_map = Vegas(dim, device=device, ninc=10)

# PART 1: Unit Circle Integration

# Initialize MC and MCMC integrators for the unit circle
mc_integrator = MonteCarlo(
    f=unit_circle_integrand, bounds=bounds, batch_size=batch_size
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=unit_circle_integrand, bounds=bounds, batch_size=batch_size, nburnin=n_therm
)

print("Unit Circle Integration Results:")
print("Plain MC:", mc_integrator(n_eval))  # True value: π ≈ 3.14159...
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

# Train VEGAS map specifically for the unit circle integrand
vegas_map.adaptive_training(batch_size, unit_circle_integrand, alpha=0.5)

# Initialize integrators that use the trained VEGAS map
vegas_integrator = MonteCarlo(
    bounds, f=unit_circle_integrand, maps=vegas_map, batch_size=batch_size
)
vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    f=unit_circle_integrand,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)

print("VEGAS:", vegas_integrator(n_eval))
print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))

# PART 2: Half-Sphere Integration

# Reuse the same integrators but with the half-sphere integrand
mc_integrator.f = half_sphere_integrand
mcmc_integrator.f = half_sphere_integrand

print("\nHalf-Sphere Integration Results:")
print("Plain MC:", mc_integrator(n_eval))  # True value: 2π/3 ≈ 2.09440...
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

# Reset and retrain the VEGAS map for the half-sphere integrand
vegas_map.make_uniform()
vegas_map.adaptive_training(
    batch_size, half_sphere_integrand, epoch=10, alpha=0.5)

# Update the integrators to use the half-sphere integrand
vegas_integrator.f = half_sphere_integrand
vegasmcmc_integrator.f = half_sphere_integrand

print("VEGAS:", vegas_integrator(n_eval))
print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))
