# Example 1: Unit Circle and Half-Sphere Integrands Comparison

import torch
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas, set_seed, get_device

set_seed(42)
device = get_device()


def unit_circle_integrand(x, f):
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return f[:, 0]


def half_sphere_integrand(x, f):
    f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
    return f[:, 0]


dim = 2
bounds = [(-1, 1), (-1, 1)]
n_eval = 6400000
batch_size = 10000
n_therm = 20

vegas_map = Vegas(dim, device=device, ninc=10)

# Monte Carlo and MCMC for Unit Circle
mc_integrator = MonteCarlo(
    f=unit_circle_integrand, bounds=bounds, batch_size=batch_size
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=unit_circle_integrand, bounds=bounds, batch_size=batch_size, nburnin=n_therm
)

print("Unit Circle Integration Results:")
print("Plain MC:", mc_integrator(n_eval))
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

# Train VEGAS map for Unit Circle
vegas_map.adaptive_training(batch_size, unit_circle_integrand, alpha=0.5)
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

# Monte Carlo and MCMC for Half-Sphere
mc_integrator.f = half_sphere_integrand
mcmc_integrator.f = half_sphere_integrand

print("\nHalf-Sphere Integration Results:")
print("Plain MC:", mc_integrator(n_eval))
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

vegas_map.make_uniform()
vegas_map.adaptive_training(batch_size, half_sphere_integrand, epoch=10, alpha=0.5)
vegas_integrator.f = half_sphere_integrand
vegasmcmc_integrator.f = half_sphere_integrand

print("VEGAS:", vegas_integrator(n_eval))
print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))
