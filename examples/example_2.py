# Example 2: Sharp Peak Integrand in Higher Dimensions

import torch
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas, set_seed, get_device

set_seed(42)
device = get_device()


def sharp_integrands(x, f):
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
    f[:, 0] *= -200
    f[:, 0].exp_()
    f[:, 1] = f[:, 0] * x[:, 0]
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f.mean(dim=-1)


dim = 4
bounds = [(0, 1)] * dim
n_eval = 500000
batch_size = 10000
n_therm = 20

vegas_map = Vegas(dim, device=device, ninc=1000)

# Plain MC and MCMC
mc_integrator = MonteCarlo(
    f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size, nburnin=n_therm
)

print("Sharp Peak Integration Results:")
print("Plain MC:", mc_integrator(n_eval))
print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

# Train VEGAS map
vegas_map.adaptive_training(batch_size, sharp_integrands, f_dim=3, epoch=10, alpha=2.0)
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
