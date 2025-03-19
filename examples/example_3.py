# Example 3: Integration of log(x)/sqrt(x) using VEGAS

import torch
from MCintegration import MonteCarlo, MarkovChainMonteCarlo
from MCintegration import Vegas, set_seed, get_device

set_seed(42)
device = get_device()


def func(x, f):
    f[:, 0] = torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
    return f[:, 0]


dim = 1
bounds = [[0, 1]] * dim
n_eval = 500000
batch_size = 10000
alpha = 2.0
ninc = 1000
n_therm = 20

vegas_map = Vegas(dim, device=device, ninc=ninc)

# Train VEGAS map
print("Training VEGAS map for log(x)/sqrt(x)... \n")
vegas_map.adaptive_training(batch_size, func, epoch=10, alpha=alpha)

print("Integration Results for log(x)/sqrt(x):")


# Plain MC Integration
mc_integrator = MonteCarlo(bounds, func, batch_size=batch_size)
print("Plain MC Integral Result:", mc_integrator(n_eval))

# MCMC Integration
mcmc_integrator = MarkovChainMonteCarlo(
    bounds, func, batch_size=batch_size, nburnin=n_therm
)
print("MCMC Integral Result:", mcmc_integrator(n_eval, mix_rate=0.5))

# Perform VEGAS integration
vegas_integrator = MonteCarlo(bounds, func, maps=vegas_map, batch_size=batch_size)
res = vegas_integrator(n_eval)

print("VEGAS Integral Result:", res)

# VEGAS-MCMC Integration
vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds, func, maps=vegas_map, batch_size=batch_size, nburnin=n_therm
)
res_vegasmcmc = vegasmcmc_integrator(n_eval, mix_rate=0.5)
print("VEGAS-MCMC Integral Result:", res_vegasmcmc)
