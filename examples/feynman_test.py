from feynmandiag import init_feynfunc
import torch
from MCintegration import (
    MonteCarlo,
    MarkovChainMonteCarlo,
    Vegas,
    set_seed,
    get_device,
)
import time

device = get_device()
batch_size = 5000
n_eval = 1000000
n_therm = 10

num_roots = [1, 2, 3, 4, 5, 6]
order = 2
beta = 10.0
feynfunc = init_feynfunc(order, beta, batch_size)
feynfunc.to(device)
f_dim = num_roots[order]


vegas_map = Vegas(feynfunc.ndims, ninc=1000, device=device)
# vegas_map = Vegas(feynfunc.ndims, ninc=1000, device=torch.device("cpu"))
print("Training the vegas map...")
# feynfunc.to(torch.device("cpu"))
begin_time = time.time()
vegas_map.adaptive_training(batch_size, feynfunc, f_dim=f_dim, epoch=10, alpha=1.0)
print("training time: ", time.time() - begin_time, "s\n")

begin_time = time.time()
bounds = [[0, 1]] * feynfunc.ndims

mc_integrator = MonteCarlo(
    bounds, feynfunc, f_dim=f_dim, batch_size=batch_size, device=device
)

res = mc_integrator(neval=n_eval, mix_rate=0.5)
print("Plain MC Integral results: ", res)

mcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    feynfunc,
    f_dim=f_dim,
    batch_size=batch_size,
    nburnin=n_therm,
    device=device,
)
res = mcmc_integrator(neval=n_eval)
print("MCMC Integral results: ", res)


vegas_integrator = MonteCarlo(
    bounds,
    feynfunc,
    f_dim=f_dim,
    maps=vegas_map,
    batch_size=batch_size,
    device=device,
)
res = vegas_integrator(neval=n_eval, mix_rate=0.5)
print("VEGAS Integral results: ", res)


vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    feynfunc,
    f_dim=f_dim,
    batch_size=batch_size,
    nburnin=n_therm,
    maps=vegas_map,
    device=device,
)
res = vegasmcmc_integrator(neval=n_eval, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)

print("Total time: ", time.time() - begin_time, "s")
