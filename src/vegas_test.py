# Integration tests for VEGAS + MonteCarlo/MCMC integral methods.
import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas
from utils import set_seed, get_device

set_seed(42)
device = get_device()
# device = torch.device("cpu")


def sharp_peak(x, f, dim=4):
    f.zero_()
    for d in range(dim):
        f[:, 0] += (x[:, d] - 0.5) ** 2
    f[:, 0] *= -200
    f[:, 0].exp_()
    return f[:, 0]


def sharp_integrands(x, f, dim=4):
    f.zero_()
    for d in range(dim):
        f[:, 0] += (x[:, d] - 0.5) ** 2
    f[:, 0] *= -200
    f[:, 0].exp_()
    f[:, 1] = f[:, 0] * x[:, 0]
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f.mean(dim=-1)


def func(x, f):
    f[:, 0] = torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
    return f[:, 0]


ninc = 1000
nsamples_train = 40000
n_eval = 50000
n_batch = 10000
n_therm = 10

print("\nCalculate the integral log(x)/x^0.5 in the bounds [0, 1]")

print("train VEGAS map by importance sampling data")
vegas_map = Vegas([(0, 1)], device=device, ninc=ninc)
vegas_map.train(nsamples_train, func, epoch=10, alpha=1.0)

vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=1000000,
    nbatch=n_batch,
    device=device,
)
res = vegas_integrator(func)
print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=1000000,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(func, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)

vegas_map.make_uniform()
print("\ntrain VEGAS map by MCMC sampling data")
vegas_map.train_mcmc(nsamples_train, func, epoch=10, alpha=1.0)
res = vegas_integrator(func)
print("VEGAS Integral results: ", res)

res = vegasmcmc_integrator(func, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)

vegas_map.make_uniform()
res = vegas_integrator(func)
print("\nNaive MC Integral results: ", res)
res = vegasmcmc_integrator(func, mix_rate=0.5)
print("MCMC Integral results: ", res)


# Start Monte Carlo integration, including plain-MC, MCMC, vegas, and vegas-MCMC
print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

bounds = [(0, 1)] * 4
vegas_map = Vegas(bounds, device=device, ninc=ninc)
print("train VEGAS map for h(X)...")
vegas_map.train(nsamples_train, sharp_peak, epoch=10, alpha=1.0)

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    device=device,
)
res = vegas_integrator(sharp_integrands, f_dim=3)
print(
    "  I[0] =",
    res[0],
    "  I[1] =",
    res[1],
    "  I[2] =",
    res[2],
    "  I[1]/I[0] =",
    res[1] / res[0],
)
# print(type(res))
# print(type(res[0]))
# print(res[0].sum_neval)
# print(res[0].itn_results)
# print(res[0].nitn)


print("VEGAS-MCMC Integral results:")
vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(sharp_integrands, f_dim=3, mix_rate=0.5)
print(
    "  I[0] =",
    res[0],
    "  I[1] =",
    res[1],
    "  I[2] =",
    res[2],
    "  I[1]/I[0] =",
    res[1] / res[0],
)

vegas_map.make_uniform()
print("\ntrain VEGAS map by MCMC sampling data...")
vegas_map.train_mcmc(nsamples_train, sharp_peak, epoch=10, alpha=1.0)
res = vegas_integrator(sharp_integrands, f_dim=3)
print("VEGAS Integral results: ")
print(
    "  I[0] =",
    res[0],
    "  I[1] =",
    res[1],
    "  I[2] =",
    res[2],
    "  I[1]/I[0] =",
    res[1] / res[0],
)
res = vegasmcmc_integrator(sharp_integrands, f_dim=3, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ")
print(
    "  I[0] =",
    res[0],
    "  I[1] =",
    res[1],
    "  I[2] =",
    res[2],
    "  I[1]/I[0] =",
    res[1] / res[0],
)
