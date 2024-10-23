import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Linear
from utils import set_seed, get_device

# set_seed(42)
# device = get_device()
device = torch.device("cpu")


def integrand_list1(x):
    dx2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d in range(4):
        dx2 += (x[:, d] - 0.5) ** 2
    f = torch.exp(-200 * dx2)
    return [f, f * x[:, 0], f * x[:, 0] ** 2]


def sharp_peak(x):
    dx2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d in range(4):
        dx2 += (x[:, d] - 0.5) ** 2
    return torch.exp(-200 * dx2)


def func(x):
    return torch.log(x[:, 0]) / torch.sqrt(x[:, 0])


ninc = 1000
n_eval = 50000
n_batch = 10000
n_therm = 10

print("\nCalculate the integral log(x)/x^0.5 in the bounds [0, 1]")

print("train VEGAS map")
vegas_map = Vegas([(0, 1)], device=device, ninc=ninc)
vegas_map.train(20000, func, epoch=10, alpha=0.5)

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

# Start Monte Carlo integration, including plain-MC, MCMC, vegas, and vegas-MCMC
print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

bounds = [(0, 1)] * 4
vegas_map = Vegas(bounds, device=device, ninc=ninc)
print("train VEGAS map for h(X)...")
vegas_map.train(20000, sharp_peak, epoch=10, alpha=0.5)
# print(vegas_map.extract_grid())

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    # nbatch=n_batch,
    nbatch=n_eval,
    device=device,
)
res = vegas_integrator(integrand_list1)
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

print("VEGAS-MCMC Integral results:")
vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(integrand_list1, mix_rate=0.5)
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
