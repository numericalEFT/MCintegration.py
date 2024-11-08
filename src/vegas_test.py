# Integration tests for VEGAS + MonteCarlo/MCMC integral methods.
import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Linear
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


alpha = 2.0
ninc = 1000
n_eval = 1000000
n_batch = 20000
n_therm = 10

print("\nCalculate the integral log(x)/x^0.5 in the bounds [0, 1]")

print("train VEGAS map")
vegas_map = Vegas([(0, 1)], device=device, ninc=ninc)
vegas_map.train(100000, func, epoch=10, alpha=alpha)

vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    # nbatch=n_batch,
    device=device,
)
res = vegas_integrator(func)
print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(func, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)
print(type(res))

# Start Monte Carlo integration, including plain-MC, MCMC, vegas, and vegas-MCMC
print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

bounds = [(0, 1)] * 4
vegas_map = Vegas(bounds, device=device, ninc=ninc)
print("train VEGAS map for h(X)...")
vegas_map.train(20000, sharp_peak, epoch=10, alpha=alpha)
# print(vegas_map.extract_grid())

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=50000,
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
print(type(res))
print(type(res[0]))
print(res[0].sum_neval)
print(res[0].itn_results)
print(res[0].nitn)


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
