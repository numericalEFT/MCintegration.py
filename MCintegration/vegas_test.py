# Integration tests for VEGAS + MonteCarlo/MarkovChainMonteCarlo integral methods.
import torch
from integrators import MonteCarlo, MarkovChainMonteCarlo
from maps import Vegas
from utils import set_seed, get_device

set_seed(42)
device = get_device()
# device = torch.device("mps")
dtype = torch.float32


def sharp_peak(x, f):
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
    f[:, 0] *= -200
    f[:, 0].exp_()
    return f[:, 0]


def sharp_integrands(x, f):
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
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
# n_eval = 1000000
n_eval = 500000
batch_size = 10000
n_therm = 10

print("\nCalculate the integral log(x)/x^0.5 in the bounds [0, 1]")
dim = 1
bounds = [[0, 1]] * dim
print("train VEGAS map")
vegas_map = Vegas(dim, device=device, ninc=ninc, dtype=dtype)
vegas_map.adaptive_training(100000, func, epoch=10, alpha=alpha)

vegas_integrator = MonteCarlo(
    bounds,
    func,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(n_eval)
print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    func,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
print("VEGAS-MarkovChainMonteCarlo Integral results: ", res)
print(type(res))
print(res.sum_neval)
print(res.itn_results)
print(res.nitn)

# Start Monte Carlo integration, including plain-MC, MarkovChainMonteCarlo, vegas, and vegas-MarkovChainMonteCarlo
print(
    "\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

dim = 4
bounds = [(0, 1)] * dim
vegas_map = Vegas(dim, device=device, ninc=ninc, dtype=dtype)
print("train VEGAS map for h(X)...")
vegas_map.adaptive_training(20000, sharp_peak, epoch=10, alpha=alpha)
# print(vegas_map.extract_grid())

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(neval=500000)
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


print("VEGAS-MarkovChainMonteCarlo Integral results:")
vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(neval=500000, mix_rate=0.5)
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
