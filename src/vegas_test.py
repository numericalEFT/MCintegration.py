# Integration tests for VEGAS + MonteCarlo/MCMC integral methods.
import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Linear
from utils import set_seed, get_device

# set_seed(42)
# device = get_device()
device = torch.device("mps")
# device = torch.device("cpu")
dtype = torch.float32


def reset_nan_to_zero(tensor):
    """
    Resets NaN values in a tensor to zero in-place.

    Args:
      tensor: The PyTorch tensor to modify.
    """
    mask = torch.isnan(
        tensor)  # Create a boolean mask where True indicates NaN values
    tensor[mask] = 0  # U


def integrand_list1(x):
    dx2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d in range(4):
        dx2 += (x[:, d] - 0.5) ** 2
    f = torch.exp(-200 * dx2)
    reset_nan_to_zero(f)
    if torch.isnan(f).any():
        print("NaN detected in func")
    return [f, f * x[:, 0], f * x[:, 0] ** 2]


def sharp_peak(x):
    dx2 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for d in range(4):
        dx2 += (x[:, d] - 0.5) ** 2
    res = torch.exp(-200 * dx2)
    reset_nan_to_zero(res)
    if torch.isnan(res).any():
        print("NaN detected in func")
    return res


def func(x):
    res = torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
    reset_nan_to_zero(res)
    if torch.isnan(res).any():
        print("NaN detected in func")
    return res


ninc = 1000
n_eval = 500000
n_batch = 10000
n_therm = 10

print("\nCalculate the integral log(x)/x^0.5 in the bounds [0, 1]")

print("train VEGAS map")
vegas_map = Vegas([(0, 1)], device=device, ninc=ninc, dtype=dtype)
vegas_map.train(20000, func, epoch=10, alpha=0.5)

vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=1000000,
    nbatch=n_batch,
    device=device,
    dtype=dtype
)
res = vegas_integrator(func)
print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=1000000,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
    dtype=dtype
)
res = vegasmcmc_integrator(func, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)
print(type(res))

# Start Monte Carlo integration, including plain-MC, MCMC, vegas, and vegas-MCMC
print(
    "\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

bounds = [(0, 1)] * 4
vegas_map = Vegas(bounds, device=device, ninc=ninc, dtype=dtype)
print("train VEGAS map for h(X)...")
vegas_map.train(20000, sharp_peak, epoch=10, alpha=0.5)
# print(vegas_map.extract_grid())

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    device=device,
    dtype=dtype
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
    dtype=dtype
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
