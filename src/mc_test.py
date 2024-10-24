import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Linear
from utils import set_seed, get_device

set_seed(42)
# device = get_device()
device = torch.device("cpu")


def test_nothing():
    pass


def unit_circle_integrand(x):
    inside_circle = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return inside_circle


def half_sphere_integrand(x):
    return torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2


def integrand_list(x):
    return [unit_circle_integrand(x), half_sphere_integrand(x) / 2]


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


dim = 2
bounds = [(-1, 1), (-1, 1)]
n_eval = 400000
n_batch = 10000
n_therm = 10

vegas_map = Vegas(bounds, device=device, ninc=10)


# True value of pi
print(f"pi = {torch.pi} \n")

# Start Monte Carlo integration, including plain-MC, MCMC, vegas, and vegas-MCMC
mc_integrator = MonteCarlo(
    bounds=bounds,
    neval=n_eval,
    nbatch=n_batch,
    device=device,
)
mcmc_integrator = MCMC(
    bounds=bounds, neval=n_eval, nbatch=n_batch, nburnin=n_therm, device=device
)

print("Calculate the area of the unit circle f(x1, x2) in the bounds [-1, 1]^2...")
res = mc_integrator(unit_circle_integrand)
print("Plain MC Integral results: ", res)

res = mcmc_integrator(unit_circle_integrand, mix_rate=0.5)
print("MCMC Integral results: ", res)

vegas_map.train(20000, unit_circle_integrand, alpha=0.5)
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    # nbatch=n_eval,
    device=device,
)
res = vegas_integrator(unit_circle_integrand)
print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(unit_circle_integrand, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res, "\n")


print(
    r"Calculate the integral g(x1, x2) = $2 \max(1-(x_1^2+x_2^2), 0)$ in the bounds [-1, 1]^2..."
)

res = mc_integrator(half_sphere_integrand)
print("Plain MC Integral results: ", res)

res = mcmc_integrator(half_sphere_integrand, mix_rate=0.5)
print("MCMC Integral results:", res)

# train the vegas map
vegas_map.train(20000, half_sphere_integrand, epoch=10, alpha=0.5)

res = vegas_integrator(half_sphere_integrand)
print("VEGAS Integral results: ", res)

res = vegasmcmc_integrator(half_sphere_integrand, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)


print("\nCalculate the integral [f(x1, x2), g(x1, x2)/2] in the bounds [-1, 1]^2")
# Two integrands
res = mc_integrator(integrand_list)
print("Plain MC Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])

res = mcmc_integrator(integrand_list, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral 1: ", res[0])
print(f"  Integral 2: ", res[1])

# print("VEAGS map is trained for g(x1, x2)")
vegas_map.train(20000, integrand_list, epoch=10, alpha=0.5)
res = vegas_integrator(integrand_list)
print("VEGAS Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])

res = vegasmcmc_integrator(integrand_list, mix_rate=0.5)
print("VEGAS-MCMC Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])

print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

bounds = [(0, 1)] * 4
mc_integrator = MonteCarlo(
    bounds=bounds,
    neval=n_eval,
    nbatch=n_batch,
    # nbatch=n_eval,
    device=device,
)
mcmc_integrator = MCMC(
    bounds=bounds, neval=n_eval, nbatch=n_batch, nburnin=n_therm, device=device
)
print("Plain MC Integral results:")
res = mc_integrator(integrand_list1)
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
print("MCMC Integral results:")
res = mcmc_integrator(integrand_list1, mix_rate=0.5)
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

vegas_map = Vegas(bounds, device=device)
print("train VEGAS map for h(X)...")
# vegas_map.train(20000, sharp_peak, epoch=10, alpha=0.5)
vegas_map.train(20000, integrand_list1, epoch=10, alpha=0.5)

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    # nbatch=n_eval,
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
