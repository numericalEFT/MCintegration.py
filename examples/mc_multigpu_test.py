import torch
import MCintegration
from MCintegration import (
    MonteCarlo,
    MarkovChainMonteCarlo,
    setup,
    Vegas,
    set_seed,
    get_device,
)


# backend = "nccl"
backend = "gloo"
# set_seed(42)
# setup(backend=backend)
device = get_device()
print(device)


def test_nothing():
    pass


def unit_circle_integrand(x, f):
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return f[:, 0]


def half_sphere_integrand(x, f):
    f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
    return f[:, 0]


def two_integrands(x, f):
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    f[:, 1] = -torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0)
    return f.mean(dim=-1)


def sharp_integrands(x, f):
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
    f[:, 0] *= -200
    f[:, 0].exp_()
    f[:, 1] = f[:, 0] * x[:, 0]
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f.mean(dim=-1)


dim = 2
bounds = [(-1, 1), (-1, 1)]
n_eval = 6400000
batch_size = 10000
n_therm = 10


vegas_map = Vegas(dim, device=device, ninc=10)


# True value of pi
print(f"pi = {torch.pi} \n")

# Start Monte Carlo integration, including plain-MC, MarkovChainMonteCarlo, vegas, and vegas-MarkovChainMonteCarlo
mc_integrator = MonteCarlo(
    bounds,
    unit_circle_integrand,
    batch_size=batch_size,
)
mcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    unit_circle_integrand,
    batch_size=batch_size,
    nburnin=n_therm,
)

print("Calculate the area of the unit circle f(x1, x2) in the bounds [-1, 1]^2...")
res = mc_integrator(neval=n_eval)
if res is not None:
    print("Plain MC Integral results: ", res)

res = mcmc_integrator(neval=n_eval, mix_rate=0.5)
if res is not None:
    print("MarkovChainMonteCarlo Integral results: ", res)

vegas_map.adaptive_training(20000, unit_circle_integrand, alpha=0.5)
vegas_integrator = MonteCarlo(
    bounds,
    unit_circle_integrand,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(neval=n_eval)
if res is not None:
    print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    unit_circle_integrand,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(neval=n_eval, mix_rate=0.5)
if res is not None:
    print("VEGAS-MarkovChainMonteCarlo Integral results: ", res, "\n")


print(
    r"Calculate the integral g(x1, x2) = $2 \max(1-(x_1^2+x_2^2), 0)$ in the bounds [-1, 1]^2..."
)

mc_integrator.f = half_sphere_integrand
res = mc_integrator(n_eval)
if res is not None:
    print("Plain MC Integral results: ", res)

mcmc_integrator.f = half_sphere_integrand
res = mcmc_integrator(n_eval, mix_rate=0.5)
if res is not None:
    print("MarkovChainMonteCarlo Integral results:", res)

vegas_map.make_uniform()
# train the vegas map
vegas_map.adaptive_training(20000, half_sphere_integrand, epoch=10, alpha=0.5)

vegas_integrator.f = half_sphere_integrand
res = vegas_integrator(n_eval)
if res is not None:
    print("VEGAS Integral results: ", res)

vegasmcmc_integrator.f = half_sphere_integrand
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
if res is not None:
    print("VEGAS-MarkovChainMonteCarlo Integral results: ", res)


print("\nCalculate the integral [f(x1, x2), g(x1, x2)/2] in the bounds [-1, 1]^2")
# Two integrands
mc_integrator.f = two_integrands
mc_integrator.f_dim = 2
res = mc_integrator(n_eval)
if res is not None:
    print("Plain MC Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

mcmc_integrator.f = two_integrands
mcmc_integrator.f_dim = 2
res = mcmc_integrator(n_eval, mix_rate=0.5)
if res is not None:
    print("MarkovChainMonteCarlo Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

# print("VEAGS map is trained for g(x1, x2)")
vegas_map.make_uniform()
vegas_map.adaptive_training(20000, two_integrands, f_dim=2, epoch=10, alpha=0.5)

vegas_integrator.f = two_integrands
vegas_integrator.f_dim = 2
res = vegas_integrator(n_eval)
if res is not None:
    print("VEGAS Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

vegasmcmc_integrator.f = two_integrands
vegasmcmc_integrator.f_dim = 2
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
if res is not None:
    print("VEGAS-MarkovChainMonteCarlo Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

dim = 4
bounds = [(0, 1)] * dim
mc_integrator = MonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    batch_size=batch_size,
)
mcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = mc_integrator(n_eval)
if res is not None:
    print("Plain MC Integral results:")
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
res = mcmc_integrator(n_eval, mix_rate=0.5)
if res is not None:
    print("MarkovChainMonteCarlo Integral results:")
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

vegas_map = Vegas(dim, device=device)
print("train VEGAS map for h(X)...")
# vegas_map.adaptive_training(20000, sharp_peak, epoch=10, alpha=0.5)
vegas_map.adaptive_training(20000, sharp_integrands, f_dim=3, epoch=10, alpha=0.5)

vegas_integrator = MonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(neval=n_eval)
if res is not None:
    print("VEGAS Integral results:")
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

vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(neval=n_eval, mix_rate=0.5)
if res is not None:
    print("VEGAS-MarkovChainMonteCarlo Integral results:")
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
