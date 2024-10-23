import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Linear
from utils import set_seed, get_device

set_seed(42)
device = get_device()
# device = torch.device("cpu")


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


dim = 2
bounds = [(-1, 1), (-1, 1)]
n_eval = 400000

affine_map = Linear(bounds, device=device)
# vegas_map = Vegas(bounds, device=device)

# Monte Carlo integration
print("Calculate the area of the unit circle using Monte Carlo integration...")

mc_integrator = MonteCarlo(
    # bounds, maps=affine_map, neval=n_eval, nbatch=1000, device=device
    bounds,
    neval=n_eval,
    nbatch=1000,
    device=device,
)
res = mc_integrator(unit_circle_integrand)
print("Plain MC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

res = MonteCarlo(bounds, neval=n_eval, nbatch=1000, device=device)(
    unit_circle_integrand
)
print("Plain MC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

mcmc_integrator = MCMC(bounds, neval=n_eval, nbatch=1000, nburnin=100, device=device)
res = mcmc_integrator(unit_circle_integrand, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

# True value of pi
print(f"True value of pi: {torch.pi:.6f}")

res = mc_integrator(half_sphere_integrand)
print("Plain MC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

mcmc_integrator = MCMC(bounds, neval=n_eval, nbatch=1000, nburnin=100, device=device)
res = mcmc_integrator(half_sphere_integrand, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

# Two integrands
res = mc_integrator(integrand_list)
print("Plain MC Integral results:")
print(f"  Integral 1: {res[0].mean} +- {res[0].sdev}")
print(f"  Integral 2: {res[1].mean} +- {res[1].sdev}")

res = mcmc_integrator(integrand_list, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral 1: {res[0].mean} +- {res[0].sdev}")
print(f"  Integral 2: {res[1].mean} +- {res[1].sdev}")

bounds = [(0, 1)] * 4
mc_integrator = MonteCarlo(
    bounds,
    neval=n_eval,
    nbatch=1000,
    device=device,
)
mcmc_integrator = MCMC(bounds, neval=n_eval, nbatch=1000, nburnin=100, device=device)
res = mc_integrator(integrand_list1)
print("I[0] =", res[0], "  I[1] =", res[1], "  I[2] =", res[2])
print("<x> =", res[1] / res[0])

res = mcmc_integrator(integrand_list1, mix_rate=0.5)
print("I[0] =", res[0], "  I[1] =", res[1], "  I[2] =", res[2])
print("<x> =", res[1] / res[0])
