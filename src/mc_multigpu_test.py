import torch
from integrators import MonteCarlo, MCMC, setup
from maps import Vegas, Linear
from utils import set_seed, get_device

# set_seed(42)
# device = get_device()
setup()
# device = torch.device("cpu")
device = torch.cuda.current_device()
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


def sharp_integrands(x, f, dim=4):
    f.zero_()
    for d in range(dim):
        f[:, 0] += (x[:, d] - 0.5) ** 2
    f[:, 0] *= -200
    f[:, 0].exp_()
    f[:, 1] = f[:, 0] * x[:, 0]
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f.mean(dim=-1)


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
res = mc_integrator(unit_circle_integrand, multigpu=True)
if res is not None:
    print("Plain MC Integral results: ", res)

res = mcmc_integrator(unit_circle_integrand, mix_rate=0.5, multigpu=True)
if res is not None:
    print("MCMC Integral results: ", res)

vegas_map.train(20000, unit_circle_integrand, alpha=0.5, multigpu=True)
vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    # nbatch=n_eval,
    device=device,
)
res = vegas_integrator(unit_circle_integrand, multigpu=True)
if res is not None:
    print("VEGAS Integral results: ", res)

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(unit_circle_integrand, mix_rate=0.5, multigpu=True)
if res is not None:
    print("VEGAS-MCMC Integral results: ", res, "\n")


print(
    r"Calculate the integral g(x1, x2) = $2 \max(1-(x_1^2+x_2^2), 0)$ in the bounds [-1, 1]^2..."
)

res = mc_integrator(half_sphere_integrand, multigpu=True)
if res is not None:
    print("Plain MC Integral results: ", res)

res = mcmc_integrator(half_sphere_integrand, mix_rate=0.5, multigpu=True)
if res is not None:
    print("MCMC Integral results:", res)

vegas_map.make_uniform()
# train the vegas map
vegas_map.train(20000, half_sphere_integrand, epoch=10, alpha=0.5, multigpu=True)

res = vegas_integrator(half_sphere_integrand, multigpu=True)
if res is not None:
    print("VEGAS Integral results: ", res)

res = vegasmcmc_integrator(half_sphere_integrand, mix_rate=0.5, multigpu=True)
if res is not None:
    print("VEGAS-MCMC Integral results: ", res)


print("\nCalculate the integral [f(x1, x2), g(x1, x2)/2] in the bounds [-1, 1]^2")
# Two integrands
res = mc_integrator(two_integrands, f_dim=2, multigpu=True)
if res is not None:
    print("Plain MC Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

res = mcmc_integrator(two_integrands, f_dim=2, mix_rate=0.5, multigpu=True)
if res is not None:
    print("MCMC Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

# print("VEAGS map is trained for g(x1, x2)")
vegas_map.make_uniform()
vegas_map.train(20000, two_integrands, f_dim=2, epoch=10, alpha=0.5, multigpu=True)
res = vegas_integrator(two_integrands, f_dim=2, multigpu=True)
if res is not None:
    print("VEGAS Integral results:")
    print("  Integral 1: ", res[0])
    print("  Integral 2: ", res[1])

res = vegasmcmc_integrator(two_integrands, f_dim=2, mix_rate=0.5, multigpu=True)
if res is not None:
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
res = mc_integrator(sharp_integrands, f_dim=3, multigpu=True)
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
res = mcmc_integrator(sharp_integrands, f_dim=3, mix_rate=0.5, multigpu=True)
if res is not None:
    print("MCMC Integral results:")
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
vegas_map.train(20000, sharp_integrands, f_dim=3, epoch=10, alpha=0.5, multigpu=True)

vegas_integrator = MonteCarlo(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    # nbatch=n_eval,
    device=device,
)
res = vegas_integrator(sharp_integrands, f_dim=3, multigpu=True)
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

vegasmcmc_integrator = MCMC(
    maps=vegas_map,
    neval=n_eval,
    nbatch=n_batch,
    nburnin=n_therm,
    device=device,
)
res = vegasmcmc_integrator(sharp_integrands, f_dim=3, mix_rate=0.5, multigpu=True)
if res is not None:
    print("VEGAS-MCMC Integral results:")
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
