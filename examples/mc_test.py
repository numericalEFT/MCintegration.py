# Integration tests for MonteCarlo and MarkovChainMonteCarlo integrators class.
import torch
import MCintegration
from MCintegration import MonteCarlo, MarkovChainMonteCarlo
from MCintegration import Vegas, CompositeMap
from MCintegration import set_seed, get_device
import torch.utils.benchmark as benchmark
import normflows as nf

set_seed(42)
device = get_device()
# device = torch.device("cpu")


def test_nothing():
    pass


# def unit_circle_integrand(x, f):
#     f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
#     return f[:, 0]


# def gaussian_integrand(x, f):
#     sigma = 1.0

#     f[:, 0] = (1.0 / (2 * torch.pi * sigma**2)) * torch.exp(
#         -1.0 * torch.sum((x - 0.5) ** 2 / sigma**2, -1)
#     )
#     return f[:, 0]

# def half_sphere_integrand(x, f):
#     f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
#     return f[:, 0]


# def two_integrands(x, f):
#     f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
#     f[:, 1] = -torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0)
#     return f.mean(dim=-1)


# def sharp_integrands(x, f):
#     f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
#     f[:, 0] *= -200
#     f[:, 0].exp_()
#     f[:, 1] = f[:, 0] * x[:, 0]
#     f[:, 2] = f[:, 0] * x[:, 0] ** 2
#     return f.mean(dim=-1)


def unit_circle_integrand(x):
    f = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return f, f[:, 0]


def gaussian_integrand(x):
    sigma = 0.2
    f = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    f[:, 0] = (1.0 / (2 * torch.pi * sigma**2)) * torch.exp(
        -1.0 * torch.sum((x - 0.5) ** 2 / sigma**2, -1)
    )
    return f, f[:, 0]


def sharp_log(x):
    f = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    f[:, 0] = -torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
    return f, f[:, 0]


def half_sphere_integrand(x):
    f = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
    return f, f[:, 0]


def two_integrands(x):
    f = torch.zeros((x.shape[0], 2), device=x.device, dtype=x.dtype)
    f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    f[:, 1] = -torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0)
    return f, f.mean(dim=-1)


def sharp_integrands(x):
    f = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
    f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
    f[:, 0] *= -200
    f[:, 0].exp_()
    f[:, 1] = f[:, 0] * x[:, 0]
    f[:, 2] = f[:, 0] * x[:, 0] ** 2
    return f, f.mean(dim=-1)


dim = 2
bounds = [(0, 1), (0, 1)]
n_eval = 10000
batch_size = 5000
n_therm = 10

latent_size = 2
hidden_units = 8
num_blocks = 2
masks = nf.utils.iflow_binary_masks(latent_size)  # mask0
# masks = [torch.ones(ndims)]
print(masks)
maps = []
for mask in masks[::-1]:
    maps += [
        nf.flows.CoupledRationalQuadraticSpline(
            latent_size, num_blocks, hidden_units, mask=mask
        )
    ]
nf_map = CompositeMap(maps, device=device)
print(
    "total model params", sum(p.numel() for p in nf_map.parameters() if p.requires_grad)
)
nf_map.adaptive_train(dim, batch_size, sharp_log, epoch=100, alpha=0.5)
nf_integrator = MonteCarlo(
    bounds=bounds,
    f=sharp_log,
    maps=nf_map,
    batch_size=batch_size,
)
res = nf_integrator(n_eval)
print("NF Integral results: ", res)


vegas_map = Vegas(dim, device=device, ninc=10)


# True value of pi
print(f"pi = {torch.pi} \n")

# Start Monte Carlo integration, including plain-MC, MarkovChainMonteCarlo, vegas, and vegas-MarkovChainMonteCarlo
mc_integrator = MonteCarlo(
    f=unit_circle_integrand,
    bounds=bounds,
    batch_size=batch_size,
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=unit_circle_integrand,
    bounds=bounds,
    batch_size=batch_size,
    nburnin=n_therm,
)

print("Calculate the area of the unit circle f(x1, x2) in the bounds [-1, 1]^2...")
res = mc_integrator(n_eval)
print("Plain MC Integral results: ", res)

# result = benchmark.Timer(stmt="mc_integrator(neval=n_eval,nblock=1)", globals=globals())
# print(result.timeit(10))
# result = benchmark.Timer(
#     stmt="mc_integrator(neval=n_eval,nblock=32)", globals=globals()
# )
# print(result.timeit(10))

res = mcmc_integrator(n_eval, mix_rate=0.5)
print("MCMC Integral results: ", res)

# result = benchmark.Timer(
#     stmt="mcmc_integrator(neval=n_eval,nblock=1)", globals=globals()
# )
# print(result.timeit(10))
# result = benchmark.Timer(
#     stmt="mcmc_integrator(neval=n_eval,nblock=32)", globals=globals()
# )
# print(result.timeit(10))

vegas_map.adaptive_training(batch_size, sharp_log, alpha=0.5)
vegas_integrator = MonteCarlo(
    bounds,
    f=sharp_log,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(n_eval)
print("VEGAS Integral results: ", res)

# result = benchmark.Timer(
#     stmt="vegas_integrator(neval=n_eval,nblock=1)", globals=globals()
# )
# print(result.timeit(10))
# result = benchmark.Timer(
#     stmt="vegas_integrator(neval=n_eval,nblock=32)", globals=globals()
# )
# print(result.timeit(10))

vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    f=unit_circle_integrand,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res, "\n")

# result = benchmark.Timer(
#     stmt="vegasmcmc_integrator(neval=n_eval,nblock=1)", globals=globals()
# )
# print(result.timeit(10))
# result = benchmark.Timer(
#     stmt="vegasmcmc_integrator(neval=n_eval,nblock=32)", globals=globals()
# )
# print(result.timeit(10))

print(
    r"Calculate the integral g(x1, x2) = $2 \max(1-(x_1^2+x_2^2), 0)$ in the bounds [-1, 1]^2..."
)
mc_integrator.f = half_sphere_integrand
res = mc_integrator(n_eval)
print("Plain MC Integral results: ", res)
mcmc_integrator.f = half_sphere_integrand
res = mcmc_integrator(n_eval, mix_rate=0.5)
print("MCMC Integral results:", res)

vegas_map.make_uniform()
# train the vegas map
vegas_map.adaptive_training(batch_size, half_sphere_integrand, epoch=10, alpha=0.5)
vegas_integrator.f = half_sphere_integrand
res = vegas_integrator(n_eval)
print("VEGAS Integral results: ", res)
vegasmcmc_integrator.f = half_sphere_integrand
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
print("VEGAS-MCMC Integral results: ", res)


print("\nCalculate the integral [f(x1, x2), g(x1, x2)/2] in the bounds [-1, 1]^2")
# Two integrands
mc_integrator.f = two_integrands
mc_integrator.f_dim = 2
res = mc_integrator(n_eval)
print("Plain MC Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])
mcmc_integrator.f = two_integrands
mcmc_integrator.f_dim = 2
res = mcmc_integrator(n_eval, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral 1: ", res[0])
print(f"  Integral 2: ", res[1])

# print("VEAGS map is trained for g(x1, x2)")
vegas_map.make_uniform()
vegas_map.adaptive_training(batch_size, two_integrands, f_dim=2, epoch=10, alpha=0.5)
vegas_integrator.f = two_integrands
vegas_integrator.f_dim = 2
res = vegas_integrator(n_eval)
print("VEGAS Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])
vegasmcmc_integrator.f = two_integrands
vegasmcmc_integrator.f_dim = 2
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
print("VEGAS-MCMC Integral results:")
print("  Integral 1: ", res[0])
print("  Integral 2: ", res[1])

print("\nCalculate the integral [h(X), x1 * h(X),  x1^2 * h(X)] in the bounds [0, 1]^4")
print("h(X) = exp(-200 * (x1^2 + x2^2 + x3^2 + x4^2))")

dim = 4
bounds = [(0, 1)] * dim
mc_integrator = MonteCarlo(
    f=sharp_integrands,
    f_dim=3,
    bounds=bounds,
    batch_size=batch_size,
)
mcmc_integrator = MarkovChainMonteCarlo(
    f=sharp_integrands,
    f_dim=3,
    bounds=bounds,
    batch_size=batch_size,
    nburnin=n_therm,
)
print("Plain MC Integral results:")
res = mc_integrator(n_eval)
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
res = mcmc_integrator(n_eval, mix_rate=0.5)
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
vegas_map.adaptive_training(batch_size, sharp_integrands, f_dim=3, epoch=10, alpha=0.5)

print("VEGAS Integral results:")
vegas_integrator = MonteCarlo(
    bounds,
    f=sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
)
res = vegas_integrator(n_eval)
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
vegasmcmc_integrator = MarkovChainMonteCarlo(
    bounds,
    f=sharp_integrands,
    f_dim=3,
    maps=vegas_map,
    batch_size=batch_size,
    nburnin=n_therm,
)
res = vegasmcmc_integrator(n_eval, mix_rate=0.5)
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
