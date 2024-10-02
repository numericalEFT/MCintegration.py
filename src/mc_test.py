import torch
from integrators import MonteCarlo, MCMC
from maps import Vegas, Affine
from utils import set_seed, get_device

set_seed(42)
device = get_device()


def unit_circle_integrand(x):
    inside_circle = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    # return {
    #     "scalar": inside_circle,
    #     "vector": torch.stack([inside_circle, 2 * inside_circle], dim=1),
    #     "matrix": torch.stack(
    #         [
    #             inside_circle.unsqueeze(1).repeat(1, 3),
    #             (2 * inside_circle).unsqueeze(1).repeat(1, 3),
    #         ],
    #         dim=1,
    #     ),
    # }
    return inside_circle


def half_sphere_integrand(x):
    return torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2


dim = 2
map_spec = [(-1, 1), (-1, 1)]

affine_map = Affine(map_spec, device=device)
# vegas_map = Vegas(map_spec, device=device)

# Monte Carlo integration
print("Calculate the area of the unit circle using Monte Carlo integration...")

mc_integrator = MonteCarlo(affine_map, neval=400000, batch_size=1000, device=device)
res = mc_integrator(unit_circle_integrand)
print("Plain MC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

res = MonteCarlo(map_spec, neval=400000, batch_size=1000, device=device)(
    unit_circle_integrand
)
print("Plain MC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")

mcmc_integrator = MCMC(
    map_spec, neval=400000, batch_size=1000, n_burnin=100, device=device
)
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

mcmc_integrator = MCMC(
    map_spec, neval=400000, batch_size=1000, n_burnin=100, device=device
)
res = mcmc_integrator(half_sphere_integrand, mix_rate=0.5)
print("MCMC Integral results:")
print(f"  Integral: {res.mean}")
print(f"  Error: {res.sdev}")
