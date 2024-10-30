import torch
from integrators import MonteCarlo, MCMC, setup
from maps import Vegas, Linear
from utils import set_seed, get_device

set_seed(42)
# device = get_device()
setup()
# device = torch.device("cpu")
device = torch.cuda.current_device()


def unit_circle_integrand(x):
    inside_circle = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
    return inside_circle


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

print("Calculate the area of the unit circle f(x1, x2) in the bounds [-1, 1]^2...")
res = mc_integrator.gpu_run(unit_circle_integrand)
print("Plain MC Integral results: ", res)
