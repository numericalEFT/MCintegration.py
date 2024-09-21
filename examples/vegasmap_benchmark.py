import numpy as np
import os
import pandas as pd
import time  # For benchmarking
from parquetAD import FeynmanDiagram, load_leaf_info
import matplotlib.pyplot as plt
import vegas
from MCFlow.vegas_torch import VegasMap

import torch.utils.benchmark as benchmark


# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")
torch.set_printoptions(precision=10)  # Set displayed output precision to 10 digits

root_dir = os.path.join(os.path.dirname(__file__), "funcs_sigma/")
num_loops = [2, 6, 15, 39, 111, 448]
order = 1
dim = 4 * order - 1
beta = 32.0
solution = 0.23  # order 2
# solution = -0.03115 # order 3
integration_domain = [[0, 1]] * dim
niters = 20
num_adapt_samples = 1000

alpha_opt = abs(solution / (solution + 1))
# batchsize = 32768
batchsize = 1000
# nblocks = 3052
nblocks = 100
# therm_steps = 1000
therm_steps = 50
mu = 0.0
step_size = 0.1
# type = "gaussian"  # "gaussian" or "uniform"
type = "uniform"  # "gaussian" or "uniform"
# type = None
mix_rate = 0.1

print(
    f"batchsize {batchsize}, nblocks {nblocks}, therm_steps {therm_steps}, mix_rate {mix_rate}"
)
if type == "gaussian":
    print(f"Gaussian random-walk N({mu}, {step_size}^2)")
elif type == "uniform":
    print(f"Uniform random-walk U(-{step_size}, {step_size})")
else:
    print("Global random sampling")
print("\n")

partition = [(order, 0, 0)]
name = "sigma"
df = pd.read_csv(os.path.join(root_dir, f"loopBasis_{name}_maxOrder6.csv"))
with torch.no_grad():
    # loopBasis = torch.tensor([df[col].iloc[:maxMomNum].tolist() for col in df.columns[:num_loops[order-1]]]).T
    loopBasis = torch.Tensor(df.iloc[: order + 1, : num_loops[order - 1]].to_numpy())
leafstates = []
leafvalues = []

for key in partition:
    key_str = "".join(map(str, key))
    state, values = load_leaf_info(root_dir, name, key_str)
    leafstates.append(state)
    leafvalues.append(values)

diagram_eval = FeynmanDiagram(
    order, beta, loopBasis, leafstates[0], leafvalues[0], batchsize
)

###### VegasMap by torch
map_torch = VegasMap(
    diagram_eval, dim, integration_domain, batchsize, num_adapt_samples
)
map_torch = map_torch.to(device)


# benchmark
def vegasmap_func():
    map_torch.y[:] = torch.rand(map_torch.y.shape, device=device)
    map_torch.x[:], map_torch.jac[:] = map_torch.forward(map_torch.y)


t1 = benchmark.Timer(
    stmt="vegasmap_func()",
    globals={"vegasmap_func": vegasmap_func},
    label="VegasMap (order {0} beta {1})".format(order, beta),
    sub_label="forward (batchsize {0})".format(batchsize),
)
print(t1.timeit(50))

# Vegas-map MCMC
len_chain = nblocks
start_time = time.time()
t1 = benchmark.Timer(
    stmt="map_torch.mcmc(len_chain, alpha=0.1, burn_in=therm_steps, type=type, mix_rate=mix_rate)",
    globals={
        "map_torch": map_torch,
        "len_chain": len_chain,
        "therm_steps": therm_steps,
        "type": type,
        "mix_rate": mix_rate,
    },
    label="VegasMap (order {0} beta {1})".format(order, beta),
    sub_label="MCMC (batchsize {0})".format(batchsize),
)
print("benchmark time:", time.time() - start_time)

print(t1.timeit(5))

# start_time = time.time()
# mean, error, adapt_step_size = map_torch.mcmc(
#     len_chain,
#     alpha=0.0,
#     burn_in=therm_steps,
#     step_size=step_size,
#     mu=mu,
#     type=type,
#     mix_rate=mix_rate,
#     adaptive=True,
# )
# print("   VEGAS-map MCMC (alpha = 0):", f"{mean:.6f} +- {error:.6f}")
# print("MCMC integration time: {:.3f}s \n".format(time.time() - start_time))


# # Importance sampling with Vegas map (torch)
# start_time = time.time()
# mean, std = map_torch.integrate_block(nblocks)
# print("   Importance sampling with VEGAS map (torch):", f"{mean:.6f} +- {std:.6f}")
# end_time = time.time()
# wall_clock_time = end_time - start_time
# print(f"Wall-clock time: {wall_clock_time:.3f} seconds \n")
