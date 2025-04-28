from feynmandiag import init_feynfunc
import torch
from MCintegration import (
    MonteCarlo,
    MarkovChainMonteCarlo,
    Vegas,
    set_seed,
    get_device,
)
import time
import numpy as np

maxOrder = 3
beta = 10.0
rs = 2.0
mass2 = 0.5

device = get_device()
batch_size = 5000
n_eval = 1000000
n_therm = 10

num_roots = [1, 2, 3, 4, 5, 6]


def zfactor_inv(maxOrder):
    sig_imag = [1.0]

    for order in range(1, maxOrder + 1):
        feynfunc = init_feynfunc(order, rs, beta, mass2, batch_size, is_real=False)
        feynfunc.to(device)
        f_dim = num_roots[order - 1]

        vegas_map = Vegas(feynfunc.ndims, ninc=1000, device=device)
        bounds = [[0, 1]] * feynfunc.ndims
        vegasmcmc_integrator = MarkovChainMonteCarlo(
            bounds,
            feynfunc,
            f_dim=f_dim,
            batch_size=batch_size,
            nburnin=n_therm,
            maps=vegas_map,
            device=device,
        )
        res = vegasmcmc_integrator(neval=n_eval, mix_rate=0.5)

        print(res)
        w0_inv = feynfunc.beta.item() / np.pi
        if order == 1:
            sig_imag.append(-res * w0_inv)
        else:
            sig_imag.append(-sum([x * w0_inv for x in res]))
    print("inverse Z:", sig_imag)

    return sig_imag


def meff_inv(maxOrder):
    sig_real = [1.0]

    for order in range(1, maxOrder + 1):
        feynfunc = init_feynfunc(
            order, rs, beta, mass2, batch_size, is_real=True, has_dk=True
        )
        # feynfunc.to(device)
        device = torch.device("cpu")
        f_dim = num_roots[order - 1]

        vegas_map = Vegas(feynfunc.ndims, ninc=1000, device=device)
        bounds = [[0, 1]] * feynfunc.ndims
        vegasmcmc_integrator = MarkovChainMonteCarlo(
            bounds,
            feynfunc,
            f_dim=f_dim,
            batch_size=batch_size,
            nburnin=n_therm,
            maps=vegas_map,
            device=device,
        )
        res = vegasmcmc_integrator(neval=n_eval, mix_rate=0.5)

        print(res)
        prefactor = feynfunc.me / feynfunc.kF
        if order == 1:
            sig_real.append(res * prefactor)
        else:
            sig_real.append(sum([x * prefactor for x in res]))
    print("inverse meff:", sig_real)

    return sig_real


print("Calculating inverse Z for each order...")
z_factor_inv = zfactor_inv(maxOrder)

print("\nCalculating inverse meff (without Z) for each order...")
m_eff_inv = meff_inv(maxOrder)

meff = sum(z_factor_inv) / sum(m_eff_inv)

print(f"\n z-factor (up to order {maxOrder}): ", 1 / sum(z_factor_inv))
print(f"\n Effective mass (up to order {maxOrder}): ", meff, "\n")
