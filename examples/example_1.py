# Example 1: Unit Circle and Half-Sphere Integrands Comparison
#
# This example demonstrates how different Monte Carlo integration methods perform
# with two different integrand functions:
# 1. Unit Circle: Integrating the indicator function of a unit circle (area = π)
# 2. Half-Sphere: Integrating a function representing a half-sphere (volume = 2π/3)
#
# The example compares:
# - Plain Monte Carlo integration
# - Markov Chain Monte Carlo (MCMC)
# - VEGAS algorithm (adaptive importance sampling)
# - VEGAS with MCMC
#
# Both integrands are defined over the square [-1,1]×[-1,1]

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import traceback
from MCintegration import MonteCarlo, MarkovChainMonteCarlo, Vegas

os.environ["NCCL_DEBUG"] = "OFF"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["GLOG_minloglevel"] = "2"
os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

backend = "nccl"
# backend = "gloo"


def init_process(rank, world_size, fn, backend=backend):
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        fn(rank, world_size)
    except Exception as e:
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e


def run_mcmc(rank, world_size):
    try:
        if rank != 0:
            sys.stderr = open(os.devnull, "w")
            sys.stdout = open(os.devnull, "w")
        torch.manual_seed(42 + rank)

        def unit_circle_integrand(x, f):
            f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
            return f[:, 0]

        def half_sphere_integrand(x, f):
            f[:, 0] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
            return f[:, 0]

        dim = 2
        bounds = [(-1, 1), (-1, 1)]
        n_eval = 6400000
        batch_size = 40000
        alpha = 2.0
        ninc = 1000
        n_therm = 20

        if backend == "gloo":
            device = torch.device("cpu")
        elif backend == "nccl":
            device = torch.device(f"cuda:{rank}")
        else:
            raise ValueError(f"Invalid backend: {backend}")

        vegas_map = Vegas(dim, device=device, ninc=ninc)

        # Monte Carlo and MCMC for Unit Circle
        mc_integrator = MonteCarlo(
            f=unit_circle_integrand, bounds=bounds, batch_size=batch_size, device=device
        )
        mcmc_integrator = MarkovChainMonteCarlo(
            f=unit_circle_integrand,
            bounds=bounds,
            batch_size=batch_size,
            nburnin=n_therm,
            device=device,
        )

        print("Unit Circle Integration Results:")
        print("Plain MC:", mc_integrator(n_eval))
        print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

        # Train VEGAS map for Unit Circle
        vegas_map.adaptive_training(batch_size, unit_circle_integrand, alpha=alpha)
        vegas_integrator = MonteCarlo(
            bounds,
            f=unit_circle_integrand,
            maps=vegas_map,
            batch_size=batch_size,
            device=device,
        )
        vegasmcmc_integrator = MarkovChainMonteCarlo(
            bounds,
            f=unit_circle_integrand,
            maps=vegas_map,
            batch_size=batch_size,
            nburnin=n_therm,
            device=device,
        )

        print("VEGAS:", vegas_integrator(n_eval))
        print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))

        # Monte Carlo and MCMC for Half-Sphere
        mc_integrator.f = half_sphere_integrand
        mcmc_integrator.f = half_sphere_integrand

        print("\nHalf-Sphere Integration Results:")
        print("Plain MC:", mc_integrator(n_eval))
        print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

        vegas_map.make_uniform()
        vegas_map.adaptive_training(
            batch_size, half_sphere_integrand, epoch=10, alpha=alpha
        )
        vegas_integrator.f = half_sphere_integrand
        vegasmcmc_integrator.f = half_sphere_integrand

        print("VEGAS:", vegas_integrator(n_eval))
        print("VEGAS-MCMC:", vegasmcmc_integrator(n_eval, mix_rate=0.5))

    except Exception as e:
        print(f"Error in run_mcmc for rank {rank}: {e}")
        traceback.print_exc()
        raise e
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_mcmc(world_size):
    world_size = min(world_size, mp.cpu_count())
    try:
        mp.spawn(
            init_process,
            args=(world_size, run_mcmc),
            nprocs=world_size,
            join=True,
            daemon=False,
        )
    except Exception as e:
        print(f"Error in test_mcmc: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test_mcmc(4)
