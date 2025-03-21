# Example 3: Integration of log(x)/sqrt(x) using VEGAS

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

        def func(x, f):
            f[:, 0] = torch.log(x[:, 0]) / torch.sqrt(x[:, 0])
            return f[:, 0]

        dim = 1
        bounds = [[0, 1]] * dim
        n_eval = 6400000
        batch_size = 40000
        alpha = 2.0
        ninc = 1000
        n_therm = 20

        device = torch.device(f"cuda:{rank}")

        print(f"Process {rank} using device: {device}")

        vegas_map = Vegas(dim, device=device, ninc=ninc)

        # Train VEGAS map
        print("Training VEGAS map for log(x)/sqrt(x)... \n")
        vegas_map.adaptive_training(batch_size, func, epoch=10, alpha=alpha)

        print("Integration Results for log(x)/sqrt(x):")


        # Plain MC Integration
        mc_integrator = MonteCarlo(bounds, func, batch_size=batch_size,device=device)
        print("Plain MC Integral Result:", mc_integrator(n_eval))

        # MCMC Integration
        mcmc_integrator = MarkovChainMonteCarlo(
            bounds, func, batch_size=batch_size, nburnin=n_therm,device=device
        )
        print("MCMC Integral Result:", mcmc_integrator(n_eval, mix_rate=0.5))

        # Perform VEGAS integration
        vegas_integrator = MonteCarlo(bounds, func, maps=vegas_map, batch_size=batch_size,device=device)
        res = vegas_integrator(n_eval)

        print("VEGAS Integral Result:", res)

        # VEGAS-MCMC Integration
        vegasmcmc_integrator = MarkovChainMonteCarlo(
            bounds, func, maps=vegas_map, batch_size=batch_size, nburnin=n_therm,device=device
        )
        res_vegasmcmc = vegasmcmc_integrator(n_eval, mix_rate=0.5)
        print("VEGAS-MCMC Integral Result:", res_vegasmcmc)

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