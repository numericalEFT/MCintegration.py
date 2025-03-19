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

        def sharp_integrands(x, f):
            f[:, 0] = torch.sum((x - 0.5) ** 2, dim=-1)
            f[:, 0] *= -200
            f[:, 0].exp_()
            f[:, 1] = f[:, 0] * x[:, 0]
            f[:, 2] = f[:, 0] * x[:, 0] ** 2
            return f.mean(dim=-1)


        dim = 4
        bounds = [(0, 1)] * dim
        n_eval = 6400000
        batch_size = 40000
        n_therm = 20

        device = torch.device(f"cuda:{rank}")

        print(f"Process {rank} using device: {device}")

        vegas_map = Vegas(dim, device=device, ninc=1000)

        # Plain MC and MCMC
        mc_integrator = MonteCarlo(
            f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size,device=device
        )
        mcmc_integrator = MarkovChainMonteCarlo(
            f=sharp_integrands, f_dim=3, bounds=bounds, batch_size=batch_size, nburnin=n_therm,device=device
        )

        print("Sharp Peak Integration Results:")
        print("Plain MC:", mc_integrator(n_eval))
        print("MCMC:", mcmc_integrator(n_eval, mix_rate=0.5))

        # Train VEGAS map
        vegas_map.adaptive_training(batch_size, sharp_integrands, f_dim=3, epoch=10, alpha=2.0)
        vegas_integrator = MonteCarlo(
            bounds, f=sharp_integrands, f_dim=3, maps=vegas_map, batch_size=batch_size,device=device
        )
        vegasmcmc_integrator = MarkovChainMonteCarlo(
            bounds,
            f=sharp_integrands,
            f_dim=3,
            maps=vegas_map,
            batch_size=batch_size,
            nburnin=n_therm,
            device=device
        )

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