import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from integrators import MonteCarlo, MarkovChainMonteCarlo


def init_process(rank, world_size, fn, backend="gloo"):
    # Set MASTER_ADDR and MASTER_PORT appropriately
    # Assuming environment variables are set by the cluster's job scheduler
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "12355")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)


def run_mcmc(rank, world_size):
    # Instantiate the MarkovChainMonteCarlo class
    bounds = [(-1, 1), (-1, 1)]
    n_eval = 8000000
    batch_size = 10000
    n_therm = 10

    # Define the function to be integrated (dummy example)
    def two_integrands(x, f):
        f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
        f[:, 1] = -torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0)
        return f.mean(dim=-1)

    # device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    mcmc = MarkovChainMonteCarlo(
        bounds,
        two_integrands,
        2,
        batch_size=batch_size,
        nburnin=n_therm,
        device=device,
    )

    # Call the MarkovChainMonteCarlo method
    mcmc_result = mcmc(
        neval=n_eval,
        f_dim=2,
    )

    if rank == 0:
        # Only rank 0 prints the result
        print("MarkovChainMonteCarlo Result:", mcmc_result)

def test_mcmc():
    world_size = 8  # Number of processes to launch
    mp.spawn(init_process, args=(world_size, run_mcmc), nprocs=world_size, join=True)
