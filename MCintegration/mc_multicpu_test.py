import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import traceback
from integrators import MonteCarlo, MarkovChainMonteCarlo

# Set environment variables before spawning processes
os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

backend = "gloo"


def init_process(rank, world_size, fn, backend=backend):
    # try:
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Call the function
    fn(rank, world_size)
    # except Exception as e:
    #     print(f"Error in process {rank}: {e}")
    #     traceback.print_exc()
    #     # Make sure to clean up
    #     if dist.is_initialized():
    #         dist.destroy_process_group()
    #     # Return non-zero to indicate error
    #     raise e


def run_mcmc(rank, world_size):
    print(world_size)
    try:
        # Set seed for reproducibility but different for each process
        torch.manual_seed(42 + rank)

        # Instantiate the MarkovChainMonteCarlo class
        bounds = [(-1, 1), (-1, 1)]
        # n_eval = 8000000 // world_size  # Divide evaluations among processes
        n_eval = 8000000
        batch_size = 10000
        n_therm = 20

        # Define the function to be integrated (dummy example)
        def two_integrands(x, f):
            f[:, 0] = (x[:, 0] ** 2 + x[:, 1] ** 2 < 1).double()
            f[:, 1] = torch.clamp(1 - (x[:, 0] ** 2 + x[:, 1] ** 2), min=0) * 2
            return f.mean(dim=-1)

        # Choose device based on availability and rank
        if torch.cuda.is_available() and torch.cuda.device_count() > world_size:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")

        print(f"Process {rank} using device: {device}")

        mcmc = MarkovChainMonteCarlo(
            bounds=bounds,
            f=two_integrands,
            f_dim=2,
            batch_size=batch_size,
            nburnin=n_therm,
            device=device,
        )

        # Call the MarkovChainMonteCarlo method
        mcmc_result = mcmc(n_eval)

        if rank == 0:
            print("MarkovChainMonteCarlo Result:", mcmc_result)

    # except Exception as e:
    #     print(f"Error in run_mcmc for rank {rank}: {e}")
    #     traceback.print_exc()
    #     raise e
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

def test_mcmc_singlethread():
    # 直接在当前进程初始化并运行，避免 mp.spawn 启动子进程
    world_size = 1
    init_process(rank=0, world_size=world_size, fn=run_mcmc, backend=backend)

def test_mcmc(world_size=2):
    # Use fewer processes than CPU cores to avoid resource contention
    # world_size = min(world_size, mp.cpu_count())
    print(f"Starting with {world_size} processes")

    # Start processes with proper error handling
    mp.spawn(
        init_process,
        args=(world_size, run_mcmc),
        nprocs=world_size,
        join=True,
        daemon=False,
    )


# if __name__ == "__main__":
#     # Prevent issues with multiprocessing on some platforms
#     mp.set_start_method("spawn", force=True)
#     test_mcmc(2)
