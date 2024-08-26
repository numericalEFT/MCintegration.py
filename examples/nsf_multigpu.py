# Import required packages
import torch
import numpy as np
import normflows as nf
import benchmark
from scipy.special import erf, gamma
import vegas
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from tqdm import tqdm
# import h5py

from absl import app, flags

# enable_cuda = True
# device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_model_parallel(
    rank,
    world_size,
    nfm,
    max_iter=1000,
    num_samples=10000,
    accum_iter=10,
    has_scheduler=1,
    proposal_model=None,
    save_checkpoint=True,
):
    # Train model
    # Move model on GPU if available
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    ddp_model = DDP(nfm.to(rank), device_ids=[rank])
    clip = 10.0

    loss_hist = []
    # writer = SummaryWriter()

    print("before training \n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=8e-3
    )  # , weight_decay=1e-5)

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    if has_scheduler == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
    elif has_scheduler == 2:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_iter - warmup_epochs
        )
    elif has_scheduler == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=(max_iter - warmup_epochs) // 3, T_mult=2
        )

    if proposal_model is not None:
        proposal_model.to(rank)
        proposal_model.mcmc_sample(200, init=True)

    # for name, module in ddp_model.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in range(max_iter):
        start_time = time.time()

        optimizer.zero_grad()

        loss_accum = torch.zeros(1, requires_grad=False, device=rank)
        for _ in range(accum_iter):
            # Compute loss
            #     if(it<max_iter/2):
            #         loss = ddp_model.reverse_kld(num_samples)
            #     else:
            if proposal_model is None:
                loss = ddp_model.module.IS_forward_kld(num_samples)
            else:
                x = proposal_model.mcmc_sample()
                loss = ddp_model.module.forward_kld(x)

            loss = loss / accum_iter
            loss_accum += loss
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                # torch.nn.utils.clip_grad_value_(ddp_model.parameters(), clip)

        torch.nn.utils.clip_grad_norm_(
            ddp_model.module.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()
        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        elif has_scheduler == 1:
            scheduler.step(loss_accum)  # ReduceLROnPlateau
        elif has_scheduler == 2:
            scheduler.step(it - warmup_epochs)  # CosineAnnealingLR
        elif has_scheduler == 3:
            scheduler.step(it - warmup_epochs)  # CosineAnnealingWarmRestarts
        # Log loss
        loss_hist.append(loss_accum.item())

        # # Log metrics
        # writer.add_scalar("Loss/train", loss.item(), it)
        # writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], it)

        if it % 10 == 0:
            print(
                f"Iteration {it}, Loss: {loss_accum.item()}, Learning Rate: {optimizer.param_groups[0]['lr']}, Running time: {time.time() - start_time:.3f}s"
            )

        # save checkpoint
        if (it % 100 == 0 or it == max_iter - 1) and save_checkpoint and rank == 0:
            torch.save(
                # {
                #    "model_state_dict": ddp_model.state_dict(),
                #    "optimizer_state_dict": optimizer.state_dict(),
                #    "scheduler_state_dict": scheduler.state_dict()
                #    if has_scheduler
                #    else None,
                #    "loss_hist": loss_hist,
                #    "it": it,
                # },
                # ddp_model.state_dict(),
                ddp_model.module,
                f"checkpoint.pt",
            )
        # dist.barrier()
    # writer.close()
    print("after training \n")
    # print(ddp_model.flows[0].pvct.grid)
    # print(ddp_model.flows[0].pvct.inc)
    print(loss_hist)
    cleanup()


def run_train(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def run_train_old(target_fn, world_size):
    mp.set_start_method("spawn")
    processes = []
    manager = mp.Manager()
    result_queue = manager.Queue()

    for rank in range(world_size):
        p = mp.Process(target=target_fn, args=(rank, world_size, result_queue))
        p.start()
        processes.append(p)

    # Collect results from each process
    for p in processes:
        p.join()

    # Retrieve the result from the process with rank=0
    while not result_queue.empty():
        rank, result = copy.deepcopy(result_queue.get())
        if rank == 0:
            cleanup()
            return result

    return None
