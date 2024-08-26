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

enable_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "function",
    "Gauss",
    "The function to integrate",
    short_name="f",
)
flags.DEFINE_float("alpha", 0.05, "The width of the Gaussians", short_name="a")
flags.DEFINE_integer(
    "ndims", 2, "The number of dimensions for the integral", short_name="d"
)
flags.DEFINE_integer("epochs", 300, "Number of epochs to train", short_name="e")
flags.DEFINE_integer(
    "nsamples", 10000, "Number of points to sample per epoch", short_name="s"
)


def generate_model(
    target,
    base_dist=None,
    num_blocks=2,
    num_hidden_channels=32,
    num_bins=8,
    has_VegasLayer=False,
):
    # Define flows
    # torch.manual_seed(31)
    # K = 3
    ndims = target.ndims
    num_input_channels = ndims

    @vegas.batchintegrand
    def func(x):
        return torch.Tensor.numpy(target.prob(torch.tensor(x)))

    if has_VegasLayer:
        flows = [
            nf.flows.VegasLinearSpline(
                func, num_input_channels, [[0, 1]] * target.ndims, target.batchsize
            )
        ]
    else:
        flows = []

    masks = nf.utils.iflow_binary_masks(num_input_channels)  # mask0
    # masks = [torch.ones(num_input_channels)]
    print(masks)
    for mask in masks[::-1]:
        flows += [
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels,
                num_blocks,
                num_hidden_channels,
                num_bins=num_bins,
                mask=mask,
            )
        ]

    # mask = masks[0] * 0 + 1
    # print(mask)
    # flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels, num_blocks, num_hidden_channels, mask=mask)]
    # Set base distribuiton
    if base_dist == None:
        base_dist = nf.distributions.base.Uniform(ndims, 0.0, 1.0)

    # Construct flow model
    nfm = nf.NormalizingFlow(base_dist, flows, target).to(device)
    return nfm


# def hook_fn(module, grad_input, grad_output):
#     print(f"--- Backward pass through module {module.__class__.__name__} ---")
#     print("Grad Input (input gradient to this layer):")
#     for idx, g in enumerate(grad_input):
#         print(f"Grad Input {idx}: {g.shape} - requires_grad: {g.requires_grad if g is not None else 'N/A'}")
#     print("Grad Output (gradient from this layer to next):")
#     for idx, g in enumerate(grad_output):
#         print(f"Grad Output {idx}: {g.shape} - requires_grad: {g.requires_grad if g is not None else 'N/A'}")
#     print("\n")


def train_model(
    nfm,
    max_iter=1000,
    num_samples=10000,
    accum_iter=10,
    init_lr=8e-3,
    has_scheduler=True,
    proposal_model=None,
    save_checkpoint=True,
):
    """
    Train a neural network model with gradient accumulation.

    Args:
        nfm: The neural network model to train.
        max_iter: The maximum number of training iterations.
        num_samples: The number of samples to use for training.
        accum_iter: The number of iterations to accumulate gradients.
        has_scheduler: Whether to use a learning rate scheduler.
        proposal_model: An optional proposal model for sampling.
        save_checkpoint: Whether to save checkpoints during training every 100 iterations.
    """
    nfm = nfm.to(device)
    nfm.train()  # Set model to training mode
    loss_hist = []
    # writer = SummaryWriter()  # Initialize TensorBoard writer

    print("start training \n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(nfm.parameters(), lr=init_lr)  # , weight_decay=1e-5)
    if has_scheduler:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    if proposal_model is not None:
        proposal_model.to(device)
        proposal_model.mcmc_sample(500, init=True)

    # for name, module in nfm.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in tqdm(range(max_iter)):
        start_time = time.time()

        optimizer.zero_grad()

        loss_accum = torch.zeros(1, requires_grad=False, device=device)
        for _ in range(accum_iter):
            # Compute loss
            #     if(it<max_iter/2):
            #         loss = nfm.reverse_kld(num_samples)
            #     else:
            if proposal_model is None:
                loss = nfm.IS_forward_kld(num_samples)
                # z, _ = nfm.q0(num_samples)
                # z = nfm.forward(z)
                # loss = nfm.IS_forward_kld_direct(z.detach())
            else:
                x = proposal_model.mcmc_sample()
                loss = nfm.forward_kld(x)

            loss = loss / accum_iter
            loss_accum += loss
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()

        torch.nn.utils.clip_grad_norm_(
            nfm.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()

        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        elif has_scheduler:
            scheduler.step(loss_accum)  # ReduceLROnPlateau
            # scheduler.step()  # CosineAnnealingLR

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
        if it % 100 == 0 and it > 0 and save_checkpoint:
            torch.save(
                {
                    "model_state_dict": nfm.module.state_dict()
                    if hasattr(nfm, "module")
                    else nfm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if has_scheduler
                    else None,
                    "loss_hist": loss_hist,
                    "it": it,
                },
                f"checkpoint_{it}.pth",
            )

    # writer.close()
    print("training finished \n")
    # print(nfm.flows[0].pvct.grid)
    # print(nfm.flows[0].pvct.inc)
    print(loss_hist)


def train_model_annealing(
    nfm,
    max_iter=1000,
    num_samples=10000,
    accum_iter=1,
    init_lr=8e-3,
    annealing_factor=0.5,
    init_beta=1.0,
    lr_threshold=2e-4,
    save_checkpoint=True,
):
    nfm = nfm.to(device)
    final_beta = nfm.p.beta.item() * nfm.p.EF
    assert final_beta > init_beta, "final_beta should be greater than init_beta"
    # nfm.p.beta = torch.tensor(init_beta, requires_grad=False, device=device)
    nfm.p.beta = init_beta / nfm.p.EF
    nfm.p.mu = chemical_potential(init_beta) * nfm.p.EF

    nfm.train()  # Set model to training mode
    current_beta = init_beta
    loss_hist = np.array([])
    # writer = SummaryWriter()  # Initialize TensorBoard writer

    print("start training \n")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(nfm.parameters(), lr=init_lr)  # , weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Use a learning rate warmup
    warmup_epochs = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    # for name, module in nfm.named_modules():
    #     module.register_backward_hook(lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output))
    for it in tqdm(range(max_iter)):
        start_time = time.time()

        optimizer.zero_grad()

        loss_accum = torch.zeros(1, requires_grad=False, device=device)
        for _ in range(accum_iter):
            # Compute loss
            # loss = nfm.IS_forward_kld(num_samples)
            x = nfm.mcmc_sample()
            loss = nfm.forward_kld(x)
            loss = loss / accum_iter
            loss_accum += loss
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()

        torch.nn.utils.clip_grad_norm_(
            nfm.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()

        # Scheduler step after optimizer step
        if it < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler.step(loss_accum)  # ReduceLROnPlateau
            # scheduler.step()  # CosineAnnealingLR

        # Log loss
        loss_hist = np.append(loss_hist, loss_accum.item())
        current_lr = optimizer.param_groups[0]["lr"]

        if it % 10 == 0:
            print(
                f"Iteration {it}, beta: {current_beta}, Loss: {loss_accum.item()}, Learning Rate: {current_lr}, Running time: {time.time() - start_time:.3f}s"
            )

        if (
            it > 20 and current_beta < final_beta and it % 80 == 0
            # and current_lr < lr_threshold
            # and np.std(loss_hist[-20:]) < 4e-4
        ):
            print(f"Saving NF model with beta={current_beta}...")
            torch.save(
                {
                    "model_state_dict": nfm.module.state_dict()
                    if hasattr(nfm, "module")
                    else nfm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_hist": loss_hist,
                },
                f"nfm_beta{current_beta}_final.pt",
            )
            current_beta = current_beta / annealing_factor
            if current_beta > final_beta:
                current_beta = final_beta
            nfm.p.beta = current_beta / nfm.p.EF
            nfm.p.mu = chemical_potential(current_beta) * nfm.p.EF
            optimizer.param_groups[0]["lr"] = init_lr

            print(f"Annealing beta to {current_beta}")

        # # save checkpoint
        # if it % 100 == 0 and it > 0 and save_checkpoint:
        #     torch.save(
        #         {
        #             "model_state_dict": nfm.module.state_dict()
        #             if hasattr(nfm, "module")
        #             else nfm.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "scheduler_state_dict": scheduler.state_dict(),
        #             "loss_hist": loss_hist,
        #             "it": it,
        #         },
        #         f"checkpoint_{it}_.pth",
        #     )

    print("training finished \n")
    print(loss_hist)


def chemical_potential(beta):
    if beta == 1.0:
        _mu = -0.021460754987022185
    elif beta == 2.0:
        _mu = 0.7431120842589388
    elif beta == 4.0:
        _mu = 0.9426157552012961
    elif beta == 8.0:
        _mu = 0.986801399943294
    elif beta == 10.0:
        _mu = 0.9916412363704453
    elif beta == 16.0:
        _mu = 0.9967680535828609
    elif beta == 32.0:
        _mu = 0.9991956396090637
    elif beta == 64.0:
        _mu = 0.9997991296749593
    elif beta == 128.0:
        _mu = 0.9999497960580543
    else:
        _mu = 1.0
    return _mu


def main(argv):
    del argv
    ndims = FLAGS.ndims
    alpha = FLAGS.alpha
    nsamples = FLAGS.nsamples
    epochs = FLAGS.epochs
    block_samples = 10000
    if FLAGS.function == "Gauss":
        target = benchmark.Gauss(block_samples, ndims, alpha)
    elif FLAGS.function == "Camel":
        target = benchmark.Camel(block_samples, ndims, alpha)
    # elif FLAGS.function == "Camel_v1":
    #     target = benchmark.Camel_v1(block_samples, ndims, alpha)
    elif FLAGS.function == "Sharp":
        target = benchmark.Sharp(block_samples)
    elif FLAGS.function == "Sphere":
        target = benchmark.Sphere(block_samples, ndims)
    elif FLAGS.function == "Tight":
        target = benchmark.Tight(block_samples)
    elif FLAGS.function == "Polynomial":
        target = benchmark.Polynomial(block_samples)
    else:
        raise ValueError("Invalid function name")
    q0 = nf.distributions.base.Uniform(ndims, 0.0, 1.0)
    # nfm = generate_model(target, q0)
    nfm = generate_model(
        target,
        q0,
        # has_VegasLayer=True,
        has_VegasLayer=False,
        num_blocks=1,
        num_hidden_channels=32,
        num_bins=8,
    )

    blocks = 100
    block_samples = 10000

    # Plot initial flow distribution
    grid_size = 1000
    xx, yy = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid_size), torch.linspace(0.0, 1.0, grid_size)
    )
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    nfm.eval()
    # mean, err = nfm.integrate_block(block_samples, blocks)
    mean, err, _, _, _ = nfm.integrate_block(blocks)

    log_prob = nfm.p.log_prob(zz).to("cpu").view(*xx.shape)
    prob = nfm.p.prob(zz).to("cpu").view(*xx.shape)
    # print(prob, log_prob)
    log_q = nfm.log_prob(zz).to("cpu").view(*xx.shape)
    # log_prob = log_prob - log_q

    nfm.train()

    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    prob_q = torch.exp(log_q)
    prob_q[torch.isnan(prob_q)] = 0

    # plt.figure(figsize=(15, 15))
    # plt.pcolormesh(xx, yy, prob.data.numpy())
    # plt.gca().set_aspect("equal", "box")
    # plt.title("original distribution")
    # plt.show()

    # plt.figure(figsize=(15, 15))
    # plt.pcolormesh(xx, yy, prob_q.data.numpy())
    # plt.gca().set_aspect("equal", "box")
    # plt.title("learned distribution")
    # plt.show()
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * block_samples, mean, err, nfm.p.targetval
        )
    )

    start_time = time.time()
    loss_hist = train_model(nfm, epochs, nsamples)
    print("Training time: {:.3f}s".format(time.time() - start_time))

    # plt.figure(figsize=(10, 10))
    # plt.plot(loss_hist + np.log(mean.detach().numpy()), label="loss")
    # plt.legend()
    # plt.show()
    nfm.eval()

    log_prob = nfm.p.log_prob(zz).to("cpu").view(*xx.shape)
    prob = nfm.p.prob(zz).to("cpu").view(*xx.shape)
    log_q = nfm.log_prob(zz).to("cpu").view(*xx.shape)

    # mean, err = nfm.integrate_block(blocks)
    num_bins = 25
    mean, err, bins, histr, histr_weight = nfm.integrate_block(blocks, num_bins)

    print(bins)
    # torch.save(histr, "histogram.pt")
    # torch.save(histr_weight, "histogramWeight.pt")
    plt.figure(figsize=(15, 15))
    plt.stairs(histr[:, 0].numpy(), bins.numpy(), label="0 Dim")
    plt.title("Histogram of learned distribution")
    plt.legend()
    plt.savefig("histogram_" + FLAGS.function + "_layer1h32_allmask_dense.png")
    plt.show()

    nfm.train()

    prob = torch.exp(log_q)
    prob[torch.isnan(prob)] = 0

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy())
    plt.gca().set_aspect("equal", "box")
    plt.show()
    print(
        "Result with {:d} is {:.5e} +/- {:.5e}. \n Target result:{:.5e}".format(
            blocks * block_samples, mean, err, nfm.p.targetval
        )
    )


if __name__ == "__main__":
    app.run(main)

    # Plot learned distribution
    # if (it + 1) % show_iter == 0:
    #     nfm.eval()
    #     log_prob = nfm.log_prob(zz)
    #     nfm.train()
    #     prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    #     prob[torch.isnan(prob)] = 0

    #     plt.figure(figsize=(15, 15))
    #     plt.pcolormesh(xx, yy, prob.data.numpy())
    #     plt.gca().set_aspect('equal', 'box')
    #     plt.show()
    # scheduler.step()
