from typing import Callable
import torch
from utils import RAvg
from maps import Linear
from base import Uniform, EPSILON
import numpy as np

import os
import torch.distributed as dist
import socket


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup():
    # get IDs of reserved GPU
    distributed_init_method = f"tcp://{get_ip()}:{get_open_port()}"
    dist.init_process_group(
        backend="gloo"
    )  # , init_method=distributed_init_method, world_size = int(os.environ["WORLD_SIZE"]), rank = int(os.environ["RANK"]))
    # init_method='env://',
    # world_size=int(os.environ["WORLD_SIZE"]),
    # rank=int(os.environ['SLURM_PROCID']))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


class Integrator:
    """
    Base class for all integrators. This class is designed to handle integration tasks
    over a specified domain (bounds) using a sampling method (q0) and optional
    transformation maps.
    """

    def __init__(
        self,
        maps=None,
        bounds=None,
        q0=None,
        neval: int = 1000,
        nbatch: int = None,
        device="cpu",
        dtype=torch.float64,
    ):
        self.dtype = dtype
        if maps:
            if not self.dtype == maps.dtype:
                raise ValueError(
                    "Data type of the variables of integrator should be same as maps."
                )
            self.bounds = maps.bounds
        else:
            if not isinstance(bounds, (list, np.ndarray)):
                raise TypeError("bounds must be a list or a NumPy array.")
            self.bounds = torch.tensor(bounds, dtype=dtype, device=device)

        self.dim = len(self.bounds)
        if not q0:
            q0 = Uniform(self.bounds, device=device, dtype=dtype)
        self.q0 = q0
        self.maps = maps
        self.neval = neval
        if nbatch is None:
            self.nbatch = neval
            self.neval = neval
        else:
            self.nbatch = nbatch
            self.neval = -(-neval // nbatch) * nbatch

        self.device = device

    def __call__(self, f: Callable, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def sample(self, nsample, **kwargs):
        u, log_detJ = self.q0.sample(nsample)
        if not self.maps:
            return u, log_detJ
        else:
            u, log_detj = self.maps.forward(u)
            return u, log_detJ + log_detj


class MonteCarlo(Integrator):
    def __init__(
        self,
        maps=None,
        bounds=None,
        q0=None,
        neval: int = 1000,
        nbatch: int = None,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(maps, bounds, q0, neval, nbatch, device, dtype)

    def __call__(self, f: Callable, f_dim: int = 1, multigpu=False, **kwargs):
        if multigpu:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        x, _ = self.sample(self.nbatch)
        fx = torch.empty((self.nbatch, f_dim), dtype=self.dtype, device=self.device)

        epoch = self.neval // self.nbatch
        integ_values = torch.zeros(
            (self.nbatch, f_dim), dtype=self.dtype, device=self.device
        )
        results = np.array([RAvg() for _ in range(f_dim)])

        for _ in range(epoch):
            x, log_detJ = self.sample(self.nbatch)
            f(x, fx)
            fx.mul_(log_detJ.exp_().unsqueeze_(1))
            integ_values += fx / epoch

        results = self.statistics(integ_values, results, rank, world_size)
        if rank == 0:
            return results

    def statistics(
        self,
        values,
        results,
        rank=0,
        world_size=1,
    ):
        f_dim = values.shape[1]
        _mean = values.mean(dim=0)
        _var = values.var(dim=0) / self.nbatch

        if world_size > 1:
            # Gather mean and variance statistics to rank 0
            if rank == 0:
                gathered_means = [
                    torch.zeros(f_dim, dtype=self.dtype, device=self.device)
                    for _ in range(world_size)
                ]
                gathered_vars = [
                    torch.zeros(f_dim, dtype=self.dtype, device=self.device)
                    for _ in range(world_size)
                ]
            dist.gather(_mean, gathered_means if rank == 0 else None, dst=0)
            dist.gather(_var, gathered_vars if rank == 0 else None, dst=0)

            if rank == 0:
                for ngpu in range(world_size):
                    for i in range(f_dim):
                        results[i].update(
                            gathered_means[ngpu][i].item(),
                            gathered_vars[ngpu][i].item(),
                            self.neval,
                        )
        else:
            for i in range(f_dim):
                results[i].update(_mean[i].item(), _var[i].item(), self.neval)

        if rank == 0:
            if f_dim == 1:
                return results[0]
            else:
                return results


def random_walk(dim, bounds, device, dtype, u, **kwargs):
    rangebounds = bounds[:, 1] - bounds[:, 0]
    step_size = kwargs.get("step_size", 0.2)
    step_sizes = rangebounds * step_size
    step = torch.empty(dim, device=device, dtype=dtype).uniform_(-1, 1) * step_sizes
    new_u = (u + step - bounds[:, 0]) % rangebounds + bounds[:, 0]
    return new_u


def uniform(dim, bounds, device, dtype, u, **kwargs):
    rangebounds = bounds[:, 1] - bounds[:, 0]
    return torch.rand_like(u) * rangebounds + bounds[:, 0]


def gaussian(dim, bounds, device, dtype, u, **kwargs):
    mean = kwargs.get("mean", torch.zeros_like(u))
    std = kwargs.get("std", torch.ones_like(u))
    return torch.normal(mean, std)


class MCMC(MonteCarlo):
    def __init__(
        self,
        maps=None,
        bounds=None,
        q0=None,
        neval: int = 10000,
        nbatch: int = None,
        nburnin: int = 500,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(maps, bounds, q0, neval, nbatch, device, dtype)
        self.nburnin = nburnin

        # If no transformation maps are provided, use a linear map as default
        if maps is None:
            self.maps = Linear([(0, 1)] * self.dim, device=device)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def __call__(
        self,
        f: Callable,
        f_dim: int = 1,
        proposal_dist: Callable = uniform,
        mix_rate=0.5,
        meas_freq: int = 1,
        multigpu=False,
        **kwargs,
    ):
        if multigpu:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        epoch = self.neval // self.nbatch
        current_y, current_jac = self.q0.sample(self.nbatch)
        current_x, detJ = self.maps.forward(current_y)
        current_jac += detJ
        current_jac.exp_()
        fx = torch.empty((self.nbatch, f_dim), dtype=self.dtype, device=self.device)

        current_weight = (
            mix_rate / current_jac + (1 - mix_rate) * f(current_x, fx).abs_()
        )
        current_weight.masked_fill_(current_weight < EPSILON, EPSILON)

        n_meas = epoch // meas_freq

        def one_step(current_y, current_x, current_weight, current_jac):
            proposed_y = proposal_dist(
                self.dim, self.bounds, self.device, self.dtype, current_y, **kwargs
            )
            proposed_x, new_jac = self.maps.forward(proposed_y)
            new_jac.exp_()

            new_weight = mix_rate / new_jac + (1 - mix_rate) * f(proposed_x, fx).abs_()
            new_weight.masked_fill_(new_weight < EPSILON, EPSILON)

            acceptance_probs = new_weight / current_weight * new_jac / current_jac

            accept = (
                torch.rand(self.nbatch, dtype=torch.float64, device=self.device)
                <= acceptance_probs
            )

            accept_expanded = accept.unsqueeze(1)
            current_y.mul_(~accept_expanded).add_(proposed_y * accept_expanded)
            current_x.mul_(~accept_expanded).add_(proposed_x * accept_expanded)
            current_weight.mul_(~accept).add_(new_weight * accept)
            current_jac.mul_(~accept).add_(new_jac * accept)

        for _ in range(self.nburnin):
            one_step(current_y, current_x, current_weight, current_jac)

        values = torch.zeros((self.nbatch, f_dim), dtype=self.dtype, device=self.device)
        refvalues = torch.zeros(self.nbatch, dtype=self.dtype, device=self.device)
        results_unnorm = np.array([RAvg() for _ in range(f_dim)])
        results_ref = RAvg()

        for _ in range(n_meas):
            for _ in range(meas_freq):
                one_step(current_y, current_x, current_weight, current_jac)
            f(current_x, fx)

            fx.div_(current_weight.unsqueeze(1))
            values += fx / n_meas
            refvalues += 1 / (current_jac * current_weight) / n_meas

        results = self.statistics(
            values, refvalues, results_unnorm, results_ref, rank, world_size
        )
        if rank == 0:
            return results

    def statistics(
        self,
        values,
        refvalues,
        results_unnorm,
        results_ref,
        rank=0,
        world_size=1,
    ):
        f_dim = values.shape[1]
        _mean_ref = refvalues.mean()
        _var_ref = refvalues.var() / self.nbatch
        _mean = values.mean(dim=0)
        _var = values.var(dim=0) / self.nbatch

        if world_size > 1:
            # Gather mean and variance statistics to rank 0
            if rank == 0:
                gathered_means = [torch.zeros_like(_mean) for _ in range(world_size)]
                gathered_vars = [torch.zeros_like(_var) for _ in range(world_size)]
                gathered_means_ref = [
                    torch.zeros_like(_mean_ref) for _ in range(world_size)
                ]
                gathered_vars_ref = [
                    torch.zeros_like(_var_ref) for _ in range(world_size)
                ]
            dist.gather(_mean, gathered_means if rank == 0 else None, dst=0)
            dist.gather(_var, gathered_vars if rank == 0 else None, dst=0)
            dist.gather(_mean_ref, gathered_means_ref if rank == 0 else None, dst=0)
            dist.gather(_var_ref, gathered_vars_ref if rank == 0 else None, dst=0)

            if rank == 0:
                for ngpu in range(world_size):
                    for i in range(f_dim):
                        results_unnorm[i].update(
                            gathered_means[ngpu][i].item(),
                            gathered_vars[ngpu][i].item(),
                            self.neval,
                        )
                    results_ref.update(
                        gathered_means_ref[ngpu].item(),
                        gathered_vars_ref[ngpu].item(),
                        self.neval,
                    )
        else:
            for i in range(f_dim):
                results_unnorm[i].update(_mean[i].item(), _var[i].item(), self.neval)
            results_ref.update(_mean_ref.item(), _var_ref.item(), self.neval)

        if rank == 0:
            if f_dim == 1:
                res = results_unnorm[0] / results_ref * self._rangebounds.prod()
                result = RAvg(itn_results=[res], sum_neval=self.neval)
                return result
            else:
                return results_unnorm / results_ref * self._rangebounds.prod().item()
