from typing import Callable
import torch
from utils import RAvg
from maps import Linear, Configuration
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


def setup(backend="gloo"):
    # get IDs of reserved GPU
    distributed_init_method = f"tcp://{get_ip()}:{get_open_port()}"
    dist.init_process_group(
        backend=backend
    )  # , init_method=distributed_init_method, self.world_size = int(os.environ["self.world_size"]), self.rank = int(os.environ["self.rank"]))
    # init_method='env://',
    # self.world_size=int(os.environ["self.world_size"]),
    # self.rank=int(os.environ['SLURM_PROCID']))
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
        f: Callable,
        f_dim=1,
        maps=None,
        bounds=None,
        q0=None,
        nbatch=1000,
        device=None,
        dtype=torch.float64,
    ):
        self.dtype = dtype
        if maps:
            if not self.dtype == maps.dtype:
                raise ValueError(
                    "Data type of the variables of integrator should be same as maps."
                )
            if device is None:
                self.device = maps.device
            else:
                self.device = device
                maps.to(self.device)
                maps.device = self.device
            self.bounds = maps.bounds
        else:
            if not isinstance(bounds, (list, np.ndarray)):
                raise TypeError("bounds must be a list or a NumPy array.")
            if device is None:
                self.device = torch.device("cpu")
            else:
                self.device = device
            self.bounds = torch.tensor(bounds, dtype=dtype, device=self.device)

        self.dim = len(self.bounds)
        if not q0:
            q0 = Uniform(self.bounds, device=self.device, dtype=dtype)
        self.q0 = q0
        self.maps = maps
        self.nbatch = nbatch
        self.f = f
        self.f_dim = f_dim

        try:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        except ValueError as e:
            self.rank = 0
            self.world_size = 1

    def __call__(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def sample(self, sample, **kwargs):
        sample.u, sample.jac = self.q0.sample(sample.nsample)
        if not self.maps:
            sample.x[:] = sample.u
        else:
            sample.x[:], log_detj = self.maps.forward(sample.u)
            sample.jac += log_detj
        self.f(sample.x, sample.fx)
        sample.jac.exp_()

    def statistics(self, values, neval, results=None):
        dim = values.shape[1]
        if results is None:
            results = np.array([RAvg() for _ in range(dim)])
        _mean = values.mean(dim=0)
        _var = values.var(dim=0) / self.nbatch

        if self.world_size > 1:
            # Gather mean and variance statistics to self.rank 0
            if self.rank == 0:
                gathered_means = [
                    torch.zeros_like(_mean) for _ in range(self.world_size)
                ]
                gathered_vars = [torch.zeros_like(_var) for _ in range(self.world_size)]
            dist.gather(_mean, gathered_means if self.rank == 0 else None, dst=0)
            dist.gather(_var, gathered_vars if self.rank == 0 else None, dst=0)

            if self.rank == 0:
                for igpu in range(self.world_size):
                    for i in range(dim):
                        results[i].update(
                            gathered_means[igpu][i].item(),
                            gathered_vars[igpu][i].item(),
                            neval,
                        )
        else:
            for i in range(dim):
                results[i].update(_mean[i].item(), _var[i].item(), neval)


class MonteCarlo(Integrator):
    def __init__(
        self,
        f: Callable,
        f_dim=1,
        maps=None,
        bounds=None,
        q0=None,
        nbatch: int = 1000,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(f, f_dim, maps, bounds, q0, nbatch, device, dtype)

    def __call__(self, neval, nblock=32, **kwargs):
        neval = neval // self.world_size
        neval = -(-neval // self.nbatch) * self.nbatch
        epoch = neval // self.nbatch
        nsteps_perblock = epoch // nblock
        print(
            f"epoch = {epoch}, nblock = {nblock}, nsteps_perblock = {nsteps_perblock}"
        )
        assert (
            nsteps_perblock > 0
        ), f"neval ({neval}) should be larger than nbatch * nblock ({self.nbatch} * {nblock})"

        sample = Configuration(
            self.nbatch, self.dim, self.f_dim, self.device, self.dtype
        )

        epoch = neval // self.nbatch
        integ_values = torch.zeros(
            (self.nbatch, self.f_dim), dtype=self.dtype, device=self.device
        )
        results = np.array([RAvg() for _ in range(self.f_dim)])

        for _ in range(nblock):
            for _ in range(nsteps_perblock):
                self.sample(sample)
                sample.fx.mul_(sample.jac.unsqueeze_(1))
                integ_values += sample.fx / nsteps_perblock
            self.statistics(integ_values, nsteps_perblock * self.nbatch, results)
            integ_values.zero_()

        if self.rank == 0:
            if self.f_dim == 1:
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


class MarkovChainMonteCarlo(Integrator):
    def __init__(
        self,
        f: Callable,
        f_dim: int = 1,
        maps=None,
        bounds=None,
        q0=None,
        proposal_dist=None,
        nbatch: int = None,
        nburnin: int = 500,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(f, f_dim, maps, bounds, q0, nbatch, device, dtype)
        self.nburnin = nburnin
        if not proposal_dist:
            self.proposal_dist = uniform
        else:
            if not isinstance(proposal_dist, Callable):
                raise TypeError("proposal_dist must be a callable function.")
            self.proposal_dist = proposal_dist
        # If no transformation maps are provided, use a linear map as default
        if maps is None:
            self.maps = Linear([(0, 1)] * self.dim, device=self.device)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def sample(self, sample, niter=10, mix_rate=0, **kwargs):
        for _ in range(niter):
            self.metropolis_hastings(sample, mix_rate, **kwargs)

    def metropolis_hastings(self, sample, mix_rate, **kwargs):
        proposed_y = self.proposal_dist(
            self.dim, self.bounds, self.device, self.dtype, sample.u, **kwargs
        )
        proposed_x, new_jac = self.maps.forward(proposed_y)
        new_jac.exp_()

        new_weight = (
            mix_rate / new_jac + (1 - mix_rate) * self.f(proposed_x, sample.fx).abs_()
        )
        new_weight.masked_fill_(new_weight < EPSILON, EPSILON)

        acceptance_probs = new_weight / sample.weight * new_jac / sample.jac

        accept = (
            torch.rand(self.nbatch, dtype=torch.float64, device=self.device)
            <= acceptance_probs
        )

        accept_expanded = accept.unsqueeze(1)
        sample.u.mul_(~accept_expanded).add_(proposed_y * accept_expanded)
        sample.x.mul_(~accept_expanded).add_(proposed_x * accept_expanded)
        sample.weight.mul_(~accept).add_(new_weight * accept)
        sample.jac.mul_(~accept).add_(new_jac * accept)

    def __call__(
        self,
        neval,
        mix_rate=0.5,
        nblock=32,
        meas_freq: int = 1,
        **kwargs,
    ):
        neval = neval // self.world_size
        neval = -(-neval // self.nbatch) * self.nbatch
        epoch = neval // self.nbatch
        nsteps_perblock = epoch // nblock
        n_meas_perblock = nsteps_perblock // meas_freq
        assert (
            n_meas_perblock > 0
        ), f"neval ({neval}) should be larger than nbatch * nblock * meas_freq ({self.nbatch} * {nblock} * {meas_freq})"
        print(
            f"epoch = {epoch}, nblock = {nblock}, nsteps_perblock = {nsteps_perblock}, meas_freq = {meas_freq}"
        )

        sample = Configuration(
            self.nbatch, self.dim, self.f_dim, self.device, self.dtype
        )
        sample.u, sample.jac = self.q0.sample(self.nbatch)
        sample.x, detJ = self.maps.forward(sample.u)
        sample.jac += detJ
        sample.jac.exp_()
        sample.weight = (
            mix_rate / sample.jac + (1 - mix_rate) * self.f(sample.x, sample.fx).abs_()
        )
        sample.weight.masked_fill_(sample.weight < EPSILON, EPSILON)

        for _ in range(self.nburnin):
            self.metropolis_hastings(sample, mix_rate, **kwargs)

        values = torch.zeros(
            (self.nbatch, self.f_dim), dtype=self.dtype, device=self.device
        )
        refvalues = torch.zeros(self.nbatch, dtype=self.dtype, device=self.device)
        results_unnorm = np.array([RAvg() for _ in range(self.f_dim)])
        results_ref = np.array([RAvg()])

        for _ in range(nblock):
            for _ in range(n_meas_perblock):
                for _ in range(meas_freq):
                    self.metropolis_hastings(sample, mix_rate, **kwargs)
                self.f(sample.x, sample.fx)

                sample.fx.div_(sample.weight.unsqueeze(1))
                values += sample.fx / n_meas_perblock
                refvalues += 1 / (sample.jac * sample.weight) / n_meas_perblock
            self.statistics(values, nsteps_perblock * self.nbatch, results_unnorm)
            self.statistics(
                refvalues.unsqueeze(1), nsteps_perblock * self.nbatch, results_ref
            )
            values.zero_()
            refvalues.zero_()
        if self.rank == 0:
            if self.f_dim == 1:
                return results_unnorm[0] / results_ref[0] * self._rangebounds.prod()
            else:
                return results_unnorm / results_ref * self._rangebounds.prod().item()
