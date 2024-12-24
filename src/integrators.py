from typing import Callable
import torch
from utils import RAvg, get_device, LinearMap
from maps import Configuration, CompositeMap
from base import Uniform, EPSILON
import numpy as np
from warnings import warn

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
        bounds,
        f: Callable,
        f_dim=1,
        maps=None,
        q0=None,
        batch_size=1000,
        device=None,
        dtype=None,
    ):
        self.f = f
        self.f_dim = f_dim

        if maps:
            if dtype is None or dtype == maps.dtype:
                self.dtype = maps.dtype
            else:
                raise ValueError(
                    "Data type of the variables of integrator should be same as maps."
                )
            if device is None:
                self.device = maps.device
            else:
                self.device = device
                maps.to(self.device)
                maps.device = self.device
        else:
            if dtype is None:
                self.dtype = torch.float64
            else:
                self.dtype = dtype
            if device is None:
                self.device = get_device()
            else:
                self.device = device

        if isinstance(bounds, (list, np.ndarray)):
            self.bounds = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        elif isinstance(bounds, torch.Tensor):
            self.bounds = bounds.to(dtype=self.dtype, device=self.device)
        else:
            raise TypeError("bounds must be a list, NumPy array, or torch.Tensor.")

        linear_map = LinearMap(
            self.bounds[:, 1] - self.bounds[:, 0],
            self.bounds[:, 0],
            device=self.device,
            dtype=self.dtype,
        )
        if maps:
            self.maps = CompositeMap(
                [maps, linear_map], device=self.device, dtype=self.dtype
            )
        else:
            self.maps = linear_map

        self.dim = self.bounds.shape[0]
        if not q0:
            q0 = Uniform(self.dim, device=self.device, dtype=self.dtype)
        self.q0 = q0
        self.batch_size = batch_size
        self.f = f
        self.f_dim = f_dim

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # Fallback defaults when distributed is not initialized
            self.rank = 0
            self.world_size = 1

    def __call__(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def sample(self, config, **kwargs):
        config.u, config.detJ = self.q0.sample_with_detJ(config.batch_size)
        if not self.maps:
            config.x[:] = config.u
        else:
            config.x[:], detj = self.maps.forward_with_detJ(config.u)
            config.detJ *= detj
        self.f(config.x, config.fx)

    def statistics(self, means, vars, neval=None):
        nblock = means.shape[0]
        f_dim = means.shape[1]
        nblock_total = nblock * self.world_size
        weighted = nblock_total < 32

        if self.world_size > 1:
            # Gather mean and variance statistics to self.rank 0
            if self.rank == 0:
                gathered_means = [
                    torch.zeros_like(means) for _ in range(self.world_size)
                ]
                if weighted:
                    gathered_vars = [
                        torch.zeros_like(vars) for _ in range(self.world_size)
                    ]
            dist.gather(means, gathered_means if self.rank == 0 else None, dst=0)
            if weighted:
                dist.gather(vars, gathered_vars if self.rank == 0 else None, dst=0)

            if self.rank == 0:
                results = np.array([RAvg() for _ in range(f_dim)])
                if weighted:
                    for i in range(f_dim):
                        for iblock in range(nblock):
                            for igpu, (_mean, _var) in enumerate(
                                zip(gathered_means, gathered_vars)
                            ):
                                results[i].update(
                                    _mean[iblock, i].item(),
                                    _var[iblock, i].item(),
                                    neval,
                                )
                else:
                    for i in range(f_dim):
                        _means = torch.empty(
                            nblock_total, dtype=self.dtype, device=self.device
                        )
                        for igpu in range(self.world_size):
                            _means[igpu * nblock : (igpu + 1) * nblock] = (
                                gathered_means[igpu][:, i]
                            )
                        results[i].update(
                            _means.mean().item(),
                            _means.var().item() / nblock_total,
                            neval * nblock,
                        )
            else:
                return None
        else:
            results = np.array([RAvg() for _ in range(f_dim)])
            if weighted:
                for i in range(f_dim):
                    for iblock in range(nblock):
                        results[i].update(
                            means[iblock, i].item(),
                            vars[iblock, i].item(),
                            neval,
                        )
            else:
                for i in range(f_dim):
                    results[i].update(
                        means[:, i].mean().item(),
                        means[:, i].var().item() / nblock_total,
                        neval * nblock,
                    )
        return results


class MonteCarlo(Integrator):
    def __init__(
        self,
        bounds,
        f: Callable,
        f_dim=1,
        maps=None,
        q0=None,
        batch_size: int = 1000,
        device=None,
        dtype=None,
    ):
        super().__init__(bounds, f, f_dim, maps, q0, batch_size, device, dtype)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def __call__(self, neval, nblock=32, verbose=-1, **kwargs):
        neval = neval // self.world_size
        neval = -(-neval // self.batch_size) * self.batch_size
        epoch = neval // self.batch_size
        epoch_perblock = epoch // nblock
        if epoch_perblock == 0:
            warn(
                f"neval is too small to be divided into {nblock} blocks. Reset nblock to {epoch}."
            )
            epoch_perblock = 1
            nblock = epoch
        else:
            nblock = epoch // epoch_perblock

        if verbose > 0:
            print(
                f"nblock = {nblock}, n_steps_perblock = {epoch_perblock}, batch_size = {self.batch_size}, actual neval = {self.batch_size*epoch_perblock*nblock}"
            )

        config = Configuration(
            self.batch_size, self.dim, self.f_dim, self.device, self.dtype
        )

        epoch = neval // self.batch_size
        integ_values = torch.zeros(
            (self.batch_size, self.f_dim), dtype=self.dtype, device=self.device
        )
        means = torch.zeros((nblock, self.f_dim), dtype=self.dtype, device=self.device)
        vars = torch.zeros_like(means)

        for iblock in range(nblock):
            for _ in range(epoch_perblock):
                self.sample(config)
                config.fx.mul_(config.detJ.unsqueeze_(1))
                integ_values += config.fx / epoch_perblock
            means[iblock, :] = integ_values.mean(dim=0)
            vars[iblock, :] = integ_values.var(dim=0) / self.batch_size
            integ_values.zero_()

        results = self.statistics(means, vars, epoch_perblock * self.batch_size)

        if self.rank == 0:
            if self.f_dim == 1:
                # return results[0] / self._rangebounds.prod()
                return results[0]
            else:
                # return results / self._rangebounds.prod().item()
                return results


def random_walk(dim, device, dtype, u, **kwargs):
    step_size = kwargs.get("step_size", 0.2)
    step_sizes = torch.ones(dim, device=device) * step_size
    step = torch.empty(dim, device=device, dtype=dtype).uniform_(-1, 1) * step_sizes
    new_u = (u + step) % 1.0
    return new_u


def uniform(dim, device, dtype, u, **kwargs):
    return torch.rand_like(u)


def gaussian(dim, device, dtype, u, **kwargs):
    mean = kwargs.get("mean", torch.zeros_like(u))
    std = kwargs.get("std", torch.ones_like(u))
    return torch.normal(mean, std)


class MarkovChainMonteCarlo(Integrator):
    def __init__(
        self,
        bounds,
        f: Callable,
        f_dim: int = 1,
        maps=None,
        q0=None,
        proposal_dist=None,
        batch_size: int = 1000,
        nburnin: int = 10,
        device=None,
        dtype=None,
    ):
        super().__init__(bounds, f, f_dim, maps, q0, batch_size, device, dtype)
        self.nburnin = nburnin
        if not proposal_dist:
            self.proposal_dist = uniform
        else:
            if not isinstance(proposal_dist, Callable):
                raise TypeError("proposal_dist must be a callable function.")
            self.proposal_dist = proposal_dist
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def sample(self, config, nsteps=1, mix_rate=0.5, **kwargs):
        for _ in range(nsteps):
            proposed_y = self.proposal_dist(
                self.dim, self.device, self.dtype, config.u, **kwargs
            )
            proposed_x, new_detJ = self.maps.forward_with_detJ(proposed_y)

            new_weight = (
                mix_rate / new_detJ
                + (1 - mix_rate) * self.f(proposed_x, config.fx).abs()
            )
            new_weight.masked_fill_(new_weight < EPSILON, EPSILON)
            acceptance_probs = new_weight / config.weight * new_detJ / config.detJ

            accept = (
                torch.rand(self.batch_size, dtype=torch.float64, device=self.device)
                <= acceptance_probs
            )

            accept_expanded = accept.unsqueeze(1)
            config.u.mul_(~accept_expanded).add_(proposed_y * accept_expanded)
            config.x.mul_(~accept_expanded).add_(proposed_x * accept_expanded)
            config.weight.mul_(~accept).add_(new_weight * accept)
            config.detJ.mul_(~accept).add_(new_detJ * accept)

    def __call__(
        self,
        neval,
        mix_rate=0.5,
        nblock=32,
        meas_freq: int = 1,
        verbose=-1,
        **kwargs,
    ):
        neval = neval // self.world_size
        neval = -(-neval // self.batch_size) * self.batch_size
        epoch = neval // self.batch_size
        nsteps_perblock = epoch // nblock
        if nsteps_perblock == 0:
            warn(
                f"neval is too small to be divided into {nblock} blocks. Reset nblock to {epoch}."
            )
            nsteps_perblock = 1
            nblock = epoch
        else:
            nblock = epoch // nsteps_perblock
        n_meas_perblock = nsteps_perblock // meas_freq
        assert (
            n_meas_perblock > 0
        ), f"neval ({neval}) should be larger than batch_size * nblock * meas_freq ({self.batch_size} * {nblock} * {meas_freq})"

        if verbose > 0:
            print(
                f"nblock = {nblock}, n_meas_perblock = {n_meas_perblock}, meas_freq = {meas_freq}, batch_size = {self.batch_size}, actual neval = {self.batch_size*nsteps_perblock*nblock}"
            )

        config = Configuration(
            self.batch_size, self.dim, self.f_dim, self.device, self.dtype
        )
        config.u, config.detJ = self.q0.sample_with_detJ(self.batch_size)
        config.x, detj = self.maps.forward_with_detJ(config.u)
        config.detJ *= detj
        config.weight = (
            mix_rate / config.detJ + (1 - mix_rate) * self.f(config.x, config.fx).abs_()
        )
        config.weight.masked_fill_(config.weight < EPSILON, EPSILON)

        for _ in range(self.nburnin):
            self.sample(config, mix_rate=mix_rate, **kwargs)

        values = torch.zeros(
            (self.batch_size, self.f_dim), dtype=self.dtype, device=self.device
        )
        refvalues = torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)

        means = torch.zeros((nblock, self.f_dim), dtype=self.dtype, device=self.device)
        vars = torch.zeros_like(means)
        means_ref = torch.zeros((nblock, 1), dtype=self.dtype, device=self.device)
        vars_ref = torch.zeros_like(means_ref)

        for iblock in range(nblock):
            for _ in range(n_meas_perblock):
                self.sample(config, meas_freq, mix_rate, **kwargs)
                self.f(config.x, config.fx)

                config.fx.div_(config.weight.unsqueeze(1))
                values += config.fx / n_meas_perblock
                refvalues += 1 / (config.detJ * config.weight) / n_meas_perblock
            means[iblock, :] = values.mean(dim=0)
            vars[iblock, :] = values.var(dim=0) / self.batch_size
            means_ref[iblock, 0] = refvalues.mean()
            vars_ref[iblock, 0] = refvalues.var() / self.batch_size
            values.zero_()
            refvalues.zero_()

        results_unnorm = self.statistics(means, vars, nsteps_perblock * self.batch_size)
        results_ref = self.statistics(
            means_ref, vars_ref, nsteps_perblock * self.batch_size
        )

        if self.rank == 0:
            if self.f_dim == 1:
                return results_unnorm[0] / results_ref[0]
            else:
                return results_unnorm / results_ref
