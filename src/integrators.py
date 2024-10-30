from typing import Callable, Union, List, Tuple, Dict
import torch
from utils import RAvg
from maps import Map, Linear, CompositeMap
from base import Uniform
import gvar
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
                raise ValueError("Float type of maps should be same as integrator.")
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

    def __call__(self, f: Callable, **kwargs):
        x, _ = self.sample(1)
        f_values = f(x)
        if isinstance(f_values, (list, tuple)) and isinstance(
            f_values[0], torch.Tensor
        ):
            f_size = len(f_values)
            type_fval = f_values[0].dtype
        elif isinstance(f_values, torch.Tensor):
            f_size = 1
            type_fval = f_values.dtype
        else:
            raise TypeError(
                "f must return a torch.Tensor or a list/tuple of torch.Tensor."
            )

        epoch = self.neval // self.nbatch
        values = torch.zeros((self.nbatch, f_size), dtype=type_fval, device=self.device)

        for iepoch in range(epoch):
            x, log_detJ = self.sample(self.nbatch)
            f_values = f(x)
            batch_results = self._multiply_by_jacobian(f_values, torch.exp(log_detJ))

            values += batch_results / epoch

        results = np.array([RAvg() for _ in range(f_size)])
        for i in range(f_size):
            _mean = values[:, i].mean().item()
            _var = values[:, i].var().item() / self.nbatch
            results[i].update(_mean, _var, self.neval)
        if f_size == 1:
            return results[0]
        else:
            return results

    def gpu_run(self, f: Callable, **kwargs):
        x, _ = self.sample(1)
        f_values = f(x)
        if isinstance(f_values, (list, tuple)) and isinstance(
            f_values[0], torch.Tensor
        ):
            f_size = len(f_values)
            type_fval = f_values[0].dtype
        elif isinstance(f_values, torch.Tensor):
            f_size = 1
            type_fval = f_values.dtype
        else:
            raise TypeError(
                "f must return a torch.Tensor or a list/tuple of torch.Tensor."
            )

        epoch = self.neval // self.nbatch
        values = torch.zeros((self.nbatch, f_size), dtype=type_fval, device=self.device)

        for iepoch in range(epoch):
            x, log_detJ = self.sample(self.nbatch)
            f_values = f(x)
            batch_results = self._multiply_by_jacobian(f_values, torch.exp(log_detJ))

            values += batch_results / epoch

        results = np.array([RAvg() for _ in range(f_size)])
        for i in range(f_size):
            _total_mean = values[:, i].mean()
            _mean = _total_mean.detach().clone()
            dist.all_reduce(_total_mean, op=dist.ReduceOp.SUM)
            _total_mean /= dist.get_world_size()

            _var_between_batch = torch.square(_mean - _total_mean)
            dist.all_reduce(_var_between_batch, op=dist.ReduceOp.SUM)
            _var_between_batch /= dist.get_world_size()

            _var = values[:, i].var() / self.nbatch
            dist.all_reduce(_var, op=dist.ReduceOp.SUM)
            _var /= dist.get_world_size()
            results[i].update(
                _mean.item(), (_var + _var_between_batch).item(), self.neval
            )
        if f_size == 1:
            return results[0]
        else:
            return results

    def _multiply_by_jacobian(self, values, jac):
        # if isinstance(values, dict):
        #     return {k: v * torch.exp(log_det_J) for k, v in values.items()}
        if isinstance(values, (list, tuple)):
            return torch.stack([v * jac for v in values], dim=-1)
        else:
            return torch.stack([values * jac], dim=-1)


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
        neval=10000,
        nbatch=None,
        nburnin=500,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(maps, bounds, q0, neval, nbatch, device, dtype)
        self.nburnin = nburnin
        if maps is None:
            self.maps = Linear([(0, 1)] * self.dim, device=device)
        self._rangebounds = self.bounds[:, 1] - self.bounds[:, 0]

    def __call__(
        self,
        f: Callable,
        proposal_dist: Callable = uniform,
        thinning=1,
        mix_rate=0.5,
        **kwargs,
    ):
        epsilon = 1e-16  # Small value to ensure numerical stability
        epoch = self.neval // self.nbatch
        current_y, current_jac = self.q0.sample(self.nbatch)
        current_x, detJ = self.maps.forward(current_y)
        current_jac += detJ
        current_jac = torch.exp(current_jac)
        current_fval = f(current_x)
        if isinstance(current_fval, (list, tuple)) and isinstance(
            current_fval[0], torch.Tensor
        ):
            f_size = len(current_fval)
            current_fval = sum(current_fval)

            def _integrand(x):
                return sum(f(x))
        elif isinstance(current_fval, torch.Tensor):
            f_size = 1

            def _integrand(x):
                return f(x)
        else:
            raise TypeError(
                "f must return a torch.Tensor or a list/tuple of torch.Tensor."
            )
        type_fval = current_fval.dtype

        current_weight = mix_rate / current_jac + (1 - mix_rate) * current_fval.abs()
        current_weight.masked_fill_(current_weight < epsilon, epsilon)

        n_meas = epoch // thinning

        def one_step(current_y, current_x, current_weight, current_jac):
            proposed_y = proposal_dist(
                self.dim, self.bounds, self.device, self.dtype, current_y, **kwargs
            )
            proposed_x, new_jac = self.maps.forward(proposed_y)
            new_jac = torch.exp(new_jac)

            new_fval = _integrand(proposed_x)
            new_weight = mix_rate / new_jac + (1 - mix_rate) * new_fval.abs()
            new_weight.masked_fill_(new_weight < epsilon, epsilon)

            acceptance_probs = new_weight / current_weight * new_jac / current_jac

            accept = (
                torch.rand(self.nbatch, dtype=self.dtype, device=self.device)
                <= acceptance_probs
            )

            current_y = torch.where(accept.unsqueeze(1), proposed_y, current_y)
            # current_fval = torch.where(accept, new_fval, current_fval)
            current_x = torch.where(accept.unsqueeze(1), proposed_x, current_x)
            current_weight = torch.where(accept, new_weight, current_weight)
            current_jac = torch.where(accept, new_jac, current_jac)
            return current_y, current_x, current_weight, current_jac

        for i in range(self.nburnin):
            current_y, current_x, current_weight, current_jac = one_step(
                current_y, current_x, current_weight, current_jac
            )

        values = torch.zeros((self.nbatch, f_size), dtype=type_fval, device=self.device)
        refvalues = torch.zeros(self.nbatch, dtype=type_fval, device=self.device)

        for imeas in range(n_meas):
            for j in range(thinning):
                current_y, current_x, current_weight, current_jac = one_step(
                    current_y, current_x, current_weight, current_jac
                )

            batch_results = self._multiply_by_jacobian(
                f(current_x), 1.0 / current_weight
            )
            batch_results_ref = 1 / (current_jac * current_weight)

            values += batch_results / n_meas
            refvalues += batch_results_ref / n_meas

        results = np.array([RAvg() for _ in range(f_size)])
        results_ref = RAvg()

        mean_ref = refvalues.mean().item()
        var_ref = refvalues.var().item() / self.nbatch

        results_ref.update(mean_ref, var_ref, self.neval)
        for i in range(f_size):
            _mean = values[:, i].mean().item()
            _var = values[:, i].var().item() / self.nbatch
            results[i].update(_mean, _var, self.neval)

        if f_size == 1:
            res = results[0] / results_ref * self._rangebounds.prod()
            result = RAvg(itn_results=[res], sum_neval=self.neval)
            return result
        else:
            return results / results_ref * self._rangebounds.prod().item()
