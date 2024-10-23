from typing import Callable, Union, List, Tuple, Dict
import torch
from utils import RAvg
from maps import Map, Linear, CompositeMap
from base import Uniform
import gvar
import numpy as np


class Integrator:
    """
    Base class for all integrators. This class is designed to handle integration tasks
    over a specified domain (bounds) using a sampling method (q0) and optional
    transformation maps.
    """

    def __init__(
        self,
        bounds,
        q0=None,
        maps=None,
        neval: int = 1000,
        nbatch: int = None,
        device="cpu",
        dtype=torch.float64,
    ):
        if not isinstance(bounds, (list, np.ndarray)):
            raise TypeError("bounds must be a list or a NumPy array.")
        self.dtype = dtype
        self.dim = len(bounds)
        if not q0:
            q0 = Uniform(bounds, device=device, dtype=dtype)
        self.bounds = torch.tensor(bounds, dtype=dtype, device=device)
        self.q0 = q0
        if maps:
            if not self.dtype == maps.dtype:
                raise ValueError("Float type of maps should be same as integrator.")
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
        bounds,
        q0=None,
        maps=None,
        neval: int = 1000,
        nbatch: int = None,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(bounds, q0, maps, neval, nbatch, device, dtype)

    def __call__(self, f: Callable, **kwargs):
        x, _ = self.sample(self.nbatch)
        f_values = f(x)
        f_size = len(f_values) if isinstance(f_values, (list, tuple)) else 1
        type_fval = f_values.dtype if f_size == 1 else f_values[0].dtype

        epoch = self.neval // self.nbatch
        mean_values = torch.zeros((f_size, epoch), dtype=type_fval, device=self.device)
        std_values = torch.zeros_like(mean_values)

        for iepoch in range(epoch):
            x, log_detJ = self.sample(self.nbatch)
            f_values = f(x)
            batch_results = self._multiply_by_jacobian(f_values, torch.exp(log_detJ))

            mean_values[:, iepoch] = torch.mean(batch_results, dim=-1)
            std_values[:, iepoch] = torch.std(batch_results, dim=-1) / self.nbatch**0.5

        results = np.array([RAvg() for _ in range(f_size)])
        for iepoch in range(epoch):
            for j in range(f_size):
                results[j].sum_neval += self.nbatch
                results[j].add(
                    gvar.gvar(
                        mean_values[j, iepoch].item(), std_values[j, iepoch].item()
                    )
                )
        if f_size == 1:
            return results[0]
        else:
            return results

    def _multiply_by_jacobian(self, values, jac):
        # if isinstance(values, dict):
        #     return {k: v * torch.exp(log_det_J) for k, v in values.items()}
        if isinstance(values, (list, tuple)):
            return torch.stack([v * jac for v in values])
        else:
            return values * jac


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
        bounds,
        q0=None,
        maps=None,
        neval=10000,
        nbatch=None,
        nburnin=500,
        device="cpu",
        dtype=torch.float64,
    ):
        super().__init__(bounds, q0, maps, neval, nbatch, device, dtype)
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
        # vars_shape = (self.nbatch, self.dim)
        current_y, current_jac = self.q0.sample(self.nbatch)
        current_x, detJ = self.maps.forward(current_y)
        current_jac += detJ
        current_jac = torch.exp(current_jac)
        current_fval = f(current_x)
        f_size = len(current_fval) if isinstance(current_fval, (list, tuple)) else 1

        if f_size > 1:
            current_fval = sum(current_fval)

            def _integrand(x):
                return sum(f(x))
        else:

            def _integrand(x):
                return f(x)

        type_fval = current_fval.dtype

        current_weight = mix_rate / current_jac + (1 - mix_rate) * current_fval.abs()
        current_weight.masked_fill_(current_weight < epsilon, epsilon)

        n_meas = epoch // thinning
        mean_values = torch.zeros((f_size, n_meas), dtype=type_fval, device=self.device)
        std_values = torch.zeros_like(mean_values)
        mean_refvalues = torch.zeros(n_meas, dtype=type_fval, device=self.device)
        std_refvalues = torch.zeros_like(mean_refvalues)

        def _propose(current_y, current_x, current_weight, current_jac):
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
            current_y, current_x, current_weight, current_jac = _propose(
                current_y, current_x, current_weight, current_jac
            )

        for imeas in range(n_meas):
            for j in range(thinning):
                current_y, current_x, current_weight, current_jac = _propose(
                    current_y, current_x, current_weight, current_jac
                )

            batch_results = self._multiply_by_jacobian(
                f(current_x), 1.0 / current_weight
            )
            batch_results_ref = 1 / (current_jac * current_weight)

            mean_values[:, imeas] = torch.mean(batch_results, dim=-1)
            std_values[:, imeas] = torch.std(batch_results, dim=-1) / self.nbatch**0.5

            mean_refvalues[imeas] = torch.mean(batch_results_ref, dim=-1)
            std_refvalues[imeas] = (
                torch.std(batch_results_ref, dim=-1) / self.nbatch**0.5
            )

        results = np.array([RAvg() for _ in range(f_size)])
        results_ref = RAvg()
        for imeas in range(n_meas):
            results_ref.sum_neval += self.nbatch
            results_ref.add(
                gvar.gvar(mean_refvalues[imeas].item(), std_refvalues[imeas].item())
            )
            for j in range(f_size):
                results[j].sum_neval += self.nbatch
                results[j].add(
                    gvar.gvar(mean_values[j, imeas].item(), std_values[j, imeas].item())
                )
        if f_size == 1:
            return results[0] / results_ref * self._rangebounds.prod()
        else:
            return results / results_ref * self._rangebounds.prod().item()
