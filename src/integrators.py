from typing import Callable
import torch
from utils import RAvg
from maps import Linear
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
        maps=None,
        bounds=None,
        q0=None,
        neval: int = 1000,
        nbatch: int = None,
        device="cpu",
        x_dtype=torch.float64,
        fx_dtype=torch.float64,
    ):
        self.x_dtype = x_dtype
        self.fx_dtype = fx_dtype
        if maps:
            if not self.x_dtype == maps.dtype:
                raise ValueError(
                    "Data type of the variables of integrator should be same as maps."
                )
            self.bounds = maps.bounds
        else:
            if not isinstance(bounds, (list, np.ndarray)):
                raise TypeError("bounds must be a list or a NumPy array.")
            self.bounds = torch.tensor(bounds, dtype=x_dtype, device=device)

        self.dim = len(self.bounds)
        if not q0:
            q0 = Uniform(self.bounds, device=device, dtype=x_dtype)
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
        x_dtype=torch.float64,
        fx_dtype=torch.float64,
    ):
        super().__init__(maps, bounds, q0, neval, nbatch, device, x_dtype, fx_dtype)

    def __call__(self, f: Callable, f_dim: int = 1, **kwargs):
        x, _ = self.sample(self.nbatch)
        fx = torch.empty((self.nbatch, f_dim), dtype=self.fx_dtype, device=self.device)

        epoch = self.neval // self.nbatch
        integ_values = torch.zeros(
            (self.nbatch, f_dim), dtype=self.fx_dtype, device=self.device
        )

        for _ in range(epoch):
            x, log_detJ = self.sample(self.nbatch)
            f(x, fx)
            fx.mul_(log_detJ.exp_().unsqueeze_(1))
            integ_values += fx / epoch

        results = np.array([RAvg() for _ in range(f_dim)])
        for i in range(f_dim):
            _mean = integ_values[:, i].mean().item()
            _var = integ_values[:, i].var().item() / self.nbatch
            results[i].update(_mean, _var, self.neval)
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
        x_dtype=torch.float64,
        fx_dtype=torch.float64,
    ):
        super().__init__(maps, bounds, q0, neval, nbatch, device, x_dtype, fx_dtype)
        self.nburnin = nburnin
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
        **kwargs,
    ):
        epsilon = 1e-16  # Small value to ensure numerical stability
        epoch = self.neval // self.nbatch
        current_y, current_jac = self.q0.sample(self.nbatch)
        current_x, detJ = self.maps.forward(current_y)
        current_jac += detJ
        current_jac.exp_()
        fx = torch.empty((self.nbatch, f_dim), dtype=self.fx_dtype, device=self.device)
        fx_weight = torch.empty(self.nbatch, dtype=self.fx_dtype, device=self.device)
        fx_weight[:] = f(current_x, fx)
        fx_weight.abs_()

        current_weight = mix_rate / current_jac + (1 - mix_rate) * fx_weight
        current_weight.masked_fill_(current_weight < epsilon, epsilon)

        n_meas = epoch // meas_freq

        def one_step(current_y, current_x, current_weight, current_jac):
            proposed_y = proposal_dist(
                self.dim, self.bounds, self.device, self.x_dtype, current_y, **kwargs
            )
            proposed_x, new_jac = self.maps.forward(proposed_y)
            new_jac.exp_()

            fx_weight[:] = f(proposed_x, fx)
            fx_weight.abs_()
            new_weight = mix_rate / new_jac + (1 - mix_rate) * fx_weight
            new_weight.masked_fill_(new_weight < epsilon, epsilon)

            acceptance_probs = new_weight / current_weight * new_jac / current_jac

            accept = (
                torch.rand(self.nbatch, dtype=torch.float64, device=self.device)
                <= acceptance_probs
            )

            current_y = torch.where(accept.unsqueeze(1), proposed_y, current_y)
            current_x = torch.where(accept.unsqueeze(1), proposed_x, current_x)
            current_weight = torch.where(accept, new_weight, current_weight)
            current_jac = torch.where(accept, new_jac, current_jac)
            return current_y, current_x, current_weight, current_jac

        for _ in range(self.nburnin):
            current_y, current_x, current_weight, current_jac = one_step(
                current_y, current_x, current_weight, current_jac
            )

        values = torch.zeros(
            (self.nbatch, f_dim), dtype=self.fx_dtype, device=self.device
        )
        refvalues = torch.zeros(self.nbatch, dtype=self.fx_dtype, device=self.device)

        for _ in range(n_meas):
            for _ in range(meas_freq):
                current_y, current_x, current_weight, current_jac = one_step(
                    current_y, current_x, current_weight, current_jac
                )
            f(current_x, fx)

            fx.div_(current_weight.unsqueeze(1))
            values += fx / n_meas
            refvalues += 1 / (current_jac * current_weight) / n_meas

        results = np.array([RAvg() for _ in range(f_dim)])
        results_ref = RAvg()

        mean_ref = refvalues.mean().item()
        var_ref = refvalues.var().item() / self.nbatch

        results_ref.update(mean_ref, var_ref, self.neval)
        for i in range(f_dim):
            _mean = values[:, i].mean().item()
            _var = values[:, i].var().item() / self.nbatch
            results[i].update(_mean, _var, self.neval)

        if f_dim == 1:
            res = results[0] / results_ref * self._rangebounds.prod()
            result = RAvg(itn_results=[res], sum_neval=self.neval)
            return result
        else:
            return results / results_ref * self._rangebounds.prod().item()
