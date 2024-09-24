from typing import Callable, Union, List, Tuple, Dict
import torch
from utils import RAvg
from maps import Map, Affine
import gvar


class Integrator:
    """
    Base class for all integrators.
    """

    def __init__(
        self,
        # bounds: Union[List[Tuple[float, float]], np.ndarray],
        map,
        neval: int = 1000,
        batch_size: int = None,
        device="cpu",
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        if not isinstance(map, Map):
            map = Affine(map)

        self.dim = map.dim
        self.map = map
        self.neval = neval
        if batch_size is None:
            self.batch_size = neval
            self.neval = neval
        else:
            self.batch_size = batch_size
            self.neval = -(-neval // batch_size) * batch_size

        self.device = device

    def __call__(self, f: Callable, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class MonteCarlo(Integrator):
    def __init__(
        self,
        map,
        nitn: int = 10,
        neval: int = 1000,
        batch_size: int = None,
        device="cpu",
        adapt=False,
        alpha=0.5,
    ):
        super().__init__(map, neval, batch_size, device)
        self.adapt = adapt
        self.alpha = alpha
        self.nitn = nitn

    def __call__(self, f: Callable, **kwargs):
        u = torch.rand(self.batch_size, self.dim, device=self.device)
        x, _ = self.map.forward(u)
        f_values = f(x)
        f_size = len(f_values) if isinstance(f_values, (list, tuple)) else 1
        type_fval = f_values.dtype if f_size == 1 else type(f_values[0].dtype)

        mean = torch.zeros(f_size, dtype=type_fval, device=self.device)
        var = torch.zeros(f_size, dtype=type_fval, device=self.device)
        # var = torch.zeros((f_size, f_size), dtype=type_fval, device=self.device)

        result = RAvg(weighted=self.adapt)
        neval_batch = self.neval // self.batch_size

        for itn in range(self.nitn):
            mean[:] = 0
            var[:] = 0
            for _ in range(neval_batch):
                y = torch.rand(
                    self.batch_size, self.dim, dtype=torch.float64, device=self.device
                )
                x, jac = self.map.forward(y)

                f_values = f(x)
                batch_results = self._multiply_by_jacobian(f_values, jac)

                mean += torch.mean(batch_results, dim=-1) / neval_batch
                var += torch.var(batch_results, dim=-1) / (self.neval * neval_batch)

                if self.adapt:
                    self.map.add_training_data(y, batch_results**2)

            result.sum_neval += self.neval
            result.add(gvar.gvar(mean.item(), (var**0.5).item()))
            if self.adapt:
                self.map.adapt(alpha=self.alpha)

        return result

    def _multiply_by_jacobian(self, values, jac):
        # if isinstance(values, dict):
        #     return {k: v * torch.exp(log_det_J) for k, v in values.items()}
        if isinstance(values, (list, tuple)):
            return torch.stack([v * jac for v in values])
        else:
            return values * jac


class MCMC(MonteCarlo):
    def __init__(
        self,
        map: Map,
        nitn: int = 10,
        neval=10000,
        batch_size=None,
        n_burnin=500,
        device="cpu",
        adapt=False,
        alpha=0.5,
    ):
        super().__init__(map, nitn, neval, batch_size, device, adapt, alpha)
        self.n_burnin = n_burnin

    def __call__(
        self,
        f: Callable,
        proposal_dist="global_uniform",
        thinning=1,
        mix_rate=0.0,
        **kwargs,
    ):
        epsilon = 1e-16  # Small value to ensure numerical stability
        vars_shape = (self.batch_size, self.dim)
        current_y = torch.rand(vars_shape, dtype=torch.float64, device=self.device)
        current_x, current_jac = self.map.forward(current_y)
        current_prob = f(current_x)
        current_weight = mix_rate / current_jac + (1 - mix_rate) * current_prob.abs()
        current_weight.masked_fill_(current_weight < epsilon, epsilon)
        # current_prob.masked_fill_(current_prob.abs() < epsilon, epsilon)

        proposed_y = torch.empty_like(current_y)
        proposed_x = torch.empty_like(proposed_y)
        new_prob = torch.empty_like(current_prob)
        new_weight = torch.empty_like(current_weight)

        f_size = len(current_prob) if isinstance(current_prob, (list, tuple)) else 1
        type_fval = current_prob.dtype if f_size == 1 else type(current_prob[0].dtype)
        mean = torch.zeros(f_size, dtype=type_fval, device=self.device)
        mean_ref = torch.zeros_like(mean)
        var = torch.zeros(f_size, dtype=type_fval, device=self.device)
        var_ref = torch.zeros_like(mean)

        result = RAvg(weighted=self.adapt)
        result_ref = RAvg(weighted=self.adapt)

        neval_batch = self.neval // self.batch_size
        n_meas = 0
        for itn in range(self.nitn):
            for i in range(self.neval // self.batch_size):
                proposed_y[:] = self._propose(current_y, proposal_dist, **kwargs)
                proposed_x[:], new_jac = self.map.forward(proposed_y)

                new_prob[:] = f(proposed_x)
                new_weight = mix_rate / new_jac + (1 - mix_rate) * new_prob.abs()

                acceptance_probs = new_weight / current_weight * new_jac / current_jac

                accept = (
                    torch.rand(self.batch_size, dtype=torch.float64, device=self.device)
                    <= acceptance_probs
                )

                current_y = torch.where(accept.unsqueeze(1), proposed_y, current_y)
                current_prob = torch.where(accept, new_prob, current_prob)
                current_weight = torch.where(accept, new_weight, current_weight)
                current_jac = torch.where(accept, new_jac, current_jac)

                if i < self.n_burnin and (self.adapt or itn == 0):
                    continue
                elif i % thinning == 0:
                    n_meas += 1
                    batch_results = current_prob / current_weight

                    mean += torch.mean(batch_results, dim=-1) / neval_batch
                    var += torch.var(batch_results, dim=-1) / neval_batch

                    batch_results_ref = 1 / (current_jac * current_weight)
                    mean_ref += torch.mean(batch_results_ref, dim=-1) / neval_batch
                    var_ref += torch.var(batch_results_ref, dim=-1) / neval_batch

                    if self.adapt:
                        self.map.add_training_data(
                            current_y, (current_prob * current_jac) ** 2
                        )
            result.sum_neval += self.neval
            result.add(gvar.gvar(mean.item(), ((var / n_meas) ** 0.5).item()))
            result_ref.sum_neval += self.batch_size
            result_ref.add(
                gvar.gvar(mean_ref.item(), ((var_ref / n_meas) ** 0.5).item())
            )

            if self.adapt:
                self.map.adapt(alpha=self.alpha)

        return result / result_ref

    def _propose(self, u, proposal_dist, **kwargs):
        if proposal_dist == "random_walk":
            step_size = kwargs.get("step_size", 0.2)
            return (u + (torch.rand_like(u) - 0.5) * step_size) % 1.0
        elif proposal_dist == "global_uniform":
            return torch.rand_like(u)
        # elif proposal_dist == "gaussian":
        #     mean = kwargs.get("mean", torch.zeros_like(u))
        #     std = kwargs.get("std", torch.ones_like(u))
        #     return torch.normal(mean, std)
        else:
            raise ValueError(f"Unknown proposal distribution: {proposal_dist}")
