from typing import Callable, Union, List, Tuple, Optional
import numpy as np
import torch
from .utils import to_tensor, from_tensor
from .vegas import AdaptiveMap

class MCMCIntegrator:
    def __init__(self,
                 ndim: int,
                 integrand: Callable,
                 prior: Callable,
                 proposal: Callable,
                 bounds: Union[List[Tuple[float, float]], np.ndarray],
                 discrete_dims: Optional[List[int]] = None,
                 use_gpu: bool = False):
        self.ndim = ndim
        self.integrand = integrand
        self.prior = prior
        self.proposal = proposal
        self.bounds = bounds
        self.discrete_dims = discrete_dims or []
        self.device = torch.device("cuda" if use_gpu else "cpu")
    
    def integrate(self, nsteps: int, burnin: int = 1000) -> torch.Tensor:
        chain = self.run_mcmc(nsteps + burnin)
        samples = chain[burnin:]
        values = to_tensor(self.integrand(from_tensor(samples)), self.device)
        return torch.mean(values, dim=0)

    def run_mcmc(self, nsteps: int) -> torch.Tensor:
        current_state = self.initialize_state()
        current_log_prob = self.log_prob(current_state)
        chain = []

        for _ in range(nsteps):
            proposed_state = self.propose(current_state)
            proposed_log_prob = self.log_prob(proposed_state)

            acceptance_ratio = torch.exp(proposed_log_prob - current_log_prob)
            if torch.rand(1, device=self.device) < acceptance_ratio:
                current_state = proposed_state
                current_log_prob = proposed_log_prob

            chain.append(current_state)

        return torch.stack(chain)

    def initialize_state(self) -> torch.Tensor:
        state = torch.rand(self.ndim, device=self.device)
        for dim in self.discrete_dims:
            state[dim] = torch.floor(state[dim] * (self.bounds[dim][1] - self.bounds[dim][0] + 1)) + self.bounds[dim][0]
        return state

    def propose(self, state: torch.Tensor) -> torch.Tensor:
        new_state = self.proposal(state)
        for dim in self.discrete_dims:
            new_state[dim] = torch.floor(new_state[dim])
        return torch.clamp(new_state, min=0, max=1)


    def log_prob(self, state: torch.Tensor) -> torch.Tensor:
        return torch.log(to_tensor(self.prior(from_tensor(state)), self.device)) + \
               torch.log(to_tensor(self.integrand(from_tensor(state)), self.device))

class VegasMCMCIntegrator(MCMCIntegrator):
    def __init__(self,
                 ndim: int,
                 integrand: Callable,
                 prior: Callable,
                 bounds: Union[List[Tuple[float, float]], np.ndarray],
                 discrete_dims: Optional[List[int]] = None,
                 use_gpu: bool = False):
        super().__init__(ndim, integrand, prior, self.vegas_proposal, bounds, discrete_dims, use_gpu)
        self.adaptive_map = AdaptiveMap(ndim)

    def vegas_proposal(self, state: torch.Tensor) -> torch.Tensor:
        return self.adaptive_map.map_points(torch.rand_like(state))[0]

    def adapt(self, neval: int, nitn: int = 5):
        for _ in range(nitn):
            samples = self.adaptive_map.map_points(torch.rand(neval, self.ndim, device=self.device))[0]
            values = to_tensor(self.integrand(from_tensor(samples)), self.device)
            self.adaptive_map.refine(samples, values)