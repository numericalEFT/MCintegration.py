import numpy as np
import torch
from torch import nn
from base import Uniform
import sys

TINY = 10 ** (sys.float_info.min_10_exp + 50)


class Map(nn.Module):
    def __init__(self, bounds, device="cpu", dtype=torch.float64):
        super().__init__()
        if isinstance(bounds, (list, np.ndarray)):
            self.bounds = torch.tensor(bounds, dtype=dtype, device=device)
        else:
            raise ValueError("Unsupported map specification")
        self.dim = self.bounds.shape[0]
        self.device = device
        self.dtype = dtype

    def forward(self, u):
        raise NotImplementedError("Subclasses must implement this method")

    def inverse(self, x):
        raise NotImplementedError("Subclasses must implement this method")


class CompositeMap(Map):
    def __init__(self, maps, device="cpu", dtype=torch.float64):
        if not maps:
            raise ValueError("Maps can not be empty.")
        super().__init__(maps[-1].bounds, device, dtype)
        self.maps = maps

    def forward(self, u):
        log_detJ = torch.zeros(len(u), device=u.device, dtype=self.dtype)
        for map in self.maps:
            u, log_detj = map.forward(u)
            log_detJ += log_detj
        return u, log_detJ

    def inverse(self, x):
        log_detJ = torch.zeros(len(x), device=x.device, dtype=self.dtype)
        for i in range(len(self.maps) - 1, -1, -1):
            x, log_detj = self.maps[i].inverse(x)
            log_detJ += log_detj
        return x, log_detJ


class Linear(Map):
    def __init__(self, bounds, device="cpu", dtype=torch.float64):
        super().__init__(bounds, device, dtype)
        self._A = self.bounds[:, 1] - self.bounds[:, 0]
        self._jac1 = torch.prod(self._A)

    def forward(self, u):
        return u * self._A + self.bounds[:, 0], torch.log(self._jac1.repeat(u.shape[0]))

    def inverse(self, x):
        return (x - self.bounds[:, 0]) / self._A, torch.log(
            self._jac1.repeat(x.shape[0])
        )


class Vegas(Map):
    def __init__(self, bounds, ninc=1000, alpha=0.5, device="cpu", dtype=torch.float64):
        super().__init__(bounds, device, dtype)
        # self.nbin = nbin
        self.alpha = alpha
        if isinstance(ninc, int):
            self.ninc = torch.ones(self.dim, dtype=torch.int32, device=device) * ninc
        else:
            self.ninc = torch.tensor(ninc, dtype=torch.int32, device=device)

        self.inc = torch.empty(
            self.dim, self.ninc.max(), dtype=self.dtype, device=self.device
        )
        self.grid = torch.empty(
            self.dim, self.ninc.max() + 1, dtype=self.dtype, device=self.device
        )

        self._A = self.bounds[:, 1] - self.bounds[:, 0]
        self._jaclinear = torch.prod(self._A)

        for d in range(self.dim):
            self.grid[d, : self.ninc[d] + 1] = torch.linspace(
                self.bounds[d, 0],
                self.bounds[d, 1],
                self.ninc[d] + 1,
                dtype=self.dtype,
                device=self.device,
            )
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )
        self.clear()

    def train(self, nsamples, f, epoch=5, alpha=0.5):
        q0 = Uniform(self.bounds, device=self.device, dtype=self.dtype)
        u, log_detJ0 = q0.sample(nsamples)

        fval = f(u)
        f_size = len(fval) if isinstance(fval, (list, tuple)) else 1
        if f_size > 1:

            def _integrand(x):
                return sum(f(x))
        else:

            def _integrand(x):
                return f(x)

        for _ in range(epoch):
            x, log_detJ = self.forward(u)
            f2 = torch.exp(2 * (log_detJ + log_detJ0)) * _integrand(x) ** 2
            self.add_training_data(u, f2)
            self.adapt(alpha)

    def add_training_data(self, u, fval):
        """Add training data ``f`` for ``u``-space points ``u``.

        Accumulates training data for later use by ``self.adapt()``.
        Grid increments will be made smaller in regions where
        ``f`` is larger than average, and larger where ``f``
        is smaller than average. The grid is unchanged (converged?)
        when ``f`` is constant across the grid.

        Args:
            u (tensor): ``u`` values corresponding to the training data.
                ``u`` is a contiguous 2-d tensor, where ``u[j, d]``
                is for points along direction ``d``.
            f (tensor): Training function values. ``f[j]`` corresponds to
                point ``u[j, d]`` in ``u``-space.
        """
        if self.sum_f is None:
            self.sum_f = torch.zeros_like(self.inc)
            self.n_f = torch.zeros_like(self.inc) + TINY
        iu = (u - self.bounds[:, 0]) / self._A * self.ninc
        iu = torch.floor(iu).long()
        for d in range(self.dim):
            indices = iu[:, d]
            self.sum_f[d].scatter_add_(0, indices, fval.abs())
            self.n_f[d].scatter_add_(0, indices, torch.ones_like(fval))

    def adapt(self, alpha=0.0):
        """Adapt grid to accumulated training data.

        ``self.adapt(...)`` projects the training data onto
        each axis independently and maps it into ``x`` space.
        It shrinks ``x``-grid increments in regions where the
        projected training data is large, and grows increments
        where the projected data is small. The grid along
        any direction is unchanged if the training data
        is constant along that direction.

        The number of increments along a direction can be
        changed by setting parameter ``ninc`` (array or number).

        The grid does not change if no training data has
        been accumulated, unless ``ninc`` is specified, in
        which case the number of increments is adjusted
        while preserving the relative density of increments
        at different values of ``x``.

        Args:
            alpha (float): Determines the speed with which the grid
                adapts to training data. Large (postive) values imply
                rapid evolution; small values (much less than one) imply
                slow evolution. Typical values are of order one. Choosing
                ``alpha<0`` causes adaptation to the unmodified training
                data (usually not a good idea).
        """
        new_grid = torch.empty(
            (self.dim, torch.max(self.ninc) + 1),
            dtype=torch.float64,
            device=self.device,
        )
        avg_f = torch.ones(self.inc.shape[1], dtype=torch.float64, device=self.device)
        if alpha > 0:
            tmp_f = torch.empty(
                self.inc.shape[1], dtype=torch.float64, device=self.device
            )
        for d in range(self.dim):
            ninc = self.ninc[d]
            if alpha != 0:
                if self.sum_f is not None:
                    mask = self.n_f[d, :] > 0
                    avg_f[mask] = self.sum_f[d, mask] / self.n_f[d, mask]
                    avg_f[~mask] = 0.0
                if alpha > 0:  # smooth
                    tmp_f[0] = torch.abs(7.0 * avg_f[0] + avg_f[1]) / 8.0
                    tmp_f[ninc - 1] = (
                        torch.abs(7.0 * avg_f[ninc - 1] + avg_f[ninc - 2]) / 8.0
                    )
                    tmp_f[1 : ninc - 1] = (
                        torch.abs(
                            6.0 * avg_f[1 : ninc - 1]
                            + avg_f[: ninc - 2]
                            + avg_f[2:ninc]
                        )
                        / 8.0
                    )
                    sum_f = torch.sum(tmp_f[:ninc])
                    if sum_f > 0:
                        avg_f[:ninc] = tmp_f[:ninc] / sum_f + TINY
                    else:
                        avg_f[:ninc] = TINY
                    avg_f[:ninc] = (
                        -(1 - avg_f[:ninc]) / torch.log(avg_f[:ninc])
                    ) ** alpha

            new_grid[d, 0] = self.grid[d, 0]
            new_grid[d, ninc] = self.grid[d, ninc]
            f_ninc = torch.sum(avg_f[:ninc]) / ninc

            j = -1
            acc_f = 0
            for i in range(1, ninc):
                while acc_f < f_ninc:
                    j += 1
                    if j < ninc:
                        acc_f += avg_f[j]
                    else:
                        break
                else:
                    acc_f -= f_ninc
                    new_grid[d, i] = (
                        self.grid[d, j + 1] - (acc_f / avg_f[j]) * self.inc[d, j]
                    )
                    continue
                break
        self.grid = new_grid
        self.inc = torch.empty(
            (self.dim, self.grid.shape[1] - 1), dtype=torch.float64, device=self.device
        )
        for d in range(self.dim):
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )
        self.clear()

    def extract_grid(self):
        "Return a list of lists specifying the map's grid."
        grid = []
        for d in range(self.dim):
            ng = self.ninc[d] + 1
            grid.append(self.grid[d, :ng].tolist())
        return grid

    def clear(self):
        "Clear information accumulated by :meth:`AdaptiveMap.add_training_data`."
        self.sum_f = None
        self.n_f = None

    @torch.no_grad()
    def forward(self, u):
        u = u.to(self.device)
        u_ninc = u * self.ninc
        # iu = torch.floor(u_ninc).long()
        iu = (u - self.bounds[:, 0]) / self._A * self.ninc
        iu = torch.floor(iu).long()
        du_ninc = u_ninc - torch.floor(u_ninc).long()

        x = torch.empty_like(u)
        jac = torch.ones(u.shape[0], device=x.device)
        # self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iu < ninc
            ninc = self.ninc[d]
            mask = iu[:, d] < ninc
            if mask.any():
                x[mask, d] = (
                    self.grid[d, iu[mask, d]]
                    + self.inc[d, iu[mask, d]] * du_ninc[mask, d]
                )
                jac[mask] *= self.inc[d, iu[mask, d]] * ninc

            # Handle the case where iu >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                x[mask_inv, d] = self.grid[d, ninc]
                jac[mask_inv] *= self.inc[d, ninc - 1] * ninc

        return x, torch.log(jac / self._jaclinear)

    @torch.no_grad()
    def inverse(self, x):
        # self.jac.fill_(1.0)
        x = x.to(self.device)
        u = torch.empty_like(x)
        jac = torch.ones(x.shape[0], device=x.device)
        for d in range(self.dim):
            ninc = self.ninc[d]
            iu = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iu > 0) & (iu <= ninc)
            mask_lower = iu <= 0
            mask_upper = iu > ninc

            # Handle valid range (0 < iu <= ninc)
            if mask_valid.any():
                iui_valid = iu[mask_valid] - 1
                u[mask_valid, d] = (
                    iui_valid
                    + (x[mask_valid, d] - self.grid[d, iui_valid])
                    / self.inc[d, iui_valid]
                ) / ninc
                jac[mask_valid] *= self.inc[d, iui_valid] * ninc

            # Handle lower bound (iu <= 0)\
            if mask_lower.any():
                u[mask_lower, d] = 0.0
                jac[mask_lower] *= self.inc[d, 0] * ninc

            # Handle upper bound (iu > ninc)
            if mask_upper.any():
                u[mask_upper, d] = 1.0
                jac[mask_upper] *= self.inc[d, ninc - 1] * ninc

        return u, torch.log(jac / self._jaclinear)


# class NormalizingFlow(Map):
#     def __init__(self, bounds, flow_model, device="cpu"):
#         super().__init__(bounds, device)
#         self.flow_model = flow_model.to(device)

#     def forward(self, u):
#         return self.flow_model.forward(u)

#     def inverse(self, x):
#         return self.flow_model.inverse(x)
