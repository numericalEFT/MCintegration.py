import torch
import numpy as np


class Map:
    def __init__(self, map_spec, device="cpu"):
        # if isinstance(map_spec, dict):
        #     self.map_spec = {
        #         k: torch.tensor(v, device=device) for k, v in map_spec.items()
        #     }
        if isinstance(map_spec, (list, np.ndarray)):
            self.map_spec = torch.tensor(map_spec, dtype=torch.float64, device=device)
        else:
            raise ValueError("Unsupported map specification")
        self.dim = self.map_spec.shape[0]
        self.device = device

    def forward(self, y):
        raise NotImplementedError("Subclasses must implement this method")

    def inverse(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def log_det_jacobian(self, y):
        raise NotImplementedError("Subclasses must implement this method")


class Affine(Map):
    def __init__(self, map_spec, device="cpu"):
        super().__init__(map_spec, device)
        self._A = self.map_spec[:, 1] - self.map_spec[:, 0]
        self._jac1 = torch.prod(self._A)

    def forward(self, y):
        return y * self._A + self.map_spec[:, 0], self._jac1.repeat(y.shape[0])

    def inverse(self, x):
        return (x - self.map_spec[:, 0]) / self._A, self._jac1.repeat(x.shape[0])

    def log_det_jacobian(self, y):
        return torch.log(self._jac1) * y.shape[0]


class AdaptiveMap(Map):
    def __init__(self, map_spec, alpha=0.5, device="cpu"):
        super().__init__(map_spec, device)
        self.alpha = alpha

    def add_training_data(self, y, f):
        pass

    def adapt(self, alpha=0.0):
        pass


class Vegas(AdaptiveMap):
    def __init__(self, map_spec, ninc=1000, alpha=0.5, device="cpu"):
        super().__init__(map_spec, alpha, device)
        # self.nbin = nbin

        if isinstance(ninc, int):
            self.ninc = torch.ones(self.dim, dtype=torch.int32, device=device) * ninc
        else:
            self.ninc = torch.tensor(ninc, dtype=torch.int32, device=device)

        self.inc = torch.empty(
            self.dim, self.ninc.max(), dtype=torch.float64, device=self.device
        )
        self.grid = torch.empty(
            self.dim, self.ninc.max() + 1, dtype=torch.float64, device=self.device
        )

        for d in range(self.dim):
            self.grid[d, : self.ninc[d] + 1] = torch.linspace(
                self.map_spec[d, 0],
                self.map_spec[d, 1],
                self.ninc[d] + 1,
                dtype=torch.float64,
                device=self.device,
            )
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )
        self.clear()

    def add_training_data(self, y, f):
        """Add training data ``f`` for ``y``-space points ``y``.

        Accumulates training data for later use by ``self.adapt()``.
        Grid increments will be made smaller in regions where
        ``f`` is larger than average, and larger where ``f``
        is smaller than average. The grid is unchanged (converged?)
        when ``f`` is constant across the grid.

        Args:
            y (tensor): ``y`` values corresponding to the training data.
                ``y`` is a contiguous 2-d tensor, where ``y[j, d]``
                is for points along direction ``d``.
            f (tensor): Training function values. ``f[j]`` corresponds to
                point ``y[j, d]`` in ``y``-space.
        """
        if self.sum_f is None:
            self.sum_f = torch.zeros_like(self.inc)
            self.n_f = torch.zeros_like(self.inc) + 1e-10
        iy = torch.floor(y * self.ninc).long()
        for d in range(self.dim):
            self.sum_f[d, iy[:, d]] += torch.abs(f)
            self.n_f[d, iy[:, d]] += 1

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
                        avg_f[:ninc] = tmp_f[:ninc] / sum_f + 1e-10
                    else:
                        avg_f[:ninc] = 1e-10
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
    def forward(self, y):
        y = y.to(self.device)
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        x = torch.empty_like(y)
        jac = torch.ones(y.shape[0], device=x.device)
        # self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            ninc = self.ninc[d]
            mask = iy[:, d] < ninc
            if mask.any():
                x[mask, d] = (
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                jac[mask] *= self.inc[d, iy[mask, d]] * ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                x[mask_inv, d] = self.grid[d, ninc]
                jac[mask_inv] *= self.inc[d, ninc - 1] * ninc

        return x, jac

    @torch.no_grad()
    def inverse(self, x):
        # self.jac.fill_(1.0)
        x = x.to(self.device)
        y = torch.empty_like(x)
        jac = torch.ones(x.shape[0], device=x.device)
        for d in range(self.dim):
            ninc = self.ninc[d]
            iy = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iy > 0) & (iy <= ninc)
            mask_lower = iy <= 0
            mask_upper = iy > ninc

            # Handle valid range (0 < iy <= ninc)
            if mask_valid.any():
                iyi_valid = iy[mask_valid] - 1
                y[mask_valid, d] = (
                    iyi_valid
                    + (x[mask_valid, d] - self.grid[d, iyi_valid])
                    / self.inc[d, iyi_valid]
                ) / ninc
                jac[mask_valid] *= self.inc[d, iyi_valid] * ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                y[mask_lower, d] = 0.0
                jac[mask_lower] *= self.inc[d, 0] * ninc

            # Handle upper bound (iy > ninc)
            if mask_upper.any():
                y[mask_upper, d] = 1.0
                jac[mask_upper] *= self.inc[d, ninc - 1] * ninc

        return y, jac

    @torch.no_grad()
    def log_det_jacobian(self, y):
        y = y.to(self.device)
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()

        jac = torch.ones(y.shape[0], device=x.device)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return torch.sum(torch.log(jac), dim=-1)


class NormalizingFlow(AdaptiveMap):
    def __init__(self, map_spec, flow_model, alpha=0.5, device="cpu"):
        super().__init__(map_spec, alpha, device)
        self.flow_model = flow_model.to(device)

    def forward(self, u):
        return self.flow_model.forward(u)[0]

    def inverse(self, x):
        return self.flow_model.inverse(x)[0]

    def log_det_jacobian(self, u):
        return self.flow_model.forward(u)[1]
