import numpy as np
import torch
from torch import nn
from MCintegration.base import Uniform
from MCintegration.utils import get_device
import sys

TINY = 10 ** (sys.float_info.min_10_exp + 50)


class Configuration:
    def __init__(self, batch_size, dim, f_dim, device=None, dtype=torch.float32):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.dim = dim
        self.f_dim = f_dim
        self.batch_size = batch_size
        self.u = torch.empty((batch_size, dim), dtype=dtype, device=self.device)
        self.x = torch.empty((batch_size, dim), dtype=dtype, device=self.device)
        self.fx = torch.empty((batch_size, f_dim), dtype=dtype, device=self.device)
        self.weight = torch.empty((batch_size,), dtype=dtype, device=self.device)
        self.detJ = torch.empty((batch_size,), dtype=dtype, device=self.device)


class Map(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.dtype = dtype

    def forward(self, u):
        raise NotImplementedError("Subclasses must implement this method")

    def forward_with_detJ(self, u):
        u, detJ = self.forward(u)
        detJ.exp_()
        return u, detJ

    def inverse(self, x):
        raise NotImplementedError("Subclasses must implement this method")


class CompositeMap(Map):
    def __init__(self, maps, device=None, dtype=None):
        if not maps:
            raise ValueError("Maps can not be empty.")
        if dtype is None:
            dtype = maps[-1].dtype
        if device is None:
            device = maps[-1].device
        elif device != maps[-1].device:
            for map in maps:
                map.to(device)
        super().__init__(device, dtype)
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


class Vegas(Map):
    def __init__(self, dim, ninc=1000, device=None, dtype=torch.float32):
        super().__init__(device, dtype)

        self.dim = dim
        # Ensure ninc is a tensor of appropriate shape and type
        if isinstance(ninc, int):
            self.ninc = torch.full(
                (self.dim,), ninc, dtype=torch.int32, device=self.device
            )
        elif isinstance(ninc, (list, np.ndarray)):
            self.ninc = torch.tensor(ninc, dtype=torch.int32, device=self.device)
        elif isinstance(ninc, torch.Tensor):
            self.ninc = ninc.to(dtype=torch.int32, device=self.device)
        else:
            raise ValueError(
                "'ninc' must be an int, list, numpy array, or torch tensor."
            )

        # Ensure ninc has the correct shape
        if self.ninc.shape != (self.dim,):
            raise ValueError(
                f"'ninc' must be a scalar or a 1D array of length {self.dim}."
            )

        # Preallocate tensors to minimize memory allocations
        self.max_ninc = self.ninc.max().item()
        # self.new_grid = torch.empty(
        #     (self.dim, self.max_ninc + 1), dtype=self.dtype, device=self.device
        # )
        # self.sum_f = None
        # self.n_f = None

        # Preallocate temporary tensors for adapt
        # self.avg_f = torch.ones(self.max_ninc, dtype=self.dtype, device=self.device)
        # self.tmp_f = torch.empty(self.max_ninc, dtype=self.dtype, device=self.device)
        self.avg_f = torch.ones(
            (self.dim, self.max_ninc), dtype=self.dtype, device=self.device
        )
        self.tmp_f = torch.zeros(
            (self.dim, self.max_ninc), dtype=self.dtype, device=self.device
        )

        self.make_uniform()
        self.sum_f = torch.zeros_like(self.inc)
        # self.n_f = torch.full_like(self.inc, TINY)
        self.n_f = torch.zeros_like(self.inc)

    def adaptive_training(
        self,
        batch_size,
        f,
        f_dim=1,
        epoch=10,
        alpha=0.5,
    ):
        """
            Perform adaptive training to adjust the grid based on the training function.

        Args:
            batch_size (int): Number of samples per batch.
            f (callable): Training function that takes x and fx as inputs.
            f_dim (int, optional): Dimension of the function f. Defaults to 1.
            epoch (int, optional): Number of training epochs. Defaults to 10.
            alpha (float, optional): Adaptation rate. Defaults to 0.5.
        """
        q0 = Uniform(self.dim, device=self.device, dtype=self.dtype)
        sample = Configuration(
            batch_size, self.dim, f_dim, device=self.device, dtype=self.dtype
        )

        for _ in range(epoch):
            sample.u, log_detJ0 = q0.sample(batch_size)
            sample.x[:], log_detJ = self.forward(sample.u)
            sample.weight = f(sample.x, sample.fx)
            sample.detJ = torch.exp(log_detJ0 + log_detJ)
            self.add_training_data(sample)
            self.adapt(alpha)

    @torch.no_grad()
    def add_training_data(self, sample):
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
        fval = (sample.detJ * sample.weight) ** 2
        # if self.sum_f is None:
        #     self.sum_f = torch.zeros_like(self.inc)
        #     self.n_f = torch.full_like(self.inc, TINY)
        iu = torch.floor(sample.u * self.ninc).long()
        for d in range(self.dim):
            indices = iu[:, d]
            self.sum_f[d].scatter_add_(0, indices, fval.abs())
            self.n_f[d].scatter_add_(0, indices, torch.ones_like(fval))

    # @torch.no_grad()
    # def adapt(self, alpha=0.0):
    #     """Adapt grid to accumulated training data.

    #     ``self.adapt(...)`` projects the training data onto
    #     each axis independently and maps it into ``x`` space.
    #     It shrinks ``x``-grid increments in regions where the
    #     projected training data is large, and grows increments
    #     where the projected data is small. The grid along
    #     any direction is unchanged if the training data
    #     is constant along that direction.

    #     The number of increments along a direction can be
    #     changed by setting parameter ``ninc`` (array or number).

    #     The grid does not change if no training data has
    #     been accumulated, unless ``ninc`` is specified, in
    #     which case the number of increments is adjusted
    #     while preserving the relative density of increments
    #     at different values of ``x``.

    #     Args:
    #         alpha (float): Determines the speed with which the grid
    #             adapts to training data. Large (postive) values imply
    #             rapid evolution; small values (much less than one) imply
    #             slow evolution. Typical values are of order one. Choosing
    #             ``alpha<0`` causes adaptation to the unmodified training
    #             data (usually not a good idea).
    #     """
    #     if torch.distributed.is_initialized():
    #         torch.distributed.all_reduce(self.sum_f, op=torch.distributed.ReduceOp.SUM)
    #         torch.distributed.all_reduce(self.n_f, op=torch.distributed.ReduceOp.SUM)

    #     mask = self.n_f > 0
    #     self.avg_f = torch.where(
    #         mask, self.sum_f / self.n_f, torch.zeros_like(self.sum_f)
    #     )

    #     if alpha > 0:
    #         # Smooth avg_f using a convolution-like operation
    #         # Pad avg_f for boundary conditions
    #         avg_f_padded = torch.nn.functional.pad(
    #             self.avg_f, (1, 1), mode="replicate"
    #         )  # Shape: (dim, max_ninc +2)

    #         # Apply smoothing kernel: [6,1,1] and [1,6,1], normalized by 8
    #         self.tmp_f[:, :] = (
    #             6.0 * self.avg_f + avg_f_padded[:, :-2] + avg_f_padded[:, 2:]
    #         ).abs_() / 8.0
    #         self.tmp_f[:, 0] = (7.0 * self.avg_f[:, 0] + self.avg_f[:, 1]).abs_() / 8.0
    #         self.tmp_f[:, -1] = (
    #             7.0 * self.avg_f[:, -1] + self.avg_f[:, -2]
    #         ).abs_() / 8.0

    #         # Normalize tmp_f
    #         sum_f = self.tmp_f.sum(dim=1, keepdim=True).clamp_min_(TINY)
    #         self.avg_f = self.tmp_f / sum_f + TINY

    #         # Apply transformation
    #         self.avg_f = (-(1 - self.avg_f) / torch.log(self.avg_f)).pow_(alpha)

    #     # Initialize new_grid with the first point as 0.0
    #     new_grid = torch.zeros(
    #         (self.dim, self.max_ninc + 1), dtype=self.dtype, device=self.device
    #     )
    #     new_grid[:, -1] = 1.0  # Set the last element to 1.0 for all dimensions

    #     # Iterate over each dimension to compute new grid points
    #     for d in range(self.dim):
    #         ninc_d = self.ninc[d].item()

    #         if alpha != 0 and self.sum_f is not None:
    #             # Compute f_ninc
    #             f_ninc = self.avg_f[d, :ninc_d].sum() / ninc_d  # Scalar

    #             # Initialize variables for grid point placement
    #             j = -1
    #             acc_f = 0.0

    #             for i in range(1, ninc_d):
    #                 while acc_f < f_ninc:
    #                     j += 1
    #                     if j < ninc_d:
    #                         acc_f += self.avg_f[d, j].item()
    #                     else:
    #                         break
    #                 else:
    #                     acc_f -= f_ninc
    #                     # Compute new_grid[d, i] based on current j and acc_f
    #                     # Ensure j+1 does not exceed ninc
    #                     if j + 1 < ninc_d:
    #                         new_grid[d, i] = (
    #                             self.grid[d, j + 1]
    #                             - (acc_f / self.avg_f[d, j]) * self.inc[d, j]
    #                         )
    #                     else:
    #                         new_grid[d, i] = self.grid[
    #                             d, j
    #                         ]  # Fallback to existing grid point
    #                     continue
    #                 break  # Exit the loop if j >= ninc_d

    #     # Assign the newly computed grid
    #     self.grid = new_grid

    #     # Update increments based on the new grid
    #     self.inc.zero_()
    #     self.inc[:, : self.max_ninc] = (
    #         self.grid[:, 1 : self.max_ninc + 1] - self.grid[:, : self.max_ninc]
    #     )

    #     # Reset training data
    #     self.clear()

    @torch.no_grad()
    def adapt(self, alpha=0.5):
        """
        Adapt the grid based on accumulated training data.

        Shrinks grid increments in regions where the accumulated f is large,
        and grows them where f is small. The adaptation speed is controlled by alpha.

        Args:
            alpha (float, optional): Determines the speed with which the grid
                adapts to training data. Large (positive) values imply
                rapid evolution; small values (much less than one) imply
                slow evolution. Typical values are of order one. Choosing
                ``alpha<0`` causes adaptation to the unmodified training
                data (usually not a good idea).
        """
        # Aggregate training data across distributed processes if applicable
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sum_f, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.n_f, op=torch.distributed.ReduceOp.SUM)

        # Initialize a new grid tensor
        new_grid = torch.empty(
            (self.dim, self.max_ninc + 1), dtype=self.dtype, device=self.device
        )

        if alpha > 0:
            tmp_f = torch.empty(self.max_ninc, dtype=self.dtype, device=self.device)

        # avg_f = torch.ones(self.inc.shape[1], dtype=self.dtype, device=self.device)
        for d in range(self.dim):
            ninc = self.ninc[d].item()

            if alpha != 0:
                # Compute average f for current dimension where n_f > 0
                mask = self.n_f[d, :ninc] > 0  # Shape: (ninc,)
                avg_f = torch.where(
                    mask,
                    self.sum_f[d, :ninc] / self.n_f[d, :ninc],
                    torch.zeros_like(self.sum_f[d, :ninc]),
                )  # Shape: (ninc,)

                if alpha > 0:
                    # Smooth avg_f
                    tmp_f[0] = (7.0 * avg_f[0] + avg_f[1]).abs() / 8.0  # Shape: ()
                    tmp_f[ninc - 1] = (
                        7.0 * avg_f[ninc - 1] + avg_f[ninc - 2]
                    ).abs() / 8.0  # Shape: ()
                    tmp_f[1 : ninc - 1] = (
                        6.0 * avg_f[1 : ninc - 1] + avg_f[: ninc - 2] + avg_f[2:ninc]
                    ).abs() / 8.0

                    # Normalize tmp_f to ensure the sum is 1
                    sum_f = torch.sum(tmp_f[:ninc]).clamp_min_(TINY)  # Scalar
                    avg_f = tmp_f[:ninc] / sum_f + TINY  # Shape: (ninc,)

                    # Apply non-linear transformation controlled by alpha
                    avg_f = (-(1 - avg_f) / torch.log(avg_f)).pow_(
                        alpha
                    )  # Shape: (ninc,)

                # Compute the target accumulated f per increment
                f_ninc = avg_f.sum() / ninc  # Scalar

                # Initialize variables for grid point placement
                j = -1
                acc_f = 0.0

                new_grid[d, 0] = self.grid[d, 0]
                new_grid[d, ninc] = self.grid[d, ninc]
                for i in range(1, ninc):
                    while acc_f < f_ninc:
                        j += 1
                        if j < ninc:
                            acc_f += avg_f[j]
                        else:
                            break
                    else:
                        acc_f -= f_ninc
                        # Place the new grid point based on accumulated f
                        new_grid[d, i] = (
                            self.grid[d, j + 1] - (acc_f / avg_f[j]) * self.inc[d, j]
                        )
                        continue
                    break  # Exit the loop if j >= ninc

            else:
                # If alpha == 0 or no training data, retain the existing grid
                new_grid[d, :] = self.grid[d, :]

        # Assign the newly computed grid
        self.grid = new_grid

        # Update increments based on the new grid
        # Compute the difference between consecutive grid points
        self.inc.zero_()  # Reset increments to zero
        for d in range(self.dim):
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )

        # Clear accumulated training data for the next adaptation cycle
        self.clear()

    @torch.no_grad()
    def adaptv0(self, alpha=0.0):
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
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sum_f, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.n_f, op=torch.distributed.ReduceOp.SUM)
        new_grid = torch.empty(
            (self.dim, torch.max(self.ninc) + 1),
            dtype=self.dtype,
            device=self.device,
        )
        avg_f = torch.ones(self.inc.shape[1], dtype=self.dtype, device=self.device)
        if alpha > 0:
            tmp_f = torch.empty(self.inc.shape[1], dtype=self.dtype, device=self.device)
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
            (self.dim, self.grid.shape[1] - 1),
            dtype=self.dtype,
            device=self.device,
        )
        for d in range(self.dim):
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )
        self.clear()

    @torch.no_grad()
    def make_uniform(self):
        self.inc = torch.empty(
            self.dim, self.max_ninc, dtype=self.dtype, device=self.device
        )
        self.grid = torch.empty(
            self.dim, self.max_ninc + 1, dtype=self.dtype, device=self.device
        )

        for d in range(self.dim):
            self.grid[d, : self.ninc[d] + 1] = torch.linspace(
                0,
                1,
                self.ninc[d] + 1,
                dtype=self.dtype,
                device=self.device,
            )
            self.inc[d, : self.ninc[d]] = (
                self.grid[d, 1 : self.ninc[d] + 1] - self.grid[d, : self.ninc[d]]
            )
        self.clear()

        # self.grid = torch.linspace(
        #     0, 1, self.max_ninc + 1, device=self.device, dtype=self.dtype
        # )
        # self.grid = self.grid.unsqueeze(0).repeat(
        #     self.dim, 1
        # )  # Shape: (dim, max_ninc + 1)
        # self.inc = self.grid[:, 1:] - self.grid[:, :-1]  # Shape: (dim, max_ninc)
        # # Mask to handle different ninc per dimension
        # # mask = torch.arange(max_ninc, device=self.device).unsqueeze(
        # #     0
        # # ) < self.ninc.unsqueeze(1)
        # # self.grid[~mask] = self.grid.max()  # Assign grid boundary for unused increments
        # # self.inc[~mask[:, :-1]] = 0.0  # Zero increments where not used
        # self.clear()

    def extract_grid(self):
        "Return a list of lists specifying the map's grid."
        grid_list = []
        for d in range(self.dim):
            ng = self.ninc[d] + 1
            grid_list.append(self.grid[d, :ng].tolist())
        return grid_list

    @torch.no_grad()
    def clear(self):
        "Clear information accumulated by :meth:`AdaptiveMap.add_training_data`."
        # self.sum_f = None
        # self.n_f = None
        # if self.sum_f is not None:
        self.sum_f.zero_()
        # self.n_f.f
        self.n_f.zero_()

    @torch.no_grad()
    def forward(self, u):
        u_ninc = u * self.ninc
        iu = torch.floor(u_ninc).long()
        du_ninc = u_ninc - iu

        batch_size = u.size(0)
        # Clamp iu to [0, ninc-1] to handle out-of-bounds indices
        min_tensor = torch.zeros((1, self.dim), dtype=iu.dtype, device=self.device)
        max_tensor = (self.ninc - 1).unsqueeze(0).to(iu.dtype)  # Shape: (1, dim)
        iu_clamped = torch.clamp(iu, min=min_tensor, max=max_tensor)

        grid_expanded = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        inc_expanded = self.inc.unsqueeze(0).expand(batch_size, -1, -1)

        grid_gather = torch.gather(grid_expanded, 2, iu_clamped.unsqueeze(2)).squeeze(
            2
        )  # Shape: (batch_size, dim)
        inc_gather = torch.gather(inc_expanded, 2, iu_clamped.unsqueeze(2)).squeeze(2)

        x = grid_gather + inc_gather * du_ninc
        log_detJ = (inc_gather * self.ninc).log_().sum(dim=1)

        # Handle out-of-bounds by setting x to grid boundary and adjusting detJ
        out_of_bounds = iu >= self.ninc
        if out_of_bounds.any():
            # Create indices for out-of-bounds
            # For each sample and dimension, set x to grid[d, ninc[d]]
            # and log_detJ += log(inc[d, ninc[d]-1] * ninc[d])
            boundary_grid = (
                self.grid[torch.arange(self.dim, device=self.device), self.ninc]
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            # x = torch.where(out_of_bounds, boundary_grid, x)
            x[out_of_bounds] = boundary_grid[out_of_bounds]

            boundary_inc = (
                self.inc[torch.arange(self.dim, device=self.device), self.ninc - 1]
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            adj_log_detJ = ((boundary_inc * self.ninc).log_() * out_of_bounds).sum(
                dim=1
            )
            log_detJ += adj_log_detJ

        return x, log_detJ

    # @torch.no_grad()
    # def inverse(self, x):
    #     """Inverse map from x-space to u-space."""
    #     # Find indices where x falls in the grid
    #     iu = torch.searchsorted(self.grid, x, right=True) - 1
    #     iu = torch.clamp(iu, 0, self.ninc - 1)

    #     # Compute du
    #     du = (x - torch.gather(self.grid, 1, iu)) / torch.gather(self.inc, 1, iu)

    #     # Compute u
    #     u = (iu.float() + du) / self.ninc.float()

    #     # Compute detJ
    #     detJ = torch.gather(self.inc, 1, iu) * self.ninc.float()
    #     log_detJ = torch.log(detJ).sum(dim=1)

    #     # Handle out-of-bounds
    #     lower_bound = x <= self.grid[:, 0]
    #     upper_bound = x >= self.grid[:, -1]
    #     u = torch.where(lower_bound, torch.zeros_like(u), u)
    #     u = torch.where(upper_bound, torch.ones_like(u), u)
    #     log_detJ = torch.where(
    #         lower_bound | upper_bound,
    #         torch.log(self.inc[:, -1] * self.ninc.float()),
    #         log_detJ,
    #     )

    #     return u, log_detJ

    @torch.no_grad()
    def inverse(self, x):
        """
        Inverse map from x-space to u-space.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, dim) representing points in x-space.

        Returns:
            u (torch.Tensor): Tensor of shape (batch_size, dim) representing points in u-space.
            log_detJ (torch.Tensor): Tensor of shape (batch_size,) representing the log determinant of the Jacobian.
        """
        x.to(self.device)
        batch_size, dim = x.shape

        # Initialize output tensors
        u = torch.empty_like(x)
        log_detJ = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        # Loop over each dimension to perform inverse mapping
        for d in range(dim):
            # Extract the grid and increment for dimension d
            grid_d = self.grid[d]  # Shape: (max_ninc + 1,)
            inc_d = self.inc[d]  # Shape: (max_ninc,)

            # ninc_d = self.ninc[d].float()  # Scalar tensor
            ninc_d = self.ninc[d]  # Scalar tensor

            # Perform searchsorted to find indices where x should be inserted to maintain order
            # torch.searchsorted returns indices in [0, max_ninc +1]
            iu = (
                torch.searchsorted(grid_d, x[:, d], right=True) - 1
            )  # Shape: (batch_size,)

            # Clamp indices to [0, ninc_d - 1] to ensure they are within valid range
            iu_clamped = torch.clamp(iu, min=0, max=ninc_d - 1)  # Shape: (batch_size,)

            # Gather grid and increment values based on iu_clamped
            # grid_gather and inc_gather have shape (batch_size,)
            grid_gather = grid_d[iu_clamped]  # Shape: (batch_size,)
            inc_gather = inc_d[iu_clamped]  # Shape: (batch_size,)

            # Compute du: fractional part within the increment
            du = (x[:, d] - grid_gather) / (inc_gather + TINY)  # Shape: (batch_size,)

            # Compute u for dimension d
            u[:, d] = (du + iu_clamped) / ninc_d  # Shape: (batch_size,)

            # Compute log determinant contribution for dimension d
            log_detJ += (inc_gather * ninc_d + TINY).log_()  # Shape: (batch_size,)

            # Handle out-of-bounds cases
            # Lower bound: x <= grid[d, 0]
            lower_mask = x[:, d] <= grid_d[0]  # Shape: (batch_size,)
            if lower_mask.any():
                u[:, d].masked_fill_(lower_mask, 0.0)
                log_detJ += (inc_d[0] * ninc_d + TINY).log_()

            # Upper bound: x >= grid[d, ninc_d]
            upper_mask = x[:, d] >= grid_d[ninc_d]  # Shape: (batch_size,)
            if upper_mask.any():
                u[:, d].masked_fill_(upper_mask, 1.0)
                log_detJ += (inc_d[ninc_d - 1] * ninc_d + TINY).log_()

        return u, log_detJ

    # @torch.no_grad()
    # def inverse(self, x):
    #     # self.detJ.fill_(1.0)
    #     x = x.to(self.device)
    #     u = torch.empty_like(x)
    #     detJ = torch.ones(x.shape[0], device=x.device)
    #     for d in range(self.dim):
    #         ninc = self.ninc[d]
    #         iu = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

    #         mask_valid = (iu > 0) & (iu <= ninc)
    #         mask_lower = iu <= 0
    #         mask_upper = iu > ninc

    #         # Handle valid range (0 < iu <= ninc)
    #         if mask_valid.any():
    #             iui_valid = iu[mask_valid] - 1
    #             u[mask_valid, d] = (
    #                 iui_valid
    #                 + (x[mask_valid, d] - self.grid[d, iui_valid])
    #                 / self.inc[d, iui_valid]
    #             ) / ninc
    #             detJ[mask_valid] *= self.inc[d, iui_valid] * ninc

    #         # Handle lower bound (iu <= 0)\
    #         if mask_lower.any():
    #             u[mask_lower, d] = 0.0
    #             detJ[mask_lower] *= self.inc[d, 0] * ninc

    #         # Handle upper bound (iu > ninc)
    #         if mask_upper.any():
    #             u[mask_upper, d] = 1.0
    #             detJ[mask_upper] *= self.inc[d, ninc - 1] * ninc

    #     return u, detJ.log_()


# class NormalizingFlow(Map):
#     def __init__(self, dim, flow_model, device="cpu"):
#         super().__init__(dim, device)
#         self.flow_model = flow_model.to(device)

#     def forward(self, u):
#         return self.flow_model.forward(u)

#     def inverse(self, x):
#         return self.flow_model.inverse(x)
