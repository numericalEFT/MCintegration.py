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
        # Preallocate temporary tensors for adapt
        self.sum_f = torch.zeros(
            self.dim, self.max_ninc, dtype=self.dtype, device=self.device
        )
        self.n_f = torch.zeros(
            self.dim, self.max_ninc, dtype=self.dtype, device=self.device
        )
        self.avg_f = torch.ones(
            (self.dim, self.max_ninc), dtype=self.dtype, device=self.device
        )
        self.tmp_f = torch.zeros(
            (self.dim, self.max_ninc), dtype=self.dtype, device=self.device
        )

        self.make_uniform()

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
        iu = torch.floor(sample.u * self.ninc).long()
        for d in range(self.dim):
            indices = iu[:, d]
            self.sum_f[d].scatter_add_(0, indices, fval.abs())
            self.n_f[d].scatter_add_(0, indices, torch.ones_like(fval))

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

                new_grid[d, 0] = self.grid[d, 0]
                new_grid[d, ninc] = self.grid[d, ninc]

                target_cumulative_weights = (
                    torch.arange(1, ninc, device=self.device) * f_ninc
                )  # Calculate the target cumulative weights for each new grid point

                cumulative_avg_f = torch.cat(
                    (
                        torch.tensor([0.0], device=self.device),
                        torch.cumsum(avg_f, dim=0),
                    )
                )  # Calculate the cumulative sum of avg_f
                interval_indices = (
                    torch.searchsorted(
                        cumulative_avg_f, target_cumulative_weights, right=True
                    )
                    - 1
                )  # Find the intervals in the original grid where the target weights fall
                # Extract the necessary values using the interval indices
                grid_left = self.grid[d, interval_indices]
                inc_relevant = self.inc[d, interval_indices]
                avg_f_relevant = avg_f[interval_indices]
                cumulative_avg_f_relevant = cumulative_avg_f[interval_indices]

                # Calculate the fractional position within each interval
                fractional_positions = (
                    target_cumulative_weights - cumulative_avg_f_relevant
                ) / avg_f_relevant

                # Calculate the new grid points using vectorized operations
                new_grid[d, 1:ninc] = grid_left + fractional_positions * inc_relevant
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
        self.sum_f.zero_()
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
                torch.searchsorted(grid_d, x[:, d].contiguous(), right=True) - 1
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


# class NormalizingFlow(Map):
#     def __init__(self, dim, flow_model, device="cpu"):
#         super().__init__(dim, device)
#         self.flow_model = flow_model.to(device)

#     def forward(self, u):
#         return self.flow_model.forward(u)

#     def inverse(self, x):
#         return self.flow_model.inverse(x)
