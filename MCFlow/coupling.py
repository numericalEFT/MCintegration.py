import warnings

import numpy as np
import torch
from torch import nn
from vegas import AdaptiveMap
import normflows as nf
from normflows import utils





class PieceWiseVegasCoupling(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        func,
        num_input_channels,
        integration_region,
        batchsize,
        num_adapt_samples=1000000,
        num_increments=1000,
        niters=20,
        alpha=1.0,
    ):
        super().__init__()

        vegas_map = AdaptiveMap(integration_region, ninc=num_increments)
        nblock = num_adapt_samples // batchsize
        num_adapt_samples = nblock * batchsize
        y_np = np.random.uniform(0.0, 1.0, (num_adapt_samples, num_input_channels))
        fx = torch.empty(num_adapt_samples, dtype=torch.float64)

        x = torch.empty(num_adapt_samples, num_input_channels, dtype=torch.float64)
        jac = torch.empty(num_adapt_samples, dtype=torch.float64)
        f2 = torch.empty(num_adapt_samples, dtype=torch.float64)
        for _ in range(niters):
            vegas_map.map(y_np, x.numpy(), jac.numpy())
            for i in range(nblock):
                fx[i * batchsize : (i + 1) * batchsize] = func(
                    x[i * batchsize : (i + 1) * batchsize]
                )
            f2 = (jac * fx) ** 2
            vegas_map.add_training_data(y_np, f2.numpy())
            vegas_map.adapt(alpha=alpha)

        self.register_buffer("y", torch.empty(batchsize, num_input_channels))
        self.register_buffer("grid", torch.Tensor(vegas_map.grid))
        self.register_buffer("inc", torch.Tensor(vegas_map.inc))
        self.register_buffer("dim", torch.tensor(num_input_channels))
        self.register_buffer("x", torch.empty(batchsize, num_input_channels))
        self.register_buffer("jac", torch.ones(batchsize))
        if num_increments < 1000:
            self.register_buffer("ninc", torch.tensor(1000))
        else:
            self.register_buffer("ninc", torch.tensor(num_increments))

    @torch.no_grad()
    def forward(self, y):
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        x = torch.empty_like(y)
        jac = torch.ones(y.shape[0], device=x.device)
        # self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                x[mask, d] = (
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                x[mask_inv, d] = self.grid[d, self.ninc]
                jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return x, torch.log(jac)

    @torch.no_grad()
    def inverse(self, x):
        # self.jac.fill_(1.0)
        y = torch.empty_like(x)
        jac = torch.ones(x.shape[0], device=x.device)
        for d in range(self.dim):
            iy = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iy > 0) & (iy <= self.ninc)
            mask_lower = iy <= 0
            mask_upper = iy > self.ninc

            # Handle valid range (0 < iy <= self.ninc)
            if mask_valid.any():
                iyi_valid = iy[mask_valid] - 1
                y[mask_valid, d] = (
                    iyi_valid
                    + (x[mask_valid, d] - self.grid[d, iyi_valid])
                    / self.inc[d, iyi_valid]
                ) / self.ninc
                jac[mask_valid] *= self.inc[d, iyi_valid] * self.ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                y[mask_lower, d] = 0.0
                jac[mask_lower] *= self.inc[d, 0] * self.ninc

            # Handle upper bound (iy > self.ninc)
            if mask_upper.any():
                y[mask_upper, d] = 1.0
                jac[mask_upper] *= self.inc[d, self.ninc - 1] * self.ninc

        return y, torch.log(1.0 / jac)
    
class PiecewiseLinearCDF(nf.flows.base.Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        identity_init=True,
        min_bin_width=utils.splines.DEFAULT_MIN_BIN_WIDTH,
        # min_bin_height=utils.splines.DEFAULT_MIN_BIN_HEIGHT
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        # self.min_bin_height = min_bin_height

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            # self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            # self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        # unnormalized_heights = self._share_across_batch(
        #     self.unnormalized_heights, batch_size
        # )

        spline_fn = utils.splines.linear_spline
        spline_kwargs = {}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            **spline_kwargs,
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)

class PiecewiseLinearCoupling(nf.flows.neural_spline.coupling.PiecewiseCoupling):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        img_shape=None,
        min_bin_width=utils.splines.DEFAULT_MIN_BIN_WIDTH,
        # min_bin_height=utils.splines.DEFAULT_MIN_BIN_HEIGHT
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        # self.min_bin_height = min_bin_height

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        # if apply_unconditional_transform:
        #     unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
        #         shape=[features] + (img_shape if img_shape else []),
        #         num_bins=num_bins,
        #         tails=tails_,
        #         tail_bound=tail_bound_,
        #         min_bin_width=min_bin_width,
        #         min_bin_height=min_bin_height,
        #         min_derivative=min_derivative,
        #     )
        # else:
        #     unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            # unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        # unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            # unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            # unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        spline_fn = utils.splines.linear_spline
        spline_kwargs = {}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            # unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            # min_bin_height=self.min_bin_height,
            **spline_kwargs,
        )


class PiecewiseRationalQuadraticFixWidthCDF(nf.flows.base.Flow):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=True,
        min_bin_width=utils.splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=utils.splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=utils.splines.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound
        self.tails = tails

        if self.tails == "linear":
            num_derivatives = num_bins - 1
        elif self.tails == "circular":
            num_derivatives = num_bins
        else:
            num_derivatives = num_bins + 1

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    @staticmethod
    def _share_across_batch(params, batch_size):
        return params[None, ...].expand(batch_size, *params.shape)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = self._share_across_batch(
            self.unnormalized_widths, batch_size
        )
        unnormalized_heights = self._share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = self._share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        if self.tails is None:
            spline_fn = utils.splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = utils.splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseRationalQuadraticCouplingFixWidth(nf.flows.neural_spline.coupling.PiecewiseCoupling ):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        init_width,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=utils.splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=utils.splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=utils.splines.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.unnormalized_widths = init_width
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Split tails parameter if needed
        features_vector = torch.arange(len(mask))
        identity_features = features_vector.masked_select(mask <= 0)
        transform_features = features_vector.masked_select(mask > 0)
        if isinstance(tails, list) or isinstance(tails, tuple):
            self.tails = [tails[i] for i in transform_features]
            tails_ = [tails[i] for i in identity_features]
        else:
            self.tails = tails
            tails_ = tails

        if torch.is_tensor(tail_bound):
            tail_bound_ = tail_bound[identity_features]
        else:
            self.tail_bound = tail_bound
            tail_bound_ = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = (
                lambda features: PiecewiseRationalQuadraticFixWidthCDF(
                    shape=[features] + (img_shape if img_shape else []),
                    num_bins=num_bins,
                    tails=tails_,
                    tail_bound=tail_bound_,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                    min_derivative=min_derivative,
                )
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound[transform_features])

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 2 - 1
        elif self.tails == "circular":
            return self.num_bins * 2
        else:
            return self.num_bins * 2 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        # if init_width == None
        unnormalized_widths = self.unnormalized_widths
        unnormalized_heights = transform_params[..., : self.num_bins]
        unnormalized_derivatives = transform_params[..., self.num_bins :]
        assert (
            unnormalized_widths.shape == unnormalized_heights.shape
        ), f"{unnormalized_widths.shape}, { unnormalized_heights.shape}\n"
        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = utils.splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = utils.splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        # print("width:", unnormalized_widths)
        # print("height:", unnormalized_heights)
        # print("derivatives:", unnormalized_derivatives)
        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )
