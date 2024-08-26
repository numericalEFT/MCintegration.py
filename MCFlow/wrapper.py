import numpy as np
import torch
from torch import nn
from vegas import AdaptiveMap
import normflows as nf
from normflows import utils
#PiecewiseRationalQuadraticCoupling,
from .coupling import (
    PiecewiseLinearCoupling,
    PieceWiseVegasCoupling,
    PiecewiseRationalQuadraticCouplingFixWidth,
)
from .resnet import ResidualNet, Dense
#from nf.mask import create_alternating_binary_mask

class CoupledRationalQuadraticSpline(nf.flows.base.Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [source](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        init_width=None,
        num_context_channels=None,
        num_bins=8,
        tails=None,  # "linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        mask=None,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tails (str): Behaviour of the tails of the distribution, can be linear, circular for periodic distribution, or None for distribution on the compact interval
          tail_bound (float): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            # net = Dense(
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - utils.splines.DEFAULT_MIN_DERIVATIVE) - 1)
                )
            return net

        if mask == None:
            mask_input = utils.create_alternating_binary_mask(
                num_input_channels, even=reverse_mask
            )
        else:
            mask_input = mask
        if init_width == None:
            self.prqct = nf.flows.neural_spline.wrapper.PiecewiseRationalQuadraticCoupling(
                mask=mask_input,
                transform_net_create_fn=transform_net_create_fn,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
                apply_unconditional_transform=True,
                # min_bin_height=1e-6,
            )
        else:
            self.prqct = PiecewiseRationalQuadraticCouplingFixWidth(
                mask=mask_input,
                transform_net_create_fn=transform_net_create_fn,
                init_width=init_width,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
                apply_unconditional_transform=True,
                # min_bin_height=1e-6,
            )

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)

class VegasLinearSpline(nf.flows.base.Flow):
    """
    Neural coupling layer using the Vegas map for transformations.
    """

    def __init__(
        self,
        func,
        num_input_channels,
        integration_region,
        batchsize,
        num_adapt_samples=1000000,
        num_increments=1000,
    ):
        super().__init__()

        self.pvct = PieceWiseVegasCoupling(
            func,
            num_input_channels,
            integration_region,
            batchsize,
            num_adapt_samples,
            num_increments,
        )

    def forward(self, z):
        z, log_det = self.pvct(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.pvct.inverse(z)
        return z, log_det.view(-1)

class CoupledLinearSpline(nf.flows.base.Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [source](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - utils.splines.DEFAULT_MIN_DERIVATIVE) - 1)
                )
            return net

        self.plct = PiecewiseLinearCoupling(
            mask=utils.create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
        )

    def forward(self, z, context=None):
        z, log_det = self.plct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.plct(z, context)
        return z, log_det.view(-1)
