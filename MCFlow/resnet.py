import torch

from torch import nn
from torch.nn import functional as F, init


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(out_features, eps=1e-3) for _ in range(2)]
            )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, out_features)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features, out_features),
                nn.Linear(out_features, out_features),
            ]
        )

        self.dropout = nn.Dropout(p=dropout_probability)

        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        if inputs.shape[-1] != temps.shape[-1]:  # If in_features != out_features
            inputs = nn.Linear(inputs.shape[-1], temps.shape[-1]).to(inputs.device)(
                inputs
            )
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        preprocessing=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.preprocessing = preprocessing
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)

        self.blocks = nn.ModuleList()
        current_features = hidden_features
        for i in range(num_blocks):
            if i % 2 == 0 and i != 0:
                next_features = current_features * 2
            else:
                next_features = current_features
            self.blocks.append(
                ResidualBlock(
                    in_features=current_features,
                    out_features=next_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            current_features = next_features
        self.final_layer = nn.Linear(current_features, out_features)

    def forward(self, inputs, context=None):
        if self.preprocessing is None:
            temps = inputs
        else:
            temps = self.preprocessing(inputs)
        if context is None:
            temps = self.initial_layer(temps)
        else:
            temps = self.initial_layer(torch.cat((temps, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)

        return outputs


class Dense(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        preprocessing=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.preprocessing = preprocessing
        # if context_features is not None:
        #     self.initial_layer = nn.Linear(
        #         #in_features + context_features, hidden_features
        #         in_features + context_features, out_features
        #     )
        # else:
        #     self.initial_layer = nn.Linear(in_features,out_features)
        # self.blocks = nn.ModuleList(
        #     [
        #         ResidualBlock(
        #             features=hidden_features,
        #             context_features=context_features,
        #             activation=activation,
        #             dropout_probability=dropout_probability,
        #             use_batch_norm=use_batch_norm,
        #         )
        #         for _ in range(num_blocks)
        #     ]
        # )
        # self.final_layer = nn.Linear(hidden_features, out_features)
        self.final_layer = nn.Linear(in_features, out_features)

    def forward(self, inputs, context=None):
        # if self.preprocessing is None:
        #     temps = inputs
        # else:
        #     temps = self.preprocessing(inputs)
        # if context is None:
        #     temps = self.initial_layer(temps)
        # else:
        #     temps = self.initial_layer(torch.cat((temps, context), dim=1))
        # for block in self.blocks:
        #     temps = block(temps, context=context)
        outputs = self.final_layer(inputs)
        return outputs
