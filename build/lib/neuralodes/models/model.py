import torch
from torch import nn
from ..utils import get_activation
from . import (
    ConvolutionalResidualBlock,
    ConvolutionalClassificationHead,
    ConvolutionalDownSampler,
    ConvolutionalODELayer,
    LinearResidualBlock,
    ContinousNormalizingFlowRHS,
)

class ResNetLinear(nn.Module):
    def __init__(
            self,
            num_blocks=6,
            input_size=28*28,
            output_size=10,
            activation="relu",
            with_norm=False,
        ):
        super().__init__()

        self.activation = get_activation(activation)()
        self.residual_blocks = nn.Sequential(
            *[
                LinearResidualBlock(
                    size=input_size,
                    activation=self.activation,
                    with_norm=with_norm,
                )
                for i in range(num_blocks)        
            ]
        )
        self.norm = None
        if with_norm:
            self.norm = nn.BatchNorm1d(input_size)

        self.classification_head = nn.Sequential(
            self.activation,
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        b = x.shape[0]
        x.reshape(b, -1)
        x = self.residual_blocks(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.classification_head(x)
        return x


class ResNetConv(nn.Module):
    def __init__(
            self,
            num_blocks=6,
            in_channels=1,
            n_channels=64,
            output_size=10,
            activation="relu",
            with_norm=False,
            kernel_size=3,
            n_downsampling_blocks=2,
            ):
        super().__init__()

        self.activation = get_activation(activation)()
        self.downsampler = ConvolutionalDownSampler(
            in_channels=in_channels,
            out_channels=n_channels,
            activation=activation,
            with_norm=with_norm,
            kernel_size=kernel_size,
            n_downsampling_blocks=n_downsampling_blocks,
        )
        self.residual_blocks = torch.nn.Sequential(
            *[
                ConvolutionalResidualBlock(
                    n_channels=n_channels,
                    activation=self.activation,
                    with_norm=with_norm,
                )
                for i in range(num_blocks)        
            ]
        )
        self.classification_head = ConvolutionalClassificationHead(
            in_channels=n_channels,
            output_size=output_size,
            activation=activation,
            with_norm=with_norm,
        )

    def forward(self, x):
        x = self.downsampler(x)
        x = self.residual_blocks(x)
        x = self.classification_head(x)
        return x


class ConvolutionalODEClassifier(nn.Module):
    def __init__(
        self,
        ode_solver,
        in_channels=1,
        n_channels=64,
        output_size=10,
        kernel_size=3,
        n_downsampling_blocks=2,
        activation="relu",
        with_norm="False",
        t0=0.0,
        t1=1.0,
        dt=0.1,
    ):
        super().__init__()
        self.ode_layer = ConvolutionalODELayer(
            ode_solver=ode_solver,
            in_channels=n_channels,
            out_channels=n_channels,
            activation=activation,
            with_norm=with_norm,
            t0=t0,
            t1=t1,
            dt=dt,
        )
        self.downsampler = ConvolutionalDownSampler(
            in_channels=in_channels,
            out_channels=n_channels,
            activation=activation,
            with_norm=with_norm,
            kernel_size=kernel_size,
            n_downsampling_blocks=n_downsampling_blocks,
        )
        self.classification_head = ConvolutionalClassificationHead(
            in_channels=n_channels,
            output_size=output_size,
            activation=activation,
            with_norm=with_norm,
        )

    def forward(self, x):
        x = self.downsampler(x)
        x = self.ode_layer(x)
        x = self.classification_head(x)
        return x


class ContinousNormalizingFlow(nn.Module):
    def __init__(
        self,
        z_size,
        n_neurons_param_net,
        n_functions,
        ode_solver,
        t0=0.0,
        t1=1.0,
        dt=0.1,
    ):    
        super().__init__()

        self.z_size = z_size
        self.n_neurons_parameter_network = n_neurons_param_net
        self.n_functions = n_functions

        self.ode_rhs = ContinousNormalizingFlowRHS(
            z_size=z_size,
            n_neurons_param_net=n_neurons_param_net,
            n_functions=n_functions,
        )

        self.ode_solver = ode_solver
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.t1 = torch.tensor(t1, dtype=torch.float32)
        self.dt = torch.tensor(dt, dtype=torch.float32)
        self.latent_to_sample = False

    def forward(self, z):
        b = z.shape[0]
        logpz = torch.zeros((b, 1), dtype=z.dtype, device=z.device)
        x = torch.concat((z, logpz), dim=-1)
        # forward ODE: latent -> sample
        # backward ODE: sample -> latent
        x_final, _, _ = self.ode_solver(
            f=self.ode_rhs,
            y0=x,
            t0=self.t0 if self.latent_to_sample else self.t1,
            t1=self.t1 if self.latent_to_sample else self.t0,
            dt=self.dt,
        )
        return x_final[:, 0:self.z_size], x_final[:, -1]
