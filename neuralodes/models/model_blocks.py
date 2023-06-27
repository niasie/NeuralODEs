import torch
from torch import nn
from ..ode_solver import rk_solve, rk_adaptive_embedded
from ..utils import get_activation


class ConvolutionalDownsamplingBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            with_norm,
            kernel_size=3,
        ):
        super().__init__()
        self.activation = get_activation(activation)()
        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size, 2, 1),
        ) if with_norm else nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size, 2, 1),
        ) 

    def forward(self, x):
        return self.layers(x)


class ConvolutionalDownSampler(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            with_norm,
            kernel_size=3,
            n_downsampling_blocks=2,
        ):
        super().__init__()
        self.activation = get_activation(activation)()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1),
            *[
                ConvolutionalDownsamplingBlock(
                    out_channels,
                    out_channels,
                    activation,
                    with_norm,
                    kernel_size,
                )
                for i in range(n_downsampling_blocks)
            ],
        ) 

    def forward(self, x):
        return self.layers(x)


class ConvolutionalClassificationHead(nn.Module):
    def __init__(self, in_channels, output_size, activation, with_norm) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(in_channels, in_channels) if with_norm else None
        self.layers = nn.Sequential(
            get_activation(activation)(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, output_size),
        )

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return self.layers(x)


class ConvolutionalResidualBlock(nn.Module):
    def __init__(self, n_channels, activation, with_norm):
        super().__init__()
        self.w1 = nn.Conv2d(n_channels, n_channels, 3, 1, 1)
        self.w2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

        self.activation = activation
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = nn.GroupNorm(n_channels, n_channels)
            self.norm2 = nn.GroupNorm(n_channels, n_channels)

    def forward(self, x):
        x_in = torch.clone(x)
        if self.with_norm:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.w1(x)
        if self.with_norm:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.w2(x)
        return x_in + x
    

class LinearResidualBlock(nn.Module):
    def __init__(self, size, activation, with_norm):
        super().__init__()
        self.w1 = nn.Linear(size, size)
        self.w2 = nn.Linear(size, size)

        self.activation = activation
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = nn.BatchNorm1d(size)
            self.norm2 = nn.BatchNorm1d(size)

    def forward(self, x):
        x_in = torch.clone(x)
        if self.with_norm:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.w1(x)
        if self.with_norm:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.w2(x)
        return x_in + x


class ConvolutionalODELayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            with_norm,
            tableau_low,
            tableau_high=None,
            kernel_size=3,
            t0=0.0,
            t1=1.0,
            dt=0.1,
            atol=1e-6,
            rtol=1e-6,
        ):
        super().__init__()
        self.w1 = nn.Conv2d(in_channels + 1, in_channels, kernel_size, 1, 1)
        self.w2 = nn.Conv2d(in_channels + 1, out_channels, kernel_size, 1, 1)

        self.solver = rk_solve if tableau_high is None else rk_adaptive_embedded
        self.tableau_low = tableau_low
        self.tableau_high = tableau_high

        self.activation = get_activation(activation)()
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = nn.GroupNorm(in_channels, in_channels)
            self.norm2 = nn.GroupNorm(in_channels, in_channels)
        
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.t1 = torch.tensor(t1, dtype=torch.float32)
        self.dt = torch.tensor(dt, dtype=torch.float32)
        self.atol = atol
        self.rtol = rtol

    def _conv_with_time(self, t, x, conv):
        ts = torch.full_like(x[:, :1], t.item())
        return conv(torch.cat((ts, x), dim=1))

    def _ode_rhs(self, t, x):
        if self.with_norm:
            x = self.norm1(x)
        x = self.activation(x)
        x = self._conv_with_time(t, x, self.w1)
        if self.with_norm:
            x = self.norm2(x)
        x = self.activation(x)
        x = self._conv_with_time(t, x, self.w2)
        return x

    def forward(self, x):
        x_final = None
        if self.tableau_high is not None:
            x_final, _, _ = self.solver(
                x,
                self.t0,
                self.t1,
                self.dt,
                self._ode_rhs,
                self.tableau_low,
                self.tableau_high,
                False,
                self.atol,
                self.rtol,
            )
        else:
            x_final, _, _ = self.solver(
                x,
                self.t0,
                self.t1,
                self.dt,
                self._ode_rhs,
                self.tableau_low,
                False,
            )
        return x_final