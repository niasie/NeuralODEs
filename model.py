import torch
from ode_solver.RKIntegrator import rk_solve, rk_adaptive_embedded


def get_activation(activation: str):
    activation = activation.lower()

    if activation == "relu":
        return torch.nn.ReLU
    elif activation == "sigmoid":
        return torch.nn.Sigmoid
    elif activation == "tanh":
        return torch.nn.Tanh
    elif activation == "lrelu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"{activation} not supported\n")


class ConvolutionalDownSampler(torch.nn.Module):
    def __init__(self, n_channels, activation, with_norm):
        super().__init__()
        self.activation = get_activation(activation)()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_channels, 3, 1),
            torch.nn.GroupNorm(n_channels, n_channels),
            self.activation,
            torch.nn.Conv2d(n_channels, n_channels, 4, 2, 1),
            torch.nn.GroupNorm(n_channels, n_channels),
            self.activation,
            torch.nn.Conv2d(n_channels, n_channels, 4, 2, 1),
        ) if with_norm else torch.nn.Sequential(
            torch.nn.Conv2d(1, n_channels, 3, 1),
            self.activation,
            torch.nn.Conv2d(n_channels, n_channels, 4, 2, 1),
            self.activation,
            torch.nn.Conv2d(n_channels, n_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ConvolutionalClassificationHead(torch.nn.Module):
    def __init__(self, n_channels, output_size, activation, with_norm) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_channels, n_channels) if with_norm else None
        self.layers = torch.nn.Sequential(
            get_activation(activation)(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(n_channels, output_size),
        )

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return self.layers(x)


class ResidualBlockConv(torch.nn.Module):
    def __init__(self, n_channels, activation, with_norm):
        super().__init__()
        self.w1 = torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1)
        self.w2 = torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1)

        self.activation = activation
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = torch.nn.GroupNorm(n_channels, n_channels)
            self.norm2 = torch.nn.GroupNorm(n_channels, n_channels)

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


class ResidualBlockLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, activation, with_norm):
        super().__init__()
        self.w1 = torch.nn.Linear(input_size, input_size)
        self.w2 = torch.nn.Linear(input_size, output_size)

        self.activation = activation
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = torch.nn.BatchNorm1d(input_size)
            self.norm2 = torch.nn.BatchNorm1d(input_size)

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


class ResNetLinear(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            input_size,
            output_size,
            activation,
            with_norm,
        ):
        super().__init__()

        self.activation = get_activation(activation)()
        self.residual_blocks = torch.nn.Sequential(
            *[
                ResidualBlockLinear(
                    input_size=input_size,
                    output_size=output_size,
                    activation=self.activation,
                    with_norm=with_norm,
                )
                for i in range(num_blocks)        
            ]
        )
        self.norm = None
        if with_norm:
            self.norm = torch.nn.BatchNorm1d(input_size)

        self.classification_head = torch.nn.Sequential(
            self.activation,
            torch.nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        b = x.shape[0]
        x.reshape(b, -1)
        x = self.residual_blocks(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.classification_head(x)
        return x


class ResNetConv(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            n_channels,
            output_size,
            activation,
            with_norm,
            ):
        super().__init__()

        self.activation = get_activation(activation)()
        self.downsampler = ConvolutionalDownSampler(n_channels, activation, with_norm)
        self.residual_blocks = torch.nn.Sequential(
            *[
                ResidualBlockConv(
                    n_channels=n_channels,
                    activation=self.activation,
                    with_norm=with_norm,
                )
                for i in range(num_blocks)        
            ]
        )
        self.classification_head = ConvolutionalClassificationHead(
            n_channels,
            output_size,
            activation,
            with_norm,
        )

    def forward(self, x):
        x = self.downsampler(x)
        x = self.residual_blocks(x)
        x = self.classification_head(x)
        return x


class ODELayer(torch.nn.Module):
    def __init__(
            self,
            n_channels,
            activation,
            with_norm,
            tableau_low,
            tableau_high=None,
            t0=0.0,
            t1=1.0,
            dt=0.1,
            atol=1e-6,
            rtol=1e-6,
        ):
        super().__init__()
        self.w1 = torch.nn.Conv2d(n_channels + 1, n_channels, 3, 1, 1)
        self.w2 = torch.nn.Conv2d(n_channels + 1, n_channels, 3, 1, 1)

        self.solver = rk_solve if tableau_high is None else rk_adaptive_embedded
        self.tableau_low = tableau_low
        self.tableau_high = tableau_high

        self.activation = activation
        self.with_norm = with_norm
        self.norm1 = None
        self.norm2 = None
        if with_norm:
            self.norm1 = torch.nn.GroupNorm(n_channels, n_channels)
            self.norm2 = torch.nn.GroupNorm(n_channels, n_channels)
        
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
            x_final, _, _ = self.solver(x, self.t0, self.t1, self.dt, self._ode_rhs, self.tableau_low, self.tableau_high, False, self.atol, self.rtol)
        else:
            x_final, _, _ = self.solver(x, self.t0, self.t1, self.dt, self._ode_rhs, self.tableau_low, False)
        return x_final


class ODEClassifier(torch.nn.Module):
    def __init__(
            self,
            n_channels,
            output_size,
            activation,
            with_norm,
            tableau_low,
            tableau_high=None,
            t0=0.0,
            t1=1.0,
            dt=0.1,
            atol=1e-6,
            rtol=1e-6,
        ):
        super().__init__()
        self.ode_layer = ODELayer(
            n_channels,
            get_activation(activation)(),
            with_norm,
            tableau_low,
            tableau_high,
            t0,
            t1,
            dt,
            atol,
            rtol,
        )
        self.downsampler = ConvolutionalDownSampler(n_channels, activation, with_norm)
        self.classification_head = ConvolutionalClassificationHead(n_channels, output_size, activation, with_norm)

    def forward(self, x):
        x = self.downsampler(x)
        x = self.ode_layer(x)
        x = self.classification_head(x)
        return x
