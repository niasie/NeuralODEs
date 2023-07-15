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
            ode_solver,
            kernel_size=3,
            t0=0.0,
            t1=1.0,
            dt=0.1,
        ):
        super().__init__()
        self.w1 = nn.Conv2d(in_channels + 1, in_channels, kernel_size, 1, 1)
        self.w2 = nn.Conv2d(in_channels + 1, out_channels, kernel_size, 1, 1)

        self.solver = ode_solver

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
        x_final, _, _ = self.solver(
            self._ode_rhs,
            x,
            self.t0,
            self.t1,
            self.dt,
        )
        return x_final


class ParemeterPredicitingNetwork(nn.Module):
    def __init__(
            self,
            z_size,
            n_neurons,
            n_functions,
        ) -> None:
        super().__init__()
        self.z_size = z_size
        self.n_neurons = n_neurons
        self.n_functions = n_functions

        self.w_in_network = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_functions * z_size),
        )

        self.w_out_network = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_functions * z_size),
        )
        
        self.b_network = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_functions),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_neurons),
            nn.Tanh(),
            nn.Linear(n_neurons, n_functions),
            nn.Sigmoid()
        )

    def forward(self, t):
        t = t.reshape(1, 1)
        w_in = self.w_in_network(t)
        w_out = self.w_out_network(t)
        b = self.b_network(t)
        gate = self.gate_network(t)

        w_in = w_in.reshape(1, self.n_functions, self.z_size, 1)
        w_out = w_out.reshape(1, self.n_functions, 1, self.z_size)
        b = b.reshape(1, self.n_functions, 1, 1)
        gate = gate.reshape(1, self.n_functions, 1, 1)

        return w_in, w_out, b, gate


class ContinousNormalizingFlowRHS(nn.Module):
    def __init__(self, z_size, n_neurons_param_net, n_functions) -> None:
        super().__init__()

        self.z_size = z_size
        self.n_neurons_parameter_network = n_neurons_param_net
        self.n_functions = n_functions

        self.parameter_predicting_network = ParemeterPredicitingNetwork(
            z_size=z_size,
            n_neurons=n_neurons_param_net,
            n_functions=n_functions,
        )
        self.tanh = nn.Tanh()

    def forward(self, t, z_and_logpz):
        t = t.squeeze()
        # w_in: [1, n_funcs, z_size, 1]
        # w_out: [1, n_funcs, 1, z_size]
        # b: [1, n_funcs, z_size, 1]
        # gate: [1, n_funcs, 1, 1]
        z = z_and_logpz[..., 0:self.z_size]
        # logpz = z_and_logpz[..., -1]

        z.requires_grad_(True)
        w_in, w_out, b, gate = self.parameter_predicting_network(t)

        z_ = z.unsqueeze(1).unsqueeze(1)
        hidden_state = self.tanh(torch.matmul(z_, w_in) + b)
        gated_hidden_state = hidden_state * gate
        dz_dt = torch.matmul(gated_hidden_state, w_out).mean(dim=1).squeeze()

        # dzdt = f, need trace of dzdf
        trace = 0.0
        for i in range(self.z_size):
            # get i-th row of jacobian
            dfi_dz = torch.autograd.grad(dz_dt[:, i].sum(), z, create_graph=True)[0]
            trace += dfi_dz[:, i]
        trace = trace.unsqueeze(1)

        dlogpz_dt = -trace
        return torch.concat((dz_dt, dlogpz_dt), dim=-1)

