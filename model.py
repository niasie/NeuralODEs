import torch

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
    def __init__(self, n_channels, activation):
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
        )

    def forward(self, x):
        return self.layers(x)


class ConvolutionalClassificationHead(torch.nn.Module):
    def __init__(self, n_channels, output_size, activation) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.GroupNorm(n_channels, n_channels),
            get_activation(activation)(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(n_channels, output_size),
        )

    def forward(self, x):
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
        self.downsampler = ConvolutionalDownSampler(n_channels, activation)
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
        )

    def forward(self, x):
        x = self.downsampler(x)
        x = self.residual_blocks(x)
        x = self.classification_head(x)
        return x

