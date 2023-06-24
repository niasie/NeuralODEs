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


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, activation, with_batchnorm):
        super().__init__()
        self.w1 = torch.nn.Linear(input_size, input_size)
        self.w2 = torch.nn.Linear(input_size, output_size)
        self.activation = activation
        self.with_batchnorm = with_batchnorm
        self.bnorm1 = torch.nn.BatchNorm1d(input_size)
        self.bnorm2 = torch.nn.BatchNorm1d(input_size)

    def forward(self, x):
        x_in = torch.clone(x)
        if self.with_batchnorm:
            x = self.bnorm1(x)
        x = self.activation(x)
        x = self.w1(x)
        if self.with_batchnorm:
            x = self.bnorm2(x)
        x = self.activation(x)
        x = self.w2(x)
        return x_in + x


class ResNet(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            input_size,
            output_size,
            activation,
            with_batchnorm,
            ):
        super().__init__()

        self.activation = get_activation(activation)()
        self.residualBlocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    input_size=input_size,
                    output_size=input_size,
                    activation=self.activation,
                    with_batchnorm=with_batchnorm           
                )
                for i in range(num_blocks)        
            ]
        )
        self.output_layer = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, -1)
        for block in self.residualBlocks:
            x = block(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

