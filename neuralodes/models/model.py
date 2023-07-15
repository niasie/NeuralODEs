import torch
from torch import nn
from ..utils import get_activation
from ..utils import convrelu
from . import (
    ConvolutionalResidualBlock,
    ConvolutionalClassificationHead,
    ConvolutionalDownSampler,
    ConvolutionalODELayer,
    LinearResidualBlock,
    ContinousNormalizingFlowRHS,
)
from torchvision import models

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
        hidden_size,
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
            hidden_size=hidden_size,
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
    

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class NeuralODEUNet(nn.Module):

    def __init__(self, n_class,
                ode_solver,
                activation="relu",
                with_norm="False",
                t0=0.0,
                t1=1.0,
                dt=0.5,):
        super().__init__()


        def ConvolutionalOdeLayerWrapper(in_channels, out_channels, kernel_size=3):

            return ConvolutionalODELayer(
                ode_solver=ode_solver,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                with_norm=with_norm,
                t0=t0,
                t1=t1,
                dt=dt,
            )
        
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.t1 = torch.tensor(t1, dtype=torch.float32)
        self.dt = torch.tensor(dt, dtype=torch.float32)
        
        self.base_model = models.resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())         
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)   
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = nn.Sequential(ConvolutionalOdeLayerWrapper(64, 64), nn.Conv2d(64, 128,  kernel_size=(3,3), padding=(1, 1),), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=(3,3), padding=(1, 1), stride=(2,2)))
        self.layer2_1x1 = convrelu(128, 128, 1, 0)       
        self.layer3 = nn.Sequential(ConvolutionalOdeLayerWrapper(128, 128), nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1, 1),), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1, 1), stride=(2,2)))
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = nn.Sequential(ConvolutionalOdeLayerWrapper(256, 256), nn.Conv2d(256, 512,  kernel_size=(3,3), padding=(1, 1),
                                                                                      ), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1, 1), stride=(2,2)))
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)          
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)       
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out