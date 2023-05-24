import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        use_bn=True,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class SRGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class SRDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
