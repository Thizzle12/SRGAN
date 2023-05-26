import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int = 1,
        padding: str = "same",
        use_bn: bool = True,
        discriminator: bool = False,
        use_act: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_act = use_act

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # nn.Identity just returns input as is, if use batch norm is false.
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        # Return activation of x if convblock uses activation.
        # It does not use activation as the second part of a residual block.
        if self.use_act:
            return self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.conv_1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.conv_2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_act=False,
        )

    def forward(self, x):
        skip = x
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x + skip


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        scale_factor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        # in_channels * 4, H, W -> in_channels, H * 2, W * 2
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return self.act(x)


class SRGenerator(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        n_residuals: int = 5,
        num_features: int = 64,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.initial = ConvBlock(
            in_channels=n_channels,
            out_channels=num_features,
            kernel_size=9,
            stride=1,
            padding=4,
            use_bn=False,
        )

        res_blocks = [
            ResidualBlock(in_channels=num_features) for _ in range(n_residuals)
        ]
        # Unpack the list of blocks in sequential by using *[].
        self.residuals = nn.Sequential(*res_blocks)

        self.conv_block = ConvBlock(
            in_channels=num_features, out_channels=num_features, use_act=False
        )

        self.upsamples = nn.Sequential(
            UpsampleBlock(in_channels=num_features, scale_factor=2),
            UpsampleBlock(in_channels=num_features, scale_factor=2),
        )

        self.out = nn.Conv2d(
            in_channels=num_features,
            out_channels=n_channels,
            kernel_size=9,
            padding=4,
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.conv_block(x)
        x = initial + x
        x = self.upsamples(x)
        x = self.out(x)
        # TODO - test if softmax is better. TanH is -1 to 1, where sigmoid is 0 to 1.
        return nn.functional.tanh(x)


class SRDiscriminator(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, int] = (64, 64),
        in_channels: int = 3,
        features: list[int] = [64, 64, 128, 128, 256, 256, 512, 512],
        *args,
        **kwargs,
    ):
        super().__init__()
        blocks = []
        division_size = 0

        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    stride=(1 + idx % 2),
                    padding=1,
                    discriminator=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            # Set input channel to feature size.
            in_channels = feature
            division_size += 1 if idx % 2 == 0 else 0

        w, h = img_shape
        classifier_feat_w = w // 2**division_size
        classifier_feat_h = h // 2**division_size

        self.conv_blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((classifier_feat_w, classifier_feat_h)),
            nn.Flatten(),
            nn.Linear(
                features[-1] * classifier_feat_w * classifier_feat_h,
                out_features=1024,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return nn.functional.sigmoid(x)


def test():
    low_resolution = 32

    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = SRGenerator()
        gen_out = gen(x)

        disc = SRDiscriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
