import torch
from torch import nn, Tensor

from typing import Optional, Callable, List

__all__ = ["MobileNetV2"]


class Conv2dNormActivation(nn.Sequential):
    """Standard Convolutional Block
    Consists of Convolutional, Normalization, Activation Layers
    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size
        stride: stride size
        padding: padding size
        dilation: dilation rate
        groups: number of groups
        activation: activation function
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            dilation: int = 1,
            groups: int = 1,
            activation: Optional[Callable[..., torch.nn.Module]] = None
    ) -> None:
        if padding is None:
            padding = kernel_size // (2 * dilation)
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=0.001,
                momentum=0.01
            )
        ]
        if activation is not None:
            layers.append(activation(inplace=True))

        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Inverted Residual Block
    Args:
        in_channels: inpout channels
        out_channels: output channels
        stride: stride size
        expand_ratio: expansion ratio
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int
    ) -> None:
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels
        mid_channels = int(round(in_channels * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=1,
                    activation=nn.ReLU6
                )
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    stride=stride,
                    groups=mid_channels,
                    activation=nn.ReLU6
                ),
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self._shortcut:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNet V2 <https://arxiv.org/abs/1801.04381>"""

    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        filters = [3, 32, 16, 24, 32, 64, 96, 160, 320, 1280]

        features: List[nn.Module] = [
            # 1/2
            Conv2dNormActivation(
                filters[0], filters[1], stride=2, activation=nn.ReLU6
            ),
            InvertedResidual(filters[1], filters[2], 1, 1),
            # 1/4
            InvertedResidual(filters[2], filters[3], 2, 6),
            InvertedResidual(filters[3], filters[3], 1, 6),
            # 1/8
            InvertedResidual(filters[3], filters[4], 2, 6),
            InvertedResidual(filters[4], filters[4], 1, 6),
            InvertedResidual(filters[4], filters[4], 1, 6),
            # 1/16
            InvertedResidual(filters[4], filters[5], 2, 6),
            InvertedResidual(filters[5], filters[5], 1, 6),
            InvertedResidual(filters[5], filters[5], 1, 6),
            InvertedResidual(filters[5], filters[5], 1, 6),
            # 1/32
            InvertedResidual(filters[5], filters[6], 1, 6),
            InvertedResidual(filters[6], filters[6], 1, 6),
            InvertedResidual(filters[6], filters[6], 1, 6),
            # 1/64
            InvertedResidual(filters[6], filters[7], 2, 6),
            InvertedResidual(filters[7], filters[7], 1, 6),
            InvertedResidual(filters[7], filters[7], 1, 6),
            InvertedResidual(filters[7], filters[8], 1, 6),

            Conv2dNormActivation(
                filters[8], filters[9], kernel_size=1, activation=nn.ReLU6
            )
        ]
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    mobilenet = MobileNetV2()

    img = torch.randn(1, 3, 224, 224)
    print(mobilenet(img).shape)
    from torchsummary import summary

    summary(mobilenet, input_size=(3, 224, 224))
    print("Num params. of MobileNetV2: {}".format(
        sum(p.numel() for p in mobilenet.parameters() if p.requires_grad)))
