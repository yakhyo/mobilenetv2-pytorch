from typing import List, Optional

import torch
from torch import nn, Tensor

from torchsummary import summary

__all__ = ["MobileNetV2"]


class Conv(nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU6"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            dilation: int = 1,
            groups: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU6(inplace=inplace)
        ]

        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Inverted Residual block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels
        mid_channels = int(round(in_channels * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # point-wise convolution
            layers.append(
                Conv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
            )
        layers.extend(
            [
                # depth-wise convolution
                Conv(in_channels=mid_channels, out_channels=mid_channels, stride=stride, groups=mid_channels),
                # point-wise-linear
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
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        if self._shortcut:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2) -> None:
        """MobileNet V2 main class
        Args:
            num_classes (int): number of classes
            dropout (float): dropout probability
        """
        super().__init__()

        _config = [
            [32, 16, 1, 1],

            [16, 24, 2, 6],
            [24, 24, 1, 6],

            [24, 32, 2, 6],
            [32, 32, 1, 6],
            [32, 32, 1, 6],

            [32, 64, 2, 6],
            [64, 64, 1, 6],
            [64, 64, 1, 6],
            [64, 64, 1, 6],
            [64, 96, 1, 6],
            [96, 96, 1, 6],
            [96, 96, 1, 6],

            [96, 160, 2, 6],
            [160, 160, 1, 6],
            [160, 160, 1, 6],
            [160, 320, 1, 6],
        ]

        # building first layer
        features: List[nn.Module] = [Conv(3, 32, stride=2)]

        # building inverted residual block
        for conf in _config:
            features.append(InvertedResidual(*conf))

        # building last several layers
        features.append(Conv(320, 1280, kernel_size=1))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

        # weight initialization
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    mobilenet = MobileNetV2()
    summary(mobilenet, input_size=(3, 224, 224), device="cpu")
