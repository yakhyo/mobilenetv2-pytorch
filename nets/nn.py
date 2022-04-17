import torch
import torch.nn as nn
from utils.misc import Conv2dAct, _init_weight, InvertedResidual


class MobileNetV2(nn.Module):
    """ [https://arxiv.org/abs/1801.04381] """

    def __init__(self, num_classes=1000, dropout=0.2, init_weights=True):
        super().__init__()
        if init_weights:
            _init_weight(self)

        self._layers = [Conv2dAct(3, 32, s=2, act=nn.ReLU6)]
        self._layers.extend([
            InvertedResidual(in_channels=32, out_channels=16, stride=1, expand_ratio=1),

            InvertedResidual(in_channels=16, out_channels=24, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=24, out_channels=24, stride=1, expand_ratio=6),

            InvertedResidual(in_channels=24, out_channels=32, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=32, out_channels=32, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=32, out_channels=32, stride=1, expand_ratio=6),

            InvertedResidual(in_channels=32, out_channels=64, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6),

            InvertedResidual(in_channels=64, out_channels=96, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expand_ratio=6),

            InvertedResidual(in_channels=96, out_channels=160, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=160, out_channels=160, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=160, out_channels=160, stride=1, expand_ratio=6),

            InvertedResidual(in_channels=160, out_channels=320, stride=1, expand_ratio=6),
        ])

        self._layers.append(Conv2dAct(320, 1280, k=1, act=nn.ReLU6))
        self.features = nn.Sequential(*self._layers)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(*args, **kwargs) -> MobileNetV2:
    """ MobileNetV2 """
    return MobileNetV2(*args, **kwargs)


if __name__ == '__main__':
    model = mobilenet_v2()

    img = torch.randn(1, 3, 224, 224)
    print(model(img).shape)

    print("Num params. of MobileNetV2: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

