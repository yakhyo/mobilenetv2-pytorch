import torch.nn as nn


class Conv2dAct(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, g=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c1,
                              out_channels=c2,
                              kernel_size=k,
                              stride=s,
                              padding=_pad(k, d) if p is None else p,
                              dilation=d,
                              groups=g,
                              bias=False)
        self.norm = nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.01)
        self.act = act(inplace=True) if act is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels
        mid_channels = int(round(in_channels * expand_ratio))

        self._block = nn.Sequential(
            Conv2dAct(in_channels, mid_channels, k=1, act=nn.ReLU6) if expand_ratio != 1 else nn.Identity(),
            Conv2dAct(mid_channels, mid_channels, s=stride, g=mid_channels, act=nn.ReLU6),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self._block(x) if self._shortcut else self._block(x)


def _make_divisible(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width


def _pad(kernel_size, dilation=1):
    return kernel_size // (2 * dilation)


def _init_weight(self):
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
