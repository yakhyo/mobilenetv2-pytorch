from copy import deepcopy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional, Callable, List

__all__ = ["MobileNetV2", "mobilenet_v2"]


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


class Conv2dNormActivation(nn.Module):
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
        super().__init__()
        if padding is None:
            padding = kernel_size // (2 * dilation)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=0.001,
            momentum=0.01
        )
        yes_activation = activation is not None
        self.act = activation(inplace=True) if yes_activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


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
            in_channels,
            out_channels,
            stride,
            expand_ratio
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
        self._block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self._shortcut:
            return x + self._block(x)
        else:
            return self._block(x)


class MobileNetV2(nn.Module):
    """MobileNet V2 <`https://arxiv.org/abs/1801.04381`>"""

    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.2,
            init_weights: bool = True
    ) -> None:
        super().__init__()
        if init_weights:
            _init_weight(self)
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
            InvertedResidual(filters[5], filters[6], 2, 6),
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

        # average pooling
        self.pool = nn.AdaptiveMaxPool2d(1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class EMA(torch.nn.Module):
    """Exponential Moving Average from Tensorflow Implementation
    Args:
        model: nn.Module model
        decay: exponential decay value
    """

    def __init__(
            self,
            model: nn.Module,
            decay: float = 0.9999
    ) -> None:
        super().__init__()
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def _update(
            self,
            model: nn.Module,
            update_fn
    ) -> None:
        with torch.no_grad():
            ema_v = self.model.state_dict().values()
            model_v = model.state_dict().values()
            for e, m in zip(ema_v, model_v):
                e.copy_(update_fn(e, m))

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (
                1. - self.decay) * m)


class CrossEntropyLoss:
    """Cross Entropy Loss"""""

    def __init__(
            self,
            reduction: str = 'mean',
            label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input=prediction,
            target=target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-7, weight_decay=0,
                 momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay,
                        lr_in_momentum=lr_in_momentum)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(
                        p.data)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group['decoupled_decay']:
                        p.data.add_(p.data, alpha=-group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(
                        group['eps']).sqrt_()
                else:
                    avg = square_avg.add(group['eps']).sqrt_()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg,
                                                             value=group['lr'])
                        p.data.add_(-buf)
                    else:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


class StepLR:

    def __init__(self, optimizer, step_size, gamma=1., warmup_epochs=0,
                 warmup_lr_init=0):

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lr_values = [group['initial_lr'] for group in
                               self.optimizer.param_groups]
        self.update_groups(self.base_lr_values)

        if self.warmup_epochs:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_epochs for v
                                 in self.base_lr_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_lr_values]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if
                key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in
                      self.warmup_steps]
        else:
            values = [base_lr * (self.gamma ** (epoch // self.step_size)) for
                      base_lr in self.base_lr_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


def mobilenet_v2(*args, **kwargs) -> MobileNetV2:
    """ MobileNetV2 """
    return MobileNetV2(*args, **kwargs)


if __name__ == '__main__':
    mobilenet = mobilenet_v2()

    img = torch.randn(1, 3, 224, 224)
    print(mobilenet(img).shape)

    print("Num params. of MobileNetV2: {}".format(
        sum(p.numel() for p in mobilenet.parameters() if p.requires_grad)))
