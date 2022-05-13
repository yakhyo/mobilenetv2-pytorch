import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


def _pad(kernel_size, dilation=1):
    return kernel_size // (2 * dilation)


def _make_divisible(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width


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


class EMA(torch.nn.Module):
    """ [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage] """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            ema_v = self.model.state_dict().values()
            model_v = model.state_dict().values()
            for e, m in zip(ema_v, model_v):
                e.copy_(update_fn(e, m))

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class PolyLoss:
    """ [https://arxiv.org/abs/2204.12511?context=cs] """

    def __init__(self, reduction='none', label_smoothing=0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, prediction, target, epsilon=1.0):
        ce = F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)
        pt = torch.sum(F.one_hot(target, num_classes=1000) * self.softmax(prediction), dim=-1)
        pl = torch.mean(ce + epsilon * (1 - pt))
        return pl


class CrossEntropyLoss:
    """ [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html] """

    def __init__(self, reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()

        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(self, prediction, target):
        return F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-7, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
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
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p.data)  # PyTorch inits to zero
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
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()
                else:
                    avg = square_avg.add(group['eps']).sqrt_()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg, value=group['lr'])
                        p.data.add_(-buf)
                    else:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


class StepLR:

    def __init__(self, optimizer, step_size, gamma=1., warmup_epochs=0, warmup_lr_init=0):

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lr_values = [group['initial_lr'] for group in self.optimizer.param_groups]
        self.update_groups(self.base_lr_values)

        if self.warmup_epochs:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_epochs for v in self.base_lr_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_lr_values]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [base_lr * (self.gamma ** (epoch // self.step_size)) for base_lr in self.base_lr_values]
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
    model = mobilenet_v2()

    img = torch.randn(1, 3, 224, 224)
    print(model(img).shape)

    print("Num params. of MobileNetV2: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
