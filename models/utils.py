import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.InstanceNorm2d)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list) # w h
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)]     
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))]   # w h
    return torch.stack(grid_list, dim=-1)


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) * self.scale
        return x.permute(0, 3, 1, 2)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='BN'):
        super().__init__()
        if norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(dim)
        elif norm_type == 'BN':
            self.norm = nn.BatchNorm2d(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class h_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
    
class ConvAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1, bias=False, use_actv=True):
        super().__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=padding, groups=groups, dilation=dilation, bias=bias) 
        self.actv = nn.GELU() if use_actv else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.actv(x)


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1, bias=False, norm_type='BN'):
        super().__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        if norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(dim_out)
        elif norm_type == 'BN':
            self.norm = nn.BatchNorm2d(dim_out)
        self.actv = nn.GELU()

    def forward(self, x):
        return self.actv(self.norm(self.conv(x)))
    

class ConvNorm(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1, bias=False, norm_type='BN'):
        super().__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        if norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(dim_out)
        elif norm_type == 'BN':
            self.norm = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.norm(self.conv(x))


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, dp_rate=0.1, norm_type='BN'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride = stride
        self.dp_rate = dp_rate
        self.norm_type = norm_type
        self.learned_sc = dim_in != dim_out
        self._build_weights()

    def _build_weights(self):
        self.downsample = PreNorm(self.dim_in, nn.AvgPool2d(kernel_size=2, stride=self.stride) if self.stride > 1 else nn.Identity(), self.norm_type)
        self.shortcut = nn.Conv2d(self.dim_in, self.dim_out, 1, 1, 0, bias=False) if self.learned_sc else nn.Identity()

        self.conv = ConvNorm(self.dim_in, self.dim_in, kernel_size=7, groups=self.dim_in, norm_type=self.norm_type)
        self.mask = ConvNorm(self.dim_in, self.dim_in, kernel_size=7, groups=self.dim_in, norm_type=self.norm_type)
        self.sigmoid = h_sigmoid()

        self.pw1 = ConvNormAct(self.dim_in, self.dim_out * 2, kernel_size=1, norm_type=self.norm_type)
        self.pw2 = ConvNorm(self.dim_out * 2, self.dim_out, kernel_size=1, norm_type=self.norm_type)
        self.scale = Scale(self.dim_out) 

        self.dw = ConvNorm(self.dim_out, self.dim_out, kernel_size=3, groups=self.dim_out, norm_type=self.norm_type)
        self.drop_path = DropPath(self.dp_rate) if self.dp_rate > 0. and not self.learned_sc else nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        res = self.conv(x) * self.sigmoid(self.mask(x))
        res = self.pw1(res)
        res = self.pw2(res)
        x = self.shortcut(x) + self.scale(self.drop_path(res))
        return self.dw(x)


    
