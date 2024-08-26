import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    assert (C % groups == 0)
    per_group = C // groups
    x = x.reshape(B, groups, per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.reshape(B, -1, H, W)
    return x


class SPADE(nn.Module):
    def __init__(self, kernel_size, norm_nc, label_nc, PONO=False, param_free_norm_type=''):
        super().__init__()
        nhidden = 128
        pw = kernel_size // 2

        if PONO:
            self.param_free_norm = PositionalNorm2d(affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)
        
        self.mlp_gamma_beta = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU(),
            nn.Conv2d(nhidden, norm_nc * 2, kernel_size=kernel_size, padding=pw)
        )

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        gamma, beta = torch.chunk(self.mlp_gamma_beta(segmap), 2, dim=1)
        return normalized * (1 + gamma) + beta
    

class PositionalNorm2d(nn.Module):
    def __init__(self, input_size=None, affine=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.input_size = input_size

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x
    


class convMod1D_actv(nn.Sequential):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1):
        super().__init__(
            Conv1DMod(dim_in, dim_out, kernel_size, stride, groups=groups, dilation=dilation),
            nn.InstanceNorm1d(dim_out, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )


class convMod2D_actv(nn.Sequential):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1):
        super().__init__(
            Conv2DMod(dim_in, dim_out, kernel_size, stride, groups=groups, dilation=dilation),
            nn.InstanceNorm2d(dim_out, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        )


class conv1D_norm_actv(nn.Sequential):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv1d(dim_in, dim_out, kernel_size, stride, padding=padding, groups=groups, dilation=dilation),
            nn.BatchNorm1d(dim_out),
            nn.LeakyReLU(0.2, inplace=True)
        )




class conv2D_norm(nn.Sequential):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, groups=1, dilation=1, label_nc=None):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=padding, groups=groups, dilation=dilation),
            nn.InstanceNorm2d(dim_out),
        )

class Conv1DMod(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, demod=True, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.demod = demod
        self.stride = stride
        self.groups = groups
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn((dim_out, dim_in//groups, kernel_size)))

    def _get_same_padding(self, size):
        return ((size - 1) * (self.stride - 1) + self.dilation * (self.kernel_size - 1)) // 2

    def forward(self, x):
        B, C, L = x.shape
        weights = self.weight
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(-1, -2), keepdim=True) + self.eps)
            weights = weights * d
        return F.conv1d(x, weights, padding=self._get_same_padding(L), groups=self.groups)
    

class Conv2DMod(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, demod=True, eps=1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.demod = demod
        self.stride = stride
        self.groups = groups
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn((dim_out, dim_in//groups, kernel_size, kernel_size)))

    def _get_same_padding(self, size):
        return ((size - 1) * (self.stride - 1) + self.dilation * (self.kernel_size - 1)) // 2

    def forward(self, x):
        B, C, H, W = x.shape
        weights = self.weight
        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(-1, -2, -3), keepdim=True) + self.eps)
            weights = weights * d
        return F.conv2d(x, weights, padding=self._get_same_padding(H), groups=self.groups)

