import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvBlock, ConvAct, PreNorm
from timm.models.vision_transformer import trunc_normal_

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])    
    grid_list = reversed(grid_list)
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)] 
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))] 
    return torch.stack(grid_list, dim=-1) 


def AWarp(features, last_flow, flow_attn, num_head, detach=False, padding_mode='border'):
    b, c, h, w = features.size()
    flow_attn = torch.repeat_interleave(flow_attn, c, 1)
    multi_feat = torch.repeat_interleave(features, num_head, 0)
    if detach:
       multi_feat = F.grid_sample(multi_feat, last_flow.permute(0, 2, 3, 1).detach(), mode='bilinear', padding_mode=padding_mode)
    else:
       multi_feat = F.grid_sample(multi_feat, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode)
    multi_att_warp_feat = multi_feat.reshape(b, -1, h, w) * flow_attn
    return sum(torch.split(multi_att_warp_feat, c, dim=1))

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.reshape(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.reshape(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).reshape(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out


def fused_leaky_relu(inputs, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (inputs.ndim - bias.ndim - 1)
        bias = bias.view(1, bias.shape[0], *rest_dim)
        inputs = inputs + bias
    return F.leaky_relu(inputs, negative_slope=negative_slope) * scale
    

def append_if(condition, var, elem):
    if condition:
        var.append(elem)
    return var


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class Upsample(nn.Module):
    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
    

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
class TryOnAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.key_dim = dim // num_heads
        self.scale = self.key_dim ** -0.5

        self.num_heads = num_heads
        self.window_size = 4      # windows所具有的pixels大小 change 8 to 4

        self.upsample = Upsample()

        self.q  = nn.Conv2d(dim, dim  , kernel_size=3, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        rel_index_coords = self.double_step_seq(2*self.window_size-1, self.window_size, 1, self.window_size)
        self.rel_position_index = rel_index_coords + rel_index_coords.T
        self.rel_position_index = self.rel_position_index.flip(1).contiguous()
    
        self.rel_position_bias_table = nn.Parameter(torch.zeros((2*self.window_size-1) * (2*self.window_size-1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.rel_position_bias_table, std=.02)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
    
    def window_partition(self, x):
        B, C, H, W = x.shape
        self.nH = H // self.window_size
        self.nW = W // self.window_size
        # 將影像資訊進行重新排列
        x = x.permute(0, 2, 3, 1)
        # B, H, W, C
        # 將影像分割成不同大小的Windows
        x = x.reshape(B, self.nH, self.window_size, self.nW, self.window_size, C).transpose(2, 3)       # B, nH, nW, H', W', C  
        x = x.reshape(B * self.nH * self.nW, self.window_size, self.window_size, C)                     # B*nH*nW, H', W', C  
        return x.permute(0, 3, 1, 2).contiguous()

    def window_concatenate(self, x):
        B_hat, C, H_hat, W_hat = x.shape            # B', C, H', W'
        x = x.permute(0, 2, 3, 1)                   # B', H', W', C
        x = x.reshape(-1, self.nH, self.nW, self.window_size, self.window_size, C).transpose(2, 3)      # B, nH, H', nW, W', C  
        return x.reshape(-1, self.nH*self.window_size, self.nW*self.window_size, C).permute(0, 3, 1, 2).contiguous() # B, C, H, W 
    
    def attention(self, src_con, ref_con, attn_forward):
        B, C, H, W = src_con.shape
        WS = self.window_size

        # 計算目標影像的query值，以及參考影像的key和value值
        Q = self.q(src_con)                                         # B, C, H, W
        K, V = self.kv(ref_con).split([self.dim, self.dim], dim=1)  # B, C, H, W

        # Feature Normalization
        Q = Q - torch.mean(Q, dim=(2, 3), keepdim=True)
        K = K - torch.mean(K, dim=(2, 3), keepdim=True)
        Q = Q / torch.norm(Q, dim=1, keepdim=True)
        K = K / torch.norm(K, dim=1, keepdim=True)

        Q = Q.reshape(B, self.num_heads, -1, H*W).transpose(-2, -1)     # B, num_heads, H*W, key_dim
        K = K.reshape(B, self.num_heads, -1, H*W)                       # B, num_heads, key_dim, H*W
        V = V.reshape(B, self.num_heads, -1, H*W).transpose(-2, -1)     # B, num_heads, H*W, key_dim

        attn = (Q @ K) * self.scale                             # B, num_heads, H*W, H*W

        relative_position_bias = self.rel_position_bias_table[self.rel_position_index.view(-1)].view(WS**2, WS**2, -1)  # N, N, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        if attn_forward is not None:
            attn = attn + attn_forward + relative_position_bias.unsqueeze(0)
        
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(-2, -1)                      # B, num_heads, key_dim, H*W
        return out.reshape(B, C, H, W), attn                         # B, C, H, W
    
    def forward(self, src_con, ref_con, skip=None):
        B, C, H, W = src_con.shape
        attn = None
        
        if H <= self.window_size and W <= self.window_size:
            tryon, attn = self.attention(src_con, ref_con, attn)
        else:
            src_con = self.window_partition(src_con)
            ref_con = self.window_partition(ref_con)
            tryon, attn  = self.attention(src_con, ref_con, attn)
            tryon  = self.window_concatenate(tryon)
        tryon =  self.proj(tryon)

        if skip is not None:
            skip = self.upsample(skip)
            tryon = tryon + skip
        return tryon
    
    
class FlowEstimator(nn.Module):
    def __init__(self, dim_in, dim_out=3, dim_mid=128, num_head=1, norm_type='BN'):
        super().__init__()
        dim_out = dim_out * num_head
        self.module = torch.nn.Sequential(
            ConvAct(dim_in , dim_mid, kernel_size=1),
            ConvAct(dim_mid, dim_mid, kernel_size=3, groups=dim_mid),
            ConvAct(dim_mid, dim_mid, kernel_size=3, groups=dim_mid),
            ConvAct(dim_mid, dim_out, kernel_size=1, use_actv=False),
        )
        self.module = PreNorm(dim_in, self.module, norm_type)
   
    def forward(self, wapring_fea, shape_fea):
        concat = torch.cat([wapring_fea, shape_fea], dim=1)
        return self.module(concat)
    
class CascadeWarpingModule(nn.Module):
    def __init__(self, dim_in, num_head, norm_type):
        super().__init__()
        self.num_head = num_head
        self.shapeNet   = FlowEstimator(dim_in * 2, num_head=num_head, norm_type=norm_type)
        self.textureNet = FlowEstimator(dim_in * 2, num_head=num_head, norm_type=norm_type)
        self.tryon_attn_coarse = TryOnAttention(dim_in, num_heads=num_head) # dim=128  # CHANGED!
        self.tryon_attn_fine = TryOnAttention(dim_in, num_heads=num_head)
    
    def forward(self, shape_fea, cloth_fea, shape_last_flow=None, cloth_last_flow=None, flow_attn=None):
        b, c, h, w = shape_fea.shape
        
        # Coarse
        if shape_last_flow is not None and flow_attn is not None:
            cloth_fea_ = AWarp(cloth_fea, shape_last_flow, flow_attn, self.num_head, detach=True)
        else:
            cloth_fea_ = cloth_fea
        
        cloth_fea_ = self.tryon_attn_coarse(cloth_fea_, shape_fea) # CHANGED!

        shape_delta_flow = self.shapeNet(cloth_fea_, shape_fea)               
        flow_attn        = shape_delta_flow[:, self.num_head*2:, ...]
        shape_delta_flow = shape_delta_flow[:, :self.num_head*2, ...].reshape(-1, 2, h, w)
        flow_attn = F.softmax(flow_attn, dim=1)             # b, num_head, h, w
        shape_flow = apply_offset(shape_delta_flow)         # b*num_head, h, w, 2
        if shape_last_flow is not None:
            shape_last_flow = F.grid_sample(shape_last_flow, shape_flow, mode='bilinear', padding_mode='border')
        else:
            shape_last_flow = shape_flow.permute(0, 3, 1, 2)                    # b*num_head, 2, h, w

        if cloth_last_flow is not None:
            cloth_last_flow_ = F.grid_sample(shape_last_flow, cloth_last_flow.permute(0, 2, 3, 1).detach(), mode='bilinear', padding_mode='border')
        else:
            cloth_last_flow_ = shape_last_flow.clone()
            
        cloth_fea_ = self.tryon_attn_coarse(cloth_fea_, shape_fea) # Do Attention one more time

        # Fine
        cloth_fea_ = AWarp(cloth_fea, cloth_last_flow_, flow_attn, self.num_head)
        
        cloth_fea_ = self.tryon_attn_fine(cloth_fea_, shape_fea)
        
        cloth_delta_flow = self.textureNet(cloth_fea_, shape_fea)
        flow_attn        = cloth_delta_flow[:, self.num_head*2:, ...]
        cloth_delta_flow = cloth_delta_flow[:, :self.num_head*2, ...].reshape(-1, 2, h, w)
        flow_attn = F.softmax(flow_attn, dim=1)             # b, num_head, h, w
        cloth_flow = apply_offset(cloth_delta_flow)         # b*num_head, h, w, 2
        if cloth_last_flow is not None:
            cloth_last_flow = F.grid_sample(cloth_last_flow, cloth_flow, mode='bilinear', padding_mode='border')
        else:
            cloth_last_flow = cloth_flow.permute(0, 3, 1, 2)                    # b*num_head, 2, h, w
        
        cloth_fea_ = self.tryon_attn_fine(cloth_fea_, shape_fea) # Do Attention one more time
        
        # Upsample
        cloth_last_flow  = F.interpolate(cloth_last_flow , scale_factor=2, mode='bilinear')
        shape_last_flow  = F.interpolate(shape_last_flow , scale_factor=2, mode='bilinear')
        flow_attn        = F.interpolate(flow_attn       , scale_factor=2, mode='bilinear')
        return shape_last_flow, cloth_last_flow, shape_delta_flow, cloth_delta_flow, flow_attn
    

class Backbone(nn.Module):
    def __init__(self, dim_in, channels, norm_type='BN'): # channels = [64, 128, 256]
        super().__init__()
        self.stage1 = ConvBlock(dim_in     , channels[0], stride=2, norm_type=norm_type)
        self.stage2 = ConvBlock(channels[0], channels[1], stride=2, norm_type=norm_type)
        self.stage3 = ConvBlock(channels[1], channels[2], stride=2, norm_type=norm_type)
        self.stage4 = ConvBlock(channels[2], channels[3], stride=2, norm_type=norm_type)
        
    def forward(self, x):
        out1 = self.stage1(x)           # x (64 , 128, 96)
        out2 = self.stage2(out1)        # x (96 , 64 , 48)
        out3 = self.stage3(out2)        # x (128, 32 , 24) 
        out4 = self.stage4(out3)        # x (256, 16 , 12) 
        return [out1, out2, out3, out4]
    

class FPN(nn.Module):
    def __init__(self, dim_ins, dim_out):
        super().__init__()
        # adaptive 
        self.adaptive = []
        for in_chns in list(reversed(dim_ins)):
            adaptive_layer = nn.Conv2d(in_chns, dim_out, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(dim_ins)):
            smooth_layer = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)
        return tuple(reversed(feature_list))


class AFlowNet(nn.Module):
    def __init__(self, channels, num_head=1, norm_type='BN'):
        super().__init__()
        self.num_head = num_head
        self.num_layers = len(channels)
        self.flow = nn.ModuleList([CascadeWarpingModule(channels[-1], num_head, norm_type) for _ in range(self.num_layers)]) 

    def forward(self, cloth, cloth_mask, shape_list, cloth_list):
        warping_masks = []
        warping_cloths = []
        cloth_last_flows = []
        shape_delta_flows = []
        for i in range(self.num_layers):
            shape_fea = shape_list[-(i+1)]
            cloth_fea = cloth_list[-(i+1)]

            # 預測外觀劉
            if i == 0:
                shape_last_flow, cloth_last_flow, shape_delta_flow, cloth_delta_flow, flow_attn = self.flow[i](shape_fea, cloth_fea)
            else:
                shape_last_flow, cloth_last_flow, shape_delta_flow, cloth_delta_flow, flow_attn = self.flow[i](shape_fea, cloth_fea, shape_last_flow, cloth_last_flow, flow_attn)
            
            # 對店內服裝和其遮罩進行縮放
            _, _, h, w = shape_last_flow.shape
            cloth_ = F.interpolate(cloth, size=(h, w), mode='nearest')
            cloth_mask_ = F.interpolate(cloth_mask, size=(h, w), mode='nearest')

            # 對店內服裝和其遮罩進行變形
            cloth_last_flow_ = F.grid_sample(shape_last_flow, cloth_last_flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode='border')
            cloth_ = AWarp(cloth_, cloth_last_flow_, flow_attn, self.num_head, padding_mode='border')
            cloth_mask_ = AWarp(cloth_mask_, shape_last_flow, flow_attn, self.num_head, padding_mode='zeros')

            # 儲存相關資料
            cloth_last_flows.append(cloth_last_flow_)
            shape_delta_flows.append(shape_delta_flow)
            warping_masks.append(cloth_mask_)
            warping_cloths.append(cloth_)

        return {
            'warping_masks':warping_masks,
            'warping_cloths':warping_cloths,
            'cloth_last_flows': cloth_last_flows, 
            'shape_delta_flows': shape_delta_flows, 
        }


class CAFWM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.channels = [64, 96, 128, 128]
        self.backbone_A = Backbone(3 , self.channels, norm_type='BN')
        self.backbone_B = Backbone(23, self.channels, norm_type='BN')
        self.FPN_A = FPN(self.channels, dim_out=self.channels[-1])
        self.FPN_B = FPN(self.channels, dim_out=self.channels[-1])
        self.dec_tryon = AFlowNet(self.channels, self.args.num_head, norm_type='BN')
    
    def forward(self, cloth, cloth_mask, person_shape):
        cloth_list = self.FPN_A(self.backbone_A(cloth)) 
        shape_list = self.FPN_B(self.backbone_B(person_shape))
        output = self.dec_tryon(cloth, cloth_mask, shape_list, cloth_list)
        return output


class LightweightGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lights_encoder = torch.nn.Sequential(
            ConvBlock(3 , 64, norm_type='IN'),
            ConvBlock(64, 64, norm_type='IN'),
            ConvBlock(64, 64, norm_type='IN'),
        )
        self.lights_decoder = torch.nn.Sequential(
            ConvBlock(64, 64, norm_type='IN'),
            ConvBlock(64, 4 , norm_type='IN'),
        )

    def forward(self, cloth, cloth_agnostic):
        cloth = self.lights_encoder(cloth)
        cloth_agnostic = self.lights_encoder(cloth_agnostic)
        gen_outputs = self.lights_decoder(cloth_agnostic + cloth)
        return gen_outputs
    