# This code is adapted from the MSFN repository.
# Original source: https://github.com/Matsuri247/MSFN-for-Spectral-Super-Resolution/blob/main/train_code/architecture/MSFN.py
# Accessed on: 2025-04-23

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from timm.models.layers import DropPath, to_2tuple
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.modules.utils import _pair
#from torchinfo import summary

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Spectral Fusion Module
# replace original skip-connection
class SpectralFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.mlp_gap_enc = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel)
        )
        self.mlp_gmp_enc = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel)
        )
        self.mlp_gap_dec =  nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel)
        )
        self.mlp_gmp_dec = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel)
        )
        self.conv = nn.Conv2d(channel*4, channel, 1, 1, bias=False)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x_enc, x_dec):
        # GAP: Global average pooling
        # GMP: Global mean pooling
        enc_shape = (x_enc.size(2),x_enc.size(3))
        dec_shape = (x_dec.size(2),x_dec.size(3))
        avg_pool_enc = F.avg_pool2d(x_enc, kernel_size=enc_shape)
        max_pool_enc = F.max_pool2d(x_enc, kernel_size=enc_shape)
        avg_pool_dec = F.avg_pool2d(x_dec, kernel_size=dec_shape)
        max_pool_dec = F.max_pool2d(x_dec, kernel_size=dec_shape)
        channel_attn_enc1 = self.mlp_gap_enc(avg_pool_enc)
        channel_attn_enc2 = self.mlp_gmp_enc(max_pool_enc)
        channel_attn_dec1 = self.mlp_gap_dec(avg_pool_dec)
        channel_attn_dec2 = self.mlp_gmp_dec(max_pool_dec)
        channel_attn_enc = (channel_attn_enc1 + channel_attn_enc2) / 2.0
        channel_attn_dec = (channel_attn_dec1 + channel_attn_dec2) / 2.0
        scale_enc = torch.sigmoid(channel_attn_enc).unsqueeze(2).unsqueeze(3).expand_as(x_enc)
        scale_dec = torch.sigmoid(channel_attn_dec).unsqueeze(2).unsqueeze(3).expand_as(x_dec)
        x_enc_after = x_enc * scale_enc
        x_dec_after = x_dec * scale_dec
        out = self.conv(torch.cat([x_enc,x_dec,x_enc_after,x_dec_after],dim=1))
        # channel_attn = (channel_attn_enc_sum + channel_attn_dec_sum) / 2.0
        # scale = torch.sigmoid(channel_attn).unsqueeze(2).unsqueeze(3).expand_as(x_enc)
        # x_after_channel = x_enc * scale
        # out = self.relu(x_after_channel)
        return out

# Spatial Fusion Module
# replace original skip-connection
class SpatialFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_gap_enc = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gmp_enc = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gap_dec = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv_gmp_dec = nn.Conv2d(1, 1, 3, 1, 1)
        self.conv = nn.Conv2d(channel * 4, channel, 1, 1, bias=False)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x_enc, x_dec):
        # GAP: Global average pooling
        # GMP: Global mean pooling
        avg_pool_enc = torch.mean(x_enc, dim=1, keepdim=True)
        max_pool_enc, _ = torch.max(x_enc, dim=1, keepdim=True)
        avg_pool_dec = torch.mean(x_dec, dim=1, keepdim=True)
        max_pool_dec, _ = torch.max(x_dec, dim=1, keepdim=True)
        channel_attn_enc1 = self.conv_gap_enc(avg_pool_enc)
        channel_attn_enc2 = self.conv_gmp_enc(max_pool_enc)
        channel_attn_dec1 = self.conv_gap_dec(avg_pool_dec)
        channel_attn_dec2 = self.conv_gmp_dec(max_pool_dec)
        channel_attn_enc = (channel_attn_enc1 + channel_attn_enc2) / 2.0
        channel_attn_dec = (channel_attn_dec1 + channel_attn_dec2) / 2.0
        scale_enc = torch.sigmoid(channel_attn_enc).expand_as(x_enc)
        scale_dec = torch.sigmoid(channel_attn_dec).expand_as(x_dec)
        x_enc_after = x_enc * scale_enc
        x_dec_after = x_dec * scale_dec
        out = self.conv(torch.cat([x_enc,x_dec,x_enc_after,x_dec_after],dim=1))
        # channel_attn = (channel_attn_enc_sum + channel_attn_dec_sum) / 2.0
        # scale = torch.sigmoid(channel_attn).expand_as(x_enc)
        # x_after_channel = x_enc * scale
        # out = self.relu(x_after_channel)
        return out

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)) # nn.Parameter自定义一个可训练参数
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim), # dw conv
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim), # dw conv
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)  # [b,h,w,c]变为[b,n,c]
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q,k,v格式：(batch,head,n,dimension), n是hw, dimension即为c
        # map是集体映射（说白了就是把list或者tuple里的所有元素都批处理，方法为map里指定的函数）
        # rearrange是维度重排，[b,n,c]变为[b,h,n,d], h为head数量，d为每个head的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,n(hw),d(c)
        q = q.transpose(-2, -1) # torch的transpose即交换两个维度,参数为维度下标
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2) # 在特定维度做归一化，dim为维度索引，p=2二范数
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q  # np.dot() 或 @ 均为矩阵乘法 【token计算是以哪个维度为准的关键就在于此】
        attn = attn * self.rescale # 这个rescale就是原论文里的自注意力调整参数σj，该参数可学习
        attn = attn.softmax(dim=-1) # torch softmax需要指定维度
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head) # 把多头输出拼接，得到输出
        out_c = self.proj(x).view(b, h, w, c) # 通过投射层proj之后再用view方法resize成(b,h,w,c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # 通过V生成位置编码
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2)) # 此时变为 [b,c,h,w]
        return out.permute(0, 2, 3, 1) # 此时变为 [b,h,w,c]


class MSpectralAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))  # LayerNorm
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1) # 此时变为 [b,h,w,c]，因为算注意力需要这个格式
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2) # 此时变回 [b,c,h,w]
        return out

class MSpatialAB(nn.Module):
    def __init__(
            self,
            dim,
            stage,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SwinTransformerBlock(dim=dim, num_heads=2 ** stage, window_size=8, shift_size=0), # W
                SwinTransformerBlock(dim=dim, num_heads=2 ** stage, window_size=8, shift_size=4), # SW
                PreNorm(dim, FeedForward(dim=dim))  # LayerNorm
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_size = x.shape[2], x.shape[3] # H,W
        x = x.permute(0, 2, 3, 1) # 此时变为 [b,h,w,c]，因为算注意力需要这个格式
        for (w_attn, sw_attn, ff) in self.blocks:
            x = w_attn(x, x_size) + x
            x = sw_attn(x, x_size) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2) # 此时变回 [b,c,h,w]
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution=(128,128), window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # shift window时才创建mask
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution) # 不用在意此处创建的Mask
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        B, H, W, C = x.shape
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # 【根据输入图像的实际大小来创建相应大小的Mask，而不是在__init__中预先定死Mask大小】
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SSFN(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, num_blocks=[1, 1, 1]):
        super(SSFN, self).__init__()
        self.dim = dim

        # input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        '''Spectral
        '''
        # Encode Spectral
        dim_stage = dim
        self.Spectral_Block_0 = MSpectralAB(dim=dim_stage, num_blocks=num_blocks[0], dim_head=dim, heads= dim_stage // dim)
        self.Spectral_down_0 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        self.Spectral_Block_1 = MSpectralAB(dim=dim_stage, num_blocks=num_blocks[1], dim_head=dim, heads= dim_stage // dim)
        self.Spectral_down_1 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        # Bottleneck
        self.bottleneck_Spectral = MSpectralAB(dim=dim_stage, num_blocks=num_blocks[-1], dim_head=dim, heads= dim_stage // dim)

        # Decode Spectral
        self.Spectral_up_0 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0) # UpSample：HW减半，通道数翻倍
        self.Spectral_Block_2 = MSpectralAB(dim=dim_stage // 2, num_blocks=num_blocks[1], dim_head=dim, heads= (dim_stage // 2) // dim)
        self.SpectralFM_0 = SpectralFM(dim_stage // 2)
        dim_stage = dim_stage // 2

        self.Spectral_up_1 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0) # UpSample：HW减半，通道数翻倍
        self.Spectral_Block_3 = MSpectralAB(dim=dim_stage // 2, num_blocks=num_blocks[0], dim_head=dim, heads= (dim_stage // 2) // dim)
        self.SpectralFM_1 = SpectralFM(dim_stage // 2)

        '''Spatial
        '''
        # Encode Spatial
        dim_stage = dim
        self.Spatial_Block_0 = MSpatialAB(dim=dim_stage, stage=0, num_blocks=num_blocks[0])
        self.Spatial_down_0 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        self.Spatial_Block_1 = MSpatialAB(dim=dim_stage, stage=1, num_blocks=num_blocks[1])
        self.Spatial_conv_0 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False)
        self.Spatial_down_1 = nn.Conv2d(dim_stage, dim_stage*2, 4, 2, 1, bias=False) # DownSample：HW减半，通道数翻倍
        dim_stage *= 2

        # Bottleneck
        self.Spatial_conv_1 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False)
        self.bottleneck_Spatial = MSpatialAB(dim=dim_stage, stage=2, num_blocks=num_blocks[-1])

        # Decoder Spatial
        self.Spatial_up_0 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.Spatial_Block_2 = MSpatialAB(dim=dim_stage // 2, stage=1, num_blocks=num_blocks[1])
        self.SpatialFM_0 = SpatialFM(dim_stage // 2)
        dim_stage = dim_stage // 2

        self.Spatial_up_1 = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.Spatial_Block_3 = MSpatialAB(dim=dim_stage // 2, stage=0, num_blocks=num_blocks[0])
        self.SpatialFM_1 = SpatialFM(dim_stage // 2)

        # output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        '''Spectral
        '''
        # Spectral Encoder
        x_0_0 = self.Spectral_Block_0(fea)
        fea = self.Spectral_down_0(x_0_0)

        x_1_0 = self.Spectral_Block_1(fea)
        fea = self.Spectral_down_1(x_1_0)

        # Bottleneck
        b_0 = self.bottleneck_Spectral(fea)

        # Spectral Decoder
        fea = self.Spectral_up_0(b_0)
        fea = self.SpectralFM_0(x_1_0, fea)
        x_2_0 = self.Spectral_Block_2(fea)

        fea = self.Spectral_up_1(x_2_0)
        fea = self.SpectralFM_1(x_0_0, fea)
        x_3_0 = self.Spectral_Block_3(fea)

        '''Spatial
        '''
        # Spatial Encoder
        x_0_1 = self.Spatial_Block_0(x_3_0)
        fea = self.Spatial_down_0(x_0_1)
        fea = self.Spatial_conv_0(torch.cat([fea, x_2_0], dim=1))

        x_1_1 = self.Spatial_Block_1(fea)
        fea = self.Spatial_down_1(x_1_1)

        # Bottleneck
        fea = self.Spatial_conv_1(torch.cat([fea, b_0], dim=1))
        b_1 = self.bottleneck_Spatial(fea)

        # Spatial Decoder
        fea = self.Spatial_up_0(b_1)
        fea = self.SpatialFM_0(x_1_1, fea)
        x_2_1 = self.Spatial_Block_2(fea)

        fea = self.Spatial_up_1(x_2_1)
        fea = self.SpatialFM_1(x_0_1, fea)
        x_3_1 = self.Spatial_Block_3(fea)

        # Mapping
        out = self.mapping(x_3_1) + x

        return out

class MSFN(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(MSFN, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)

        modules_body = [SSFN(dim=31, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        # reflect是以左右边界为起点，进行镜像填充（不含起点）
        # F.pad(x,pad,mode): pad内有2n个参数，代表对倒数n个维度进行扩充（4个时候是pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)）
        x_in = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x_in = self.conv_in(x_in)
        h = self.body(x_in)
        h = self.conv_out(h)
        h += x_in

        return h[:, :, :h_inp, :w_inp] # 多的部分不要


# if __name__ == '__main__':
#     #Their Calling Code
#     model = MSFN(stage=2).cuda()

#     #Testing
#     x = torch.randn(1, 3, 256, 256).cuda()

#     try:
#         out = model(x)
#         print("Working:", out.shape)
#     except Exception as e:
#         print("Error:", e)

    #model = MSFN(stage=2)
    #summary(model, input_size=(1,3,256,256))















