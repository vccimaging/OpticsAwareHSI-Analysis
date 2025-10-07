# This code is adapted from the SSTHyper repository by renweidian.
# Original source: https://github.com/MingyingLin/SSTHyper
# Accessed on: 2025-04-23

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import os
import warnings
# from thop import profile
# from torchstat import stat
# from torchsummary import summary
# from torch.nn.init import _calculate_fan_in_and_fan_out


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(dim)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(dim)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


# 自己的
class SS_SA(nn.Module):
    def __init__(
            self,
            dim,
            heads,
    ):
        super().__init__()
        self.num_heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = x_in
        b,c,h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        flattened_matrix = attn.view(b, self.num_heads, -1)
        top_values, top_indices = torch.topk(flattened_matrix, k=(C * C) * 1 // 2, dim=-1)
        sparattn1 = torch.full_like(flattened_matrix, float('-inf')).scatter_(-1, top_indices, top_values)
        top_values, top_indices = torch.topk(flattened_matrix, k=(C * C) * 2 // 3, dim=-1)
        sparattn2 = torch.full_like(flattened_matrix, float('-inf')).scatter_(-1, top_indices, top_values)
        top_values, top_indices = torch.topk(flattened_matrix, k=(C * C) * 3 // 4, dim=-1)
        sparattn3 = torch.full_like(flattened_matrix, float('-inf')).scatter_(-1, top_indices, top_values)
        top_values, top_indices = torch.topk(flattened_matrix, k=(C * C) * 4 // 5, dim=-1)
        sparattn4 = torch.full_like(flattened_matrix, float('-inf')).scatter_(-1, top_indices, top_values)

        sparattn1 = sparattn1.softmax(dim=-1).view(b, self.num_heads,C, C)
        sparattn2 = sparattn2.softmax(dim=-1).view(b, self.num_heads,C, C)
        sparattn3 = sparattn3.softmax(dim=-1).view(b, self.num_heads,C, C)
        sparattn4 = sparattn4.softmax(dim=-1).view(b, self.num_heads,C, C)
        sparattn = sparattn1 * self.attn1 + sparattn2 * self.attn2 +sparattn3 * self.attn3 + sparattn4 * self.attn4
        out = (sparattn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# MDTA
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# mst的
# class MS_MSA(nn.Module):
#     def __init__(
#             self,
#             dim,
#             heads,
#             dim_head =31
#     ):
#         super().__init__()
#         self.num_heads =heads
#         self.dim_head = dim_head
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
#         self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
#         self.proj = nn.Linear(dim_head* heads, dim, bias=True)
#         self.pos_emb = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#             GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#         )
#         self.dim = dim
#
#     def forward(self, x_in):
#         """
#         x_in: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         x_in = x_in.permute(0, 2, 3, 1)
#         b, h, w, c = x_in.shape
#         x = x_in.reshape(b,h*w,c)
#         q_inp = self.to_q(x)
#         k_inp = self.to_k(x)
#         v_inp = self.to_v(x)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
#                                 (q_inp, k_inp, v_inp))
#         v = v
#         # q: b,heads,hw,c
#         q = q.transpose(-2, -1)
#         k = k.transpose(-2, -1)
#         v = v.transpose(-2, -1)
#         q = F.normalize(q, dim=-1, p=2)
#         k = F.normalize(k, dim=-1, p=2)
#         attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
#         attn = attn * self.rescale
#         attn = attn.softmax(dim=-1)
#         x = attn @ v   # b,heads,d,hw
#         x = x.permute(0, 3, 1, 2)    # Transpose
#         x = x.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_c = self.proj(x).view(b, h, w, c)
#         out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#         out = out_c + out_p
#         return out.permute(0, 3, 1, 2)
#
class Split(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.g3 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        )

        # self.g3 = nn.Conv2d(dim , dim , 3, 1, 1, bias=False, groups=dim)
        # self.g4 = nn.Conv2d(dim , dim , 3, 1, 1, bias=False, groups=dim)
    def forward(self, x):
        x1, x2= x.chunk(2, dim=1)
        x1=self.g1(x1)
        x2=F.avg_pool2d(x2, kernel_size=2)
        x2=self.g2(x2)
        x2=F.interpolate(x2, scale_factor=2, mode='nearest')
        out =self.g3(x1*x2)
        # out = self.g3(x1*x2)
        # x3 = self.g3(x3*x2)
        # out = self.g4(x4*x3)
        # out = torch.cat([x1,x2,x3,x4],dim=1)
        return out

class CLFN(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.Conv2d(dim, dim* mult, 1, 1, bias=False),
            GELU(),
            Split(dim=dim),
            # GELU(),
            # nn.Conv2d(dim, dim, 1, 1, bias=False),
            # nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out

# class FeedForward(nn.Module):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
#             GELU(),
#             nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
#             GELU(),
#             nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
#         )
#
#     def forward(self, x):
#         """
#         x: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         out = self.net(x)
#         return out
## Gated-Dconv Feed-Forward Network (GDFN)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
class SSAB(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                # MS_MSA(dim=dim, heads=heads),
                # Attention(dim=dim, num_heads=heads),
                SS_SA(dim=dim, heads=heads),
                # PreNorm(dim,FeedForward(dim=dim))
                PreNorm(dim,CLFN(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # baseline
            x = ff(x) + x
        out = x
        return out



class SSAG(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4]):
        super(SSAG, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                SSAB(
                    dim=dim_stage, num_blocks=num_blocks[i],heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SSAB(
            dim=dim_stage, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                # SKFusion(dim=dim_stage // 2),
                SSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i],
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)
        # Encoder
        fea_encoder = []
        for (SSAB, FeaDownSample) in self.encoder_layers:
            fea =fea+SSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, DeSSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = fea + DeSSAB(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class SSTHyper(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(SSTHyper, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_SSAGs = [SSAG(dim=n_feat, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        self.ssags = nn.Sequential(*modules_SSAGs)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # x = self.initial_x(y)
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.ssags(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# def main():
#     model = SSTHyper().cuda()
#     input = torch.randn(1, 3, 128, 128).cuda()
#     try:
#         out = model(input)
#         print("Working:", out.shape)
#     except Exception as e:
#         print("Error:", e)

#     # summary(model.cuda(), (31,128,128))
#     # flops, params = profile(model, inputs=(input,))
#     # print('FLOPs = ' + str(flops / (1024 ** 3)) + 'G')
#     # print('Params = ' + str(params / 1e6) + 'M')
# #
# # model = SSTHyper().cuda()
# # summary(model.cuda(),(3,128,128))
# # # stat(model, (3, 128, 128))
# if __name__ == '__main__':
#     main()










