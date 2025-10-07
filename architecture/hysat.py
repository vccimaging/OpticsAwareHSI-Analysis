# This code is adapted from the HySAT repository
# Original source: https://gitee.com/hongyuan-wang-bit/models/blob/hysat_mm23/research/cv/hysat/model/HySAT_MindSpore.py
# Accessed on: 2025-04-23

# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Utility blocks
# -------------------------------------------------------------
class PreNorm(nn.Module):
    """Apply LayerNorm *before* the wrapped function."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# -------------------------------------------------------------
# Multi‑Spectral Multi‑Head Self‑Attention (MS‑MSA)
# -------------------------------------------------------------
class MS_MSA(nn.Module):
    """The MindSpore implementation works in NHWC format internally. We keep that convention
    here for a faithful port – the surrounding model will convert to/from NCHW.
    """

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = heads

        # ========================= Q / K / V depth‑wise conv kernels =========================
        def _param():
            return nn.Parameter(torch.empty(heads, 1, 3, 3))

        self.to_q_share1 = _param()
        self.to_q_share2 = _param()
        self.to_q_share3 = _param()

        self.to_k_share1 = _param()
        self.to_k_share2 = _param()
        self.to_k_share3 = _param()

        self.to_v_share1 = _param()
        self.to_v_share2 = _param()
        self.to_v_share3 = _param()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity="relu")

        # depth‑wise positional encoding
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(heads, heads, kernel_size=3, padding=1)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))

        self.proj = nn.Linear(dim_head * heads, dim)

    # -------------------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------------------
    def _dw_conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Depth‑wise 3×3 conv with grouped weight used in the original code."""
        b, c, h, w = x.shape  # x is BCHW here
        weight = weight.repeat_interleave(self.dim_head, dim=0)  # (groups,1,3,3) -> (c,1,3,3)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_in: Tensor of shape (B, H, W, C) – NHWC
        Returns:
            Tensor of shape (B, H, W, C)
        """
        b, h, w, c = x_in.shape

        # to BCHW for convs / positional encoding
        x = x_in.permute(0, 3, 1, 2)
        x = x + self.pos_emb(x)

        # --------------------------------- Q, K, V ---------------------------------
        q = self._dw_conv(x, self.to_q_share1)
        q = self._dw_conv(q, self.to_q_share2)
        q = self._dw_conv(q, self.to_q_share3)

        k = self._dw_conv(x, self.to_k_share1)
        k = self._dw_conv(k, self.to_k_share2)
        k = self._dw_conv(k, self.to_k_share3)

        v = self._dw_conv(x, self.to_v_share1)
        v = self._dw_conv(v, self.to_v_share2)
        v = self._dw_conv(v, self.to_v_share3)

        # flatten spatial, reshape for heads
        def _reshape(t):
            t = t.permute(0, 2, 3, 1).reshape(b, h * w, self.num_heads, self.dim_head)
            return t.permute(0, 2, 3, 1)  # (B, heads, dim_head, HW)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        # L2‑normalise channels
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # channel‑wise attention
        attn = torch.matmul(k, q.transpose(-2, -1))  # (B, heads, dim_head, dim_head)

        # spectral‑wise recalibration term
        y = self.avg_pool(v.reshape(b, -1, h, w)).view(b, self.num_heads, self.dim_head)  # (B, heads, dim_head)
        attn_v = self.conv(y)  # (B, heads, dim_head)

        attn = attn * self.rescale + torch.diag_embed(attn_v) * self.rescale2
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v)  # (B, heads, dim_head, HW)
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, -1)  # (B, HW, C)

        out = self.proj(x).view(b, h, w, c)
        return out


# -------------------------------------------------------------
# Feed‑Forward Network (depth‑wise convolutions)
# -------------------------------------------------------------
class Feedforward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, padding=1, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: NHWC
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        return x.permute(0, 2, 3, 1)


# -------------------------------------------------------------
# MSAB block: (MS‑MSA + FFN) × N
# -------------------------------------------------------------
class MSAB(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, Feedforward(dim))
            ]) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: NCHW
        x = x.permute(0, 2, 3, 1)  # to NHWC
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)  # back to NCHW


# -------------------------------------------------------------
# Multi‑Stage Spectral Transformer (MST)
# -------------------------------------------------------------
class MST(nn.Module):
    def __init__(self,
                 in_dim: int = 31,
                 out_dim: int = 31,
                 dim: int = 31,
                 stage: int = 2,
                 num_blocks: List[int] = [2, 4, 4]):
        super().__init__()
        self.stage = stage

        # Input embedding
        self.embedding = nn.Conv2d(in_dim, dim, 3, padding=1)

        # -------------------------- Encoder --------------------------
        encoder_layers = []
        dim_stage = dim
        for i in range(stage):
            encoder_layers.append(nn.ModuleList([
                MSAB(dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, kernel_size=4, stride=2, padding=1)
            ]))
            dim_stage *= 2
        self.encoder_layers = nn.ModuleList(encoder_layers)

        # -------------------------- Bottleneck -----------------------
        self.bottleneck = MSAB(dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # -------------------------- Decoder --------------------------
        decoder_layers = []
        for i in range(stage):
            decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, kernel_size=2, stride=2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1),
                MSAB(dim_stage // 2, dim_head=dim, heads=(dim_stage // 2) // dim, num_blocks=num_blocks[stage - 1 - i])
            ]))
            dim_stage //= 2
        self.decoder_layers = nn.ModuleList(decoder_layers)

        # Output mapping
        self.mapping = nn.Conv2d(dim, out_dim, 3, padding=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: NCHW
        feat = self.embedding(x)
        enc_feats = []
        for msab, down in self.encoder_layers:
            feat = msab(feat)
            enc_feats.append(feat)
            feat = down(feat)

        feat = self.bottleneck(feat)

        for i, (up, fuse, msab) in enumerate(self.decoder_layers):
            feat = up(feat)
            feat = fuse(torch.cat([feat, enc_feats[self.stage - 1 - i]], dim=1))
            feat = msab(feat)

        out = self.mapping(feat) + x
        return out


# -------------------------------------------------------------
# HySAT – overall network
# -------------------------------------------------------------
class HySAT(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super().__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1)
        modules_body = [MST(dim=31, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]


# -------------------------------------------------------------
# Quick sanity check – run a forward pass
# -------------------------------------------------------------
# if __name__ == "__main__":
#     with torch.no_grad():
#         inp = torch.randn(1, 3, 128, 128)
#         net = HySAT()
#         out = net(inp)
#         print("Output shape:", out.shape)  # expected: (1, 31, 128, 128)


#### Original Mindspore code

# import mindspore
# import numpy as np
# import math
# import mindspore.nn as nn
# import mindspore.ops as ops
# from mindspore.common.initializer import initializer, HeNormal

# from mindspore import load_checkpoint, load_param_into_net
# class PreNorm(nn.Cell):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm((dim,))

#     def construct(self, x, *args, **kwargs):
#         x = self.norm(x)
#         return self.fn(x, *args, **kwargs)

# class GELU(nn.Cell):
#     def construct(self, x):
#         return ops.gelu(x)
    
# class MS_MSA(nn.Cell):
#     def __init__(
#             self,
#             dim,
#             dim_head,
#             heads,
#     ):
#         super().__init__()
#         self.num_heads = heads
#         self.dim_head = dim_head


#         self.to_q_share=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_q_share2=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_q_share3=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))


#         self.to_k_share=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_k_share2=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_k_share3=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))

#         self.to_v_share=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_v_share2=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))
#         self.to_v_share3=mindspore.Parameter(initializer(HeNormal(),[heads,1,3,3],mindspore.float32))

#         k=3
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.conv = nn.Conv1d(heads, heads, kernel_size=k, padding=(k - 1) // 2, pad_mode='pad') 

#         self.rescale = mindspore.Parameter(ops.ones((heads, 1, 1),mindspore.float32))
#         self.rescale2 = mindspore.Parameter(ops.ones((heads, 1, 1),mindspore.float32))
#         self.proj = nn.Dense(self.dim_head * heads, dim, has_bias=True)

#         self.pos_emb = nn.SequentialCell(
#             nn.Conv2d(dim, dim, 3, 1,'pad', 1, group=dim),
#             GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 'pad', 1, group=dim),
#         )
#         self.dim = dim
#         self.norm=ops.L2Normalize(-1)

#     def construct(self, x_in):
#         """
#         x_in: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         b, h, w, c = x_in.shape

#         x = x_in.permute(0,3,1,2)
#         pos_embedding=self.pos_emb(x)
#         x = x + pos_embedding

#         x_toq=x 
#         x_tok=x 
#         x_tov=x 

#         # Token Independent Mapping
#         x_toq=ops.conv2d(x_toq,self.to_q_share.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_toq=ops.conv2d(x_toq,self.to_q_share2.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_toq=ops.conv2d(x_toq,self.to_q_share3.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         q_inp=x_toq.permute(0,2,3,1).reshape(b,h*w,-1)

#         x_tok=ops.conv2d(x_tok,self.to_k_share.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_tok=ops.conv2d(x_tok,self.to_k_share2.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_tok=ops.conv2d(x_tok,self.to_k_share3.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         k_inp=x_tok.permute(0,2,3,1).reshape(b,h*w,-1)

#         x_tov=ops.conv2d(x_tov,self.to_v_share.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_tov=ops.conv2d(x_tov,self.to_v_share2.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         x_tov=ops.conv2d(x_tov,self.to_v_share3.repeat_interleave(self.dim_head,0),groups=self.dim_head*self.num_heads,padding=1,pad_mode='pad')
#         v_inp=x_tov.permute(0,2,3,1).reshape(b,h*w,-1)

#         q=q_inp.reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)
#         k=k_inp.reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)
#         v=v_inp.reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)

#         v = v
#         # q: b,heads,hw,c
#         q = q.swapaxes(-2, -1)
#         k = k.swapaxes(-2, -1)
#         v = v.swapaxes(-2, -1)
#         q = self.norm(q)
#         k = self.norm(k)
#         attn = (k @ q.swapaxes(-2, -1))   # A = K^T*Q
#         y=self.avg_pool(v.reshape(b,-1,h,w))
#         y=y.reshape(b,self.num_heads,self.dim_head)
#         attn_v = self.conv(y)

#         # Spectral-wise Recalibration
#         attn = attn * self.rescale +ops.diag_embed(attn_v)*self.rescale2

#         attn = ops.softmax(attn)
#         x = attn @ v   # b,heads,d,hw
#         x = x.permute(0, 3,1,2).reshape(b,h*w,-1)    # Transpose

#         out_c = self.proj(x).reshape(b,h,w,c)

#         return out_c
    

# class Feedforward(nn.Cell):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.SequentialCell(
#             nn.Conv2d(dim, dim * mult, 1, 1),
#             GELU(),
#             nn.Conv2d(dim * mult, dim * mult, 3, 1,'pad', 1, group=dim * mult),
#             GELU(),
#             nn.Conv2d(dim * mult, dim, 1, 1),
#         )

#     def construct(self, x):
#         """
#         x: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#         out = self.net(x.permute(0, 3, 1, 2))
#         return out.permute(0, 2, 3, 1)

# class MSAB(nn.Cell):
#     def __init__(
#             self,
#             dim,
#             dim_head,
#             heads,
#             num_blocks,
#     ):
#         super().__init__()
#         self.blocks = nn.CellList([])
#         for _ in range(num_blocks):
#             self.blocks.append(nn.CellList([
#                 MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
#                 PreNorm(dim, Feedforward(dim=dim))
#             ]))

#     def construct(self, x):
#         """
#         x: [b,c,h,w]
#         return out: [b,c,h,w]
#         """
#         x = x.permute(0, 2, 3, 1)
#         for (attn, ff) in self.blocks:
#             x = attn(x) + x
#             x = ff(x) + x
#         out = x.permute(0, 3, 1, 2)
#         return out
    

# class MST(nn.Cell):
#     def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4]):
#         super(MST, self).__init__()
#         self.dim = dim
#         self.stage = stage

#         # Input projection
#         self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1,'pad', 1)

#         # Encoder
#         self.encoder_layers = nn.CellList([])
#         dim_stage = dim
#         for i in range(stage):
#             self.encoder_layers.append(nn.CellList([
#                 MSAB(
#                     dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
#                 nn.Conv2d(dim_stage, dim_stage * 2, 4, 2,'pad', 1),
#             ]))
#             dim_stage *= 2

#         # Bottleneck
#         self.bottleneck = MSAB(
#             dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

#         # Decoder
#         self.decoder_layers = nn.CellList([])
#         for i in range(stage):
#             self.decoder_layers.append(nn.CellList([
#                 nn.Conv2dTranspose(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0,has_bias=True),
#                 nn.Conv2d(dim_stage, dim_stage // 2, 1, 1),
#                 MSAB(
#                     dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
#                     heads=(dim_stage // 2) // dim),
#             ]))
#             dim_stage //= 2

#         # Output projection
#         self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1,'pad', 1)


#     def construct(self, x):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """

#         # Embedding
#         fea = self.embedding(x)

#         # Encoder
#         fea_encoder = []
#         for (MSAB, FeaDownSample) in self.encoder_layers:
#             fea = MSAB(fea)
#             fea_encoder.append(fea)
#             fea = FeaDownSample(fea)

#         # Bottleneck
#         fea = self.bottleneck(fea)

#         # Decoder
#         for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
#             fea = FeaUpSample(fea)
#             fea = Fution(ops.cat([fea, fea_encoder[self.stage-1-i]], axis=1))
#             fea = LeWinBlcok(fea)

#         # Mapping
#         out = self.mapping(fea) + x

#         return out
    

# class HySAT(nn.Cell):
#     def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
#         super().__init__()
#         self.stage = stage
#         self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3,pad_mode='pad', padding=(3 - 1) // 2)
#         modules_body = [MST(dim=31, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
#         self.body = nn.SequentialCell(*modules_body)
#         self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,pad_mode='pad')

#     def construct(self, x):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h_inp, w_inp = x.shape
#         hb, wb = 8, 8
#         pad_h = (hb - h_inp % hb) % hb
#         pad_w = (wb - w_inp % wb) % wb
#         x = ops.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
#         x = self.conv_in(x)
#         h = self.body(x)
#         h = self.conv_out(h)
#         h += x
#         return h[:, :, :h_inp, :w_inp]
    
# if __name__=='__main__':
#     inp=mindspore.Tensor(ops.randn(1,3,128,128),mindspore.float32)
#     net=HySAT()
#     model=mindspore.Model(net)
#     print(model.predict(inp).shape)
