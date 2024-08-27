# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
from torch import nn

from backbones.pixel_cnn import MyEmbedding, DecoderLayer, PixelNet
from backbones.stt import LinearEmbedding, PositionalEncoding


class DecoderLayerWithAdapative(DecoderLayer):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dropout, diffusion_step, spatial=False, emb_type='adalayernorm_abs'):
        super().__init__(d_model, d_k, d_v, h, d_ff, dropout)
        self.spatial = spatial
        self.norm1 = AdaLayerNorm(d_model, diffusion_step, emb_type)
        self.norm2 = AdaLayerNorm(d_model, diffusion_step, emb_type)
        self.norm3 = AdaLayerNorm(d_model, diffusion_step, emb_type)
        self.norm4 = AdaLayerNorm(d_model, diffusion_step, emb_type)

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask, t):
        """
        :param tgt: target input tensor (batch, seq_len, d_model)
        :param memory: source tensor from encoder (batch, seq_len, d_model)
        :param tgt_mask: temporal attention subsequent mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        """

        # residual block with masked self attention through time
        tgt_res1, temporal_attn = self.masked_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.drop1(tgt_res1)
        tgt = self.norm1(tgt, t)

        # residual block with (cross) attention through time between decoder and encoder/memory
        tgt_res2, _ = self.memory_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.drop2(tgt_res2)
        tgt = self.norm2(tgt, t)

        if self.spatial:
            tgt_res3, spatial_attn = self.memory_attn_space(tgt, tgt, tgt,
                                                            space_mask)
            tgt = tgt + self.drop3(tgt_res3)
            tgt = self.norm3(tgt, t)

        # residual block with position-wise FF
        tgt = tgt + self.drop4(self.feed_forward(tgt))
        tgt = self.norm4(tgt, t)

        return tgt


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.num_steps) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        self.embed_type = emb_type
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.diff_step = diffusion_step

    def forward(self, x, timestep):
        if timestep[0] >= self.diff_step and self.embed_type == "adalayernorm_abs":
            _emb = self.emb.weight.mean(dim=0, keepdim=True).repeat(len(timestep), 1)
            emb = self.linear(self.silu(_emb)).unsqueeze(1)
        else:
            emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class TransformerAdaptiveLayer(PixelNet):
    def __init__(self,
                 encoder_layers,
                 nc_out,
                 nc,
                 d_ff,
                 categorical_embedding: bool = False,
                 dropout=0.1,
                 n_heads=8,
                 src_nc=None,
                 n_layers=3,
                 init_weights=True,
                 categorical_embedding_temporal_dim=None,
                 diffusion_step=100,
                 spatial=False,
                 emb_type='adalayernorm_abs',
                 *args, **kwargs):
        super(TransformerAdaptiveLayer, self).__init__(encoder_layers)
        self.src_nc = src_nc if src_nc is not None else nc
        self.nc = nc
        self.decoder_embedding = LinearEmbedding(nc, nc)
        self.decoder_positional = PositionalEncoding(nc, dropout)
        self.encoder_embedding = LinearEmbedding(self.src_nc, nc) if self.src_nc != nc else lambda x: x
        self.encoder_positional = PositionalEncoding(nc, dropout) if self.src_nc != nc else lambda x: x
        if categorical_embedding:
            self.categorical_embedding = MyEmbedding(nc_out + 1, nc,
                                                     categorical_embedding_temporal_dim).requires_grad_(True)
        else:
            self.categorical_embedding = lambda x, y: x.squeeze(1).permute(0, 2, 1)

        self.out_projection = nn.Linear(nc, nc_out)
        self.decoder_layers = nn.ModuleList([DecoderLayerWithAdapative(nc, nc, nc, n_heads, d_ff, dropout,
                                                                       diffusion_step, spatial=spatial,
                                                                       emb_type=emb_type)
                                             for _ in range(n_layers)])
        if init_weights:
            self.init_weights(self.parameters())

    def decode(self, y, x, memory_mask, y_causal_mask, tgt_space_mask, t):
        # concatenate x and y on temporal dimension (dim=1)
        # create causal mask
        out = y
        for dec_layer in self.decoder_layers:
            out = dec_layer(out, x, y_causal_mask, memory_mask, tgt_space_mask, t)
        return out

    def forward(self, x, y, t, src_mask=None, space_mask=None, tgt_space_mask=None):
        # find most similar element from Memory
        x_out = self.encoder_positional(self.encoder_embedding(x))
        x_out = self.encode(x_out, src_mask[:, :x_out.shape[1], :x_out.shape[1]],
                            space_mask[:x_out.shape[1]])
        # find corresponding embedding for each pixel
        y_embedded = self.categorical_embedding(y.unsqueeze(1), x.shape[0])
        y_out = y_embedded.permute(0, 2, 1)
        # create causal mask for y
        y_mask = torch.ones(y_out.shape[1], y_out.shape[1]).repeat(
            x_out.shape[0], 1, 1).to(y_out.device)
        # debug
        # y_causal_mask = torch.ones(y_out.shape[1], y_out.shape[1]).to(y.device).repeat(
        #     x_out.shape[0], 1, 1)
        y_out = self.decoder_positional(self.decoder_embedding(y_out))
        # create causal mask
        y_out = self.decode(y_out, x_out, src_mask, y_mask, tgt_space_mask, t)
        # take only the cls token
        y_out = self.out_projection(y_out)

        return y_out.contiguous().transpose(1, 2)
