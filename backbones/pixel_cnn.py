# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torch.functional import F

from backbones.stt import MultiHeadAttention, PositionWiseFeedForward

from utils.utilities import init_weights


class MyEmbedding(nn.Module):
    def __init__(self, nc_out, nc_embedding, temporal_dim=None):
        super().__init__()
        self.temporal_dim = temporal_dim

        if temporal_dim is None:
            temporal_dim = 1

        self.embeddings = nn.ModuleList([nn.Embedding(nc_out, nc_embedding) for _ in range(temporal_dim)])

    def forward(self, x, bs=None):
        out = []
        if len(self.embeddings) == 1:
            out = self.embeddings[0](x.long())
            out = out.permute(0, 1, 3, 2)
        else:
            for i in range(x.shape[-1]):
                temp_out = self.embeddings[i](x[:, :, i].long())
                out.append(temp_out)
            out = torch.stack(out, dim=-1)
        # x = super().forward(x.long())
        # x = x.permute(0, 1, 3, 2)
        # assert x.shape[1] == 1
        out = out.squeeze(1)
        return out


class MyFakeEmbedding(nn.Module):
    def __init__(self, nc_out):
        super().__init__()
        self.nc_out = nc_out
        self.embedding_start_token = nn.Parameter(torch.randn(1, nc_out, 1))

    def forward(self, x: torch.Tensor, bs: int = None):
        start_token = self.embedding_start_token.repeat(bs, 1, 1)
        if x is not None:
            return torch.cat([start_token, x], dim=-1)
        else:
            return start_token


def sample_sample(logits):
    probs = F.softmax(logits[:, -1, :], -1)
    return probs.multinomial(1).squeeze().data


def sample_topk(logits, k=2):
    # set to zero all logits below the top-k
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    indices_to_remove = logits < torch.min(top_k_logits[:, [-1]], -1)[0][..., None]
    logits[indices_to_remove] = -float('Inf')
    probs = F.softmax(logits[:, -1, :], -1)
    return probs.multinomial(1).squeeze().data


def sample_nucleus(logits, p=0.9, return_all=False):
    # set to zero all logits below the top-k
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    if return_all:
        probs = F.softmax(logits, -1)
        # probs has 3 dimensions: batch_size, seq_len, vocab_size, cannot use probs.multinomial(1)
        return torch.cat([torch.multinomial(probs[:, i, :], 1) for i in range(probs.shape[1])], dim=1)
    else:
        probs = F.softmax(logits[:, -1, :], -1)
        return probs.multinomial(1).squeeze().data


def sample_argmax(logits):
    probs = F.softmax(logits[:, -1, :], -1)
    return probs.argmax(-1).data


class PixelNet(nn.Module):

    def __init__(self, encoder_layers: nn.ModuleList, *args, **kwargs):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.sample_method = kwargs.get('sample_method', 'sample')

    def sample(self, x, y):
        # sample z from PixelCNN1D using x_enc as conditioning
        indices = self.forward(x.permute(0, 2, 1), y)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(x.device)
        min_encodings.scatter_(1, indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(y.shape)
        return z_q

    def generate_complete(self, x, y, length):
        param = next(self.parameters())
        if y is None:
            y = torch.zeros(x.shape[0], length, 1).to(param.device)
        for i in range(length):
            logits = self.forward(x.permute(0, 2, 1), y.permute(0, 2, 1))
            probs = F.softmax(logits[:, i, :], -1)
            y.data[:, i, 0].copy_(
                probs.multinomial(1).squeeze().data
            )
        return y[..., 0]

    def generate(self, x, y, embedding_vqvae=None, src_mask=None, space_mask=None, tgt_space_mask=None):
        logits = self.forward(x, y.permute(0, 2, 1) if y is not None else y, embedding_vqvae=embedding_vqvae,
                              src_mask=src_mask,
                              space_mask=space_mask, tgt_space_mask=tgt_space_mask)
        probs = globals()['sample_' + self.sample_method](logits)
        return probs

    def forward(self, x, y, embedding_vqvae=None, src_mask=None, space_mask=None, tgt_space_mask=None):
        return NotImplementedError

    def encode(self, x, src_mask, space_mask):
        output = x

        for layer in self.encoder_layers:
            output, tp_attn, sp_attn = layer(output, src_mask, space_mask)
        return output

    def init_weights(self, params):
        init_weights(params)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, spatial=False, add_memory=False,
                 *args, **kwargs):
        """
        Single transformer Decoder layer: masked multi-head self-attention + multi-head cross-attention with
                source (memory) + position-wise FFN, all inside residual blocks.

        :param d_model: model output dimensionality
        :param d_k: queries and keys dimensionality
        :param d_v: values dimensionality
        :param h: number of attention heads
        :param d_ff: position-wise FFN inner-layer dimensionality
        :param dropout: dropout probability
        """

        super(DecoderLayer, self).__init__()
        self.spatial = spatial
        self.add_memory = add_memory
        self.masked_self_attn = MultiHeadAttention(h, d_model, d_k, d_v)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.memory_attn = MultiHeadAttention(h, d_model, d_k, d_v)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.memory_attn_space = MultiHeadAttention(h, d_model, d_k, d_v, through_space=True)
        self.drop3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.drop4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)
        if add_memory:
            self.attn_with_memory = MultiHeadAttention(h, d_model, d_k, d_v)
            self.drop_mem = nn.Dropout(dropout)
            self.norm_mem = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask, memory_future=None):
        """
        :param tgt: target input tensor (batch, seq_len, d_model)
        :param memory: source tensor from encoder (batch, seq_len, d_model)
        :param tgt_mask: temporal attention subsequent mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        """

        # residual block with masked self attention through time
        tgt_res1, temporal_attn = self.masked_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.drop1(tgt_res1)
        tgt = self.norm1(tgt)

        # residual block with (cross) attention through time between decoder and encoder/memory
        tgt_res2, _ = self.memory_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.drop2(tgt_res2)
        tgt = self.norm2(tgt)

        if self.spatial:
            tgt_res3, spatial_attn = self.memory_attn_space(tgt, tgt, tgt,
                                                            space_mask)
            tgt = tgt + self.drop3(tgt_res3)
            tgt = self.norm3(tgt)

        if self.add_memory:
            assert memory_future is not None
            tgt_res_mem, _ = self.attn_with_memory(tgt, memory_future, memory_future,
                                                   torch.ones([tgt_mask.shape[0], 1, memory_future.shape[1]]).to(
                                                       tgt_mask.device))
            tgt = tgt + self.drop_mem(tgt_res_mem)
            tgt = self.norm_mem(tgt)

        # residual block with position-wise FF
        tgt = tgt + self.drop4(self.feed_forward(tgt))
        tgt = self.norm4(tgt)

        return tgt
