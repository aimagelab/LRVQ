# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from torch.nn import Module
from utils.utilities import attention_mask, parse_bool


def get_parser(parser) -> ArgumentParser:
    """
    Defines the arguments of the model.
    """

    parser.add_argument('--enc_in_dim', default=2, type=int, help='# of features of the encoder input')
    parser.add_argument('--dec_in_dim', default=3, type=int, help='# of features of the decoder input')
    parser.add_argument('--dec_out_dim', default=3, type=int, help='# of features of the decoder output')
    parser.add_argument('--d_model', default=32, type=int, help='# of features of the embeddings')
    parser.add_argument('--d_ff', default=128, type=int, help='# of features for the position-wise FF')
    parser.add_argument('--n_layers', default=1, type=int, help='# of encoder/decoder layers')
    parser.add_argument('--n_heads', default=8, type=int, help='# of attention heads')
    parser.add_argument('--dropout', default=1e-1, type=float, help='Dropout probability')
    parser.add_argument('--flip_pos_encodings', type=parse_bool,  default=False, help='Flip positional encodings')

    return parser


class LinearEmbedding(Module):
    def __init__(self, orig_size, d_model=512):
        """
        Input and Output Embeddings: brings encoder or decoder input to the model dimensionality.

        :param orig_size: input tensor dimensionality
        :param d_model: model output dimensionality
        """

        super(LinearEmbedding, self).__init__()
        self.w = nn.Linear(orig_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: original tensor (batch, seq_len, orig_size)
        """

        return self.w(x) * math.sqrt(self.d_model)


class PositionalEncoding(Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        """
        Positional Encoding: two sinusoidal functions for even and odd positions respectively.

        :param d_model: model output dimensionality dimensionality
        :param dropout: dropout probability
        :param max_len: maximum sequence length
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even pe positions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd pe positions

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, flip=False):
        """
        :param x: input tensor (batch, seq_len, d_model)
        :param flip: before addition, flip the positional encodings
        """

        # add only the right number of values (=seq_len ---> x.size(1)) from the original 5000-length table
        return self.dropout(x + self.pe[:, :x.size(1)].flip(1)) if flip else self.dropout(x + self.pe[:, :x.size(1)])


class PositionWiseFeedForward(Module):
    def __init__(self, d_model, d_ff, dropout=0.1, d_out_model=None):
        """
        Position-wise feed-forward NN. Two linears with a ReLU activation in between (+ eventual dropout).

        :param d_model: model output dimensionality
        :param d_ff: inner layer dimensionality
        :param dropout: dropout probability
        """

        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model if d_out_model is None else d_out_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: input tensor (batch, seq_len, d_model)
        """

        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ScaledDotProductAttention(Module):
    def __init__(self, h, d_model, d_k, d_v):
        """
        Multi-head 'Scaled Dot-Product Attention' module

        :param h: number of heads
        :param d_model: model output dimensionality
        :param d_k: query and keys dimensionality
        :param d_v: values dimensionality
        """

        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

    def forward(self, queries, keys, values, attention_mask=None):
        """
        :param queries: queries (batch, nq, d_model)
        :param keys: key vectors matrix (batch, nk, d_model)
        :param values: value vectors matrix (batch, nk, d_model)
        :param attention_mask: mask over attention values (batch, h, nq, nk)

        what attention does is:
        1. compute the similarity between the query and the key
            formula: similarity = query * key
        2. normalize the similarity by dividing it by the square root of the dimensionality of the key
            formula: normalized_similarity = similarity / sqrt(d_k)
        3. apply the mask to the normalized similarity
        4. apply softmax to the normalized similarity
        5. compute the output
            formula: output = softmax(normalized_similarity) * value

        """

        batch, nq, _ = queries.shape
        _, nk, _ = keys.shape

        q = self.fc_q(queries).view(batch, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (batch, h, nq, d_k)
        k = self.fc_k(keys).view(batch, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (batch, h, d_k, nk)
        v = self.fc_v(values).view(batch, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (batch, h, nk, d_v)

        unnorm_attn = torch.matmul(q, k) / np.sqrt(self.d_k)

        # eventual masking over attention values
        masked_attn = torch.masked_fill(unnorm_attn, (attention_mask == 0),
                                        value=-np.inf) if attention_mask is not None else unnorm_attn

        attn = torch.softmax(masked_attn, -1)
        attn = torch.masked_fill(attn, torch.isnan(attn), value=.0)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch, nq,
                                                                          self.h * self.d_v)  # (batch, nq, h*d_v)
        out = self.fc_o(out)  # (batch, nq, d_model)

        return out, unnorm_attn


class SpatialScaledDotProductAttention(Module):
    def __init__(self, h, d_model, d_k, d_v):
        """
        Multi-head 'Scaled Dot-Product Attention' module

        :param h: number of heads
        :param d_model: model output dimensionality
        :param d_k: query and keys dimensionality
        :param d_v: values dimensionality
        """

        super(SpatialScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

    def forward(self, queries, keys, values, attention_mask=None):
        """
        :param queries: queries (batch, nq, d_model)
        :param keys: key vectors matrix (batch, nk d_model)
        :param values: value vectors matrix (batch, nv, d_model)
        :param attention_mask: mask over attention values (batch, h, nq, nk)
        """

        batch, nq, _ = queries.shape
        _, nk, _ = keys.shape

        q = self.fc_q(queries).view(batch, nq, self.h, self.d_k).permute(1, 2, 0, 3)  # (nq, h, batch, d_k)
        k = self.fc_k(keys).view(batch, nk, self.h, self.d_k).permute(1, 2, 3, 0)  # (nk, h, d_k, batch)
        v = self.fc_v(values).view(batch, nk, self.h, self.d_v).permute(1, 2, 0, 3)  # (nk, h, batch, d_v)

        unnorm_attn = torch.matmul(q, k) / np.sqrt(self.d_k)

        # eventual masking over attention values
        masked_attn = torch.masked_fill(unnorm_attn, (attention_mask == 0),
                                        value=-np.inf) if attention_mask is not None else unnorm_attn

        attn = torch.softmax(masked_attn, -1)
        attn = torch.masked_fill(attn, torch.isnan(attn), value=.0)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(nq, batch,
                                                                          self.h * self.d_v)  # (nq, batch, h*d_v)
        out = self.fc_o(out).transpose(0, 1)  # (batch, nq, d_model)

        return out, unnorm_attn


class MultiHeadAttention(Module):
    def __init__(self, h, d_model, d_k, d_v, through_space=False):
        """
        Multi-head attention module (N.B. without dropout and layernorm! -> apply them outside).

        :param h: number of heads
        :param d_model: model output dimensionality
        :param d_k: query and keys dimensionality
        :param d_v: values dimensionality
        """

        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention = ScaledDotProductAttention(h, d_model, d_k, d_v) if not through_space \
            else SpatialScaledDotProductAttention(h, d_model, d_k, d_v)

    def forward(self, queries, keys, values, attention_mask):
        """
        :param queries: queries (batch, nq, d_model)
        :param keys: key vectors matrix (batch, nk, d_model)
        :param values: value vectors matrix (batch, nk, d_model)
        :param attention_mask: mask over attention values (batch, nq, nk)
        """

        # repeat mask for all heads
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        return self.attention(queries, keys, values, attention_mask)


class EncoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1):
        """
        Single transformer Encoder layer: multi-head self-attention + position-wise FFN, all inside residual blocks.

        :param d_model: model output dimensionality
        :param d_k: queries and keys dimensionality
        :param d_v: values dimensionality
        :param h: number of attention heads
        :param d_ff: position-wise FFN inner-layer dimensionality
        :param dropout: dropout probability
        """

        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.self_attn_time = MultiHeadAttention(h, d_model, d_k, d_v)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn_space = MultiHeadAttention(h, d_model, d_k, d_v, through_space=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.drop3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, mask, space_mask):
        """
        :param x: input tensor (batch, seq_len, d_model)
        :param mask: temporal attention mask
        :param space_mask: spatial attention mask
        """

        # first residual block with self-att across time
        x_res1, temporal_attn = self.self_attn_time(x, x, x, mask)
        x = x + self.drop1(x_res1)
        x = self.norm1(x)

        # second residual block with self-att across space
        x_res2, spatial_attn = self.self_attn_space(x, x, x, space_mask)
        x = x + self.drop2(x_res2)
        x = self.norm2(x)

        # third residual with position-wise FF
        x = x + self.drop3(self.feed_forward(x))
        x = self.norm3(x)

        return x, temporal_attn, spatial_attn


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, *args, **kwargs):
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
        self.disable_memory_attn = kwargs.get('disable_memory_attn', False)

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask):
        """
        :param tgt: target input tensor (batch, seq_len, d_model)
        :param memory: source tensor from encoder (batch, seq_len, d_model)
        :param tgt_mask: temporal attention subsequent mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        :param space_mask: spatial attention mask for both memory and target (seq_len, 1, batch)
        """

        # residual block with masked self attention through time
        tgt_res1, temporal_attn = self.masked_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.drop1(tgt_res1)
        tgt = self.norm1(tgt)

        # residual block with (cross) attention through time between decoder and encoder/memory
        if not self.disable_memory_attn:
            tgt_res2, _ = self.memory_attn(tgt, memory, memory, memory_mask)
            tgt = tgt + self.drop2(tgt_res2)
            tgt = self.norm2(tgt)

        # residual block with masked self-attention through space
        tgt_res3, spatial_attn = self.memory_attn_space(tgt, tgt, tgt,
                                                        space_mask)
        tgt = tgt + self.drop3(tgt_res3)
        tgt = self.norm3(tgt)

        # residual block with position-wise FF
        tgt = tgt + self.drop4(self.feed_forward(tgt))
        tgt = self.norm4(tgt)

        return tgt, temporal_attn, spatial_attn


class STTModel(Module):
    NAME = 'stt'

    def __init__(self, args, init_weights=True):
        """ Spatio-Temporal Transformer model """

        super(STTModel, self).__init__()

        assert args.d_model % args.n_heads == 0

        self.model_type = 'Spatio-Temporal Transformer'
        self.encoder_input_dim = args.enc_in_dim
        self.decoder_input_dim = args.dec_in_dim
        self.decoder_output_dim = args.dec_out_dim
        self.d_model = args.d_model
        self.d_k = args.d_model // args.n_heads  # d_q equal d_k
        self.d_v = args.d_model // args.n_heads
        self.d_ff = args.d_ff
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.flip_pos_encodings = args.flip_pos_encodings

        # modules
        self.encoder_embedding = LinearEmbedding(self.encoder_input_dim, self.d_model)
        self.encoder_positional = PositionalEncoding(self.d_model, self.dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads,
                                                          self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.decoder_embedding = LinearEmbedding(self.decoder_input_dim, self.d_model)
        self.decoder_positional = PositionalEncoding(self.d_model, self.dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads,
                                                          self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.out_projection = nn.Linear(self.d_model, self.decoder_output_dim)

        # initialize model weights
        if init_weights:
            self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask, space_mask):
        """
        :param src: encoder input tensor (batch, seq_len, d_model)
        :param src_mask: temporal attention mask for source (batch, 1, seq_len)
        :param space_mask: spatial attention mask for source (seq_len, 1, batch)
        """
        attn_matrices = []
        output = src

        for layer in self.encoder_layers:
            output, tp_attn, sp_attn = layer(output, src_mask, space_mask)
        attn_matrices.extend([tp_attn, sp_attn])

        return output, attn_matrices

    def decode(self, tgt, memory, tgt_mask, memory_mask, space_mask):
        """
        :param tgt: decoder input tensor (batch, seq_len, d_model)
        :param memory: encoder stack final output (batch, seq_len, d_model)
        :param tgt_mask: subsequent temporal attention mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        :param space_mask: spatial attention mask for memory  and target (seq_len, 1, batch)
        """
        output = tgt
        attn_matrices = []

        for layer in self.decoder_layers:
            output, tp_attn, sp_attn = layer(output, memory, tgt_mask, memory_mask, space_mask)
        attn_matrices.extend([tp_attn, sp_attn])

        return output, attn_matrices
