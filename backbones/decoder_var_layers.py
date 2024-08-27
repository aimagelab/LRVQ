# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from backbones.stt import DecoderLayer, MultiHeadAttention


class DecoderVarLayer(DecoderLayer):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, do_memory_attn=True,
                 dropout_memory=0.1, detach_memory=False):
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

        super(DecoderVarLayer, self).__init__(d_model, d_k, d_v, h, d_ff, dropout)
        self.variational_attn = MultiHeadAttention(h, d_model, d_k, d_v)
        self.drop_var = nn.Dropout(dropout)
        self.norm_var = nn.LayerNorm(d_model)
        self.do_memory_attn = do_memory_attn
        self.drop2 = nn.Dropout(dropout_memory)
        self.detach_memory = detach_memory

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask, z):
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

        if self.do_memory_attn:
            # residual block with (cross) attention through time between decoder and encoder/memory
            # detach memory to avoid backprop through encoder
            if self.detach_memory:
                memory = memory.detach()
            tgt_res2, _ = self.memory_attn(tgt, memory, memory, memory_mask)
            tgt = tgt + self.drop2(tgt_res2)
            tgt = self.norm2(tgt)

        # residual block with cross attention through time between decoder and variational embedding
        tgt_res_var, _ = self.variational_attn(tgt, z, z, tgt_mask)
        tgt = tgt + self.drop_var(tgt_res_var)
        tgt = self.norm_var(tgt)

        # residual block with masked self-attention through space
        tgt_res3, spatial_attn = self.memory_attn_space(tgt, tgt, tgt,
                                                        space_mask)
        tgt = tgt + self.drop3(tgt_res3)
        tgt = self.norm3(tgt)

        # residual block with position-wise FF
        tgt = tgt + self.drop4(self.feed_forward(tgt))
        tgt = self.norm4(tgt)

        return tgt, temporal_attn, spatial_attn


class DecoderVarLayerConcat(DecoderVarLayer):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, do_memory_attn=True,
                 dropout_memory=0.1, detach_memory=False):
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

        super(DecoderVarLayerConcat, self).__init__(d_model, d_k, d_v, h, d_ff, dropout, do_memory_attn=True,
                                                    dropout_memory=0.1, detach_memory=False)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask, z):
        """
        :param tgt: target input tensor (batch, seq_len, d_model)
        :param memory: source tensor from encoder (batch, seq_len, d_model)
        :param tgt_mask: temporal attention subsequent mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        :param space_mask: spatial attention mask for both memory and target (seq_len, 1, batch)
        """
        tgt = torch.cat([tgt, z], dim=-1)
        tgt = self.proj(tgt)
        # residual block with masked self attention through time
        tgt_res1, temporal_attn = self.masked_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.drop1(tgt_res1)
        tgt = self.norm1(tgt)

        if self.do_memory_attn:
            # residual block with (cross) attention through time between decoder and encoder/memory
            # detach memory to avoid backprop through encoder
            if self.detach_memory:
                memory = memory.detach()
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


class DecoderVarLayerSum(DecoderVarLayer):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, do_memory_attn=True,
                 dropout_memory=0.1, detach_memory=False):
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

        super(DecoderVarLayerSum, self).__init__(d_model, d_k, d_v, h, d_ff, dropout, do_memory_attn,
                                                 dropout_memory, detach_memory=detach_memory)

    def forward(self, tgt, memory, tgt_mask, memory_mask, space_mask, z):
        """
        :param tgt: target input tensor (batch, seq_len, d_model)
        :param memory: source tensor from encoder (batch, seq_len, d_model)
        :param tgt_mask: temporal attention subsequent mask for target (batch, seq_len, seq_len)
        :param memory_mask: temporal attention mask for source/memory (batch, 1, seq_len)
        :param space_mask: spatial attention mask for both memory and target (seq_len, 1, batch)
        """

        # residual block with cross attention through time between decoder and variational embedding
        tgt_res_var = self.drop_var(tgt + z)
        tgt = self.norm_var(tgt_res_var)

        # # residual block with masked self attention through time
        tgt_res1, temporal_attn = self.masked_self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.drop1(tgt_res1)
        tgt = self.norm1(tgt)

        if self.do_memory_attn:
            # residual block with (cross) attention through time between decoder and encoder/memory
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