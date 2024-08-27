# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from einops import einsum
from torch import nn

from backbones.stt import MultiHeadAttention, PositionalEncoding


class MySelfAttention(MultiHeadAttention):
    def __init__(self, h, d_model, d_k, d_v, normalize=True, dropout=0):
        super(MySelfAttention, self).__init__(h, d_model, d_k, d_v)
        self.norm = nn.LayerNorm(d_model)
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attention_mask):
        """
        :param queries: (batch, seq_len, d_model)
        :param keys: (batch, seq_len, d_model)
        :param values: (batch, seq_len, d_model)
        :param attention_mask: (batch, seq_len, seq_len)
        """
        # compute attention
        y, attn = super(MySelfAttention, self).forward(queries, keys, values, attention_mask)
        y = queries + self.dropout(y)
        y = self.norm(y)
        return y, attn


class VariationalQuantizationLayer(nn.Module):
    def __init__(self, d_model=512, d_proj=512, dff=2048, dropout=0, k_way=64, d_k=32, d_v=32, n_heads=1, beta=0.5,
                 force_hard=False, detach_memory=False, **kwargs):
        """
        Single transformer Decoder layer: masked multi-head self-attention + multi-head cross-attention with
                source (memory) + position-wise FFN, all inside residual blocks.

        :param d_model: model output dimensionality
        """

        super(VariationalQuantizationLayer, self).__init__()
        # parameters
        self.d_model = d_model
        self.d_proj = d_proj
        self.n_e = k_way
        self.beta = beta
        self.force_hard = force_hard
        self.force_gumble = False
        self.detach_memory = detach_memory
        # blocks
        # initialize memory like in VQ-VAE and use it as a codebook. The codebook is initialized with random values
        self.embedding = nn.Embedding(self.n_e, self.d_model).requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        # only one head is used
        self.sha = MultiHeadAttention(n_heads, d_model, d_k, d_v)

        self.norm = nn.LayerNorm(d_model)
        # initialize also PixelCNN to use as prior

        self.embedding_positional = PositionalEncoding(self.d_model, dropout)
        self.embedding_sha = MySelfAttention(n_heads, d_model, d_k, d_v)

        if self.d_proj != self.d_model:
            self.proj = nn.Linear(d_model, self.d_proj)
        else:
            self.proj = nn.Identity()

    def create_conditioned_embedding(self, x_enc):
        # create a mask based on x_enc
        mask = torch.ones(x_enc.shape[0], 1, x_enc.shape[1]).to(x_enc.device)
        # repeat embedding for each element in batch dimension
        embedding = self.embedding.weight.unsqueeze(0).repeat(x_enc.shape[0], 1, 1)
        # apply positional encoding
        embedding = self.embedding_positional(embedding)
        if self.detach_memory:
            x_enc = x_enc.detach()
        # apply sha
        embedding_sha, _ = self.sha(embedding, x_enc, x_enc, mask)
        embedding = embedding + self.dropout(embedding_sha)
        embedding = self.norm(embedding)

        embedding_mask = torch.ones(embedding.shape[0], 1, embedding.shape[1]).to(embedding.device)
        embedding, _ = self.embedding_sha(embedding, embedding, embedding, embedding_mask)

        embedding = self.proj(embedding)
        return embedding

    def extract_zq(self, min_encodings, embedding):
        reshaped_min_encodings = min_encodings.reshape(embedding.shape[0], min_encodings.shape[0] // embedding.shape[0],
                                                       -1)
        # get quantized latent vectors
        if embedding.dim() == 4:
            # enter inside if the embedding has a time dimension (the third dimension)
            # the second dimension is the number of time steps
            z_q = []
            for i in range(reshaped_min_encodings.shape[1]):
                z_q.append(torch.matmul(reshaped_min_encodings[:, [i], :], embedding[:, :, i, :]))
            z_q = torch.cat(z_q, dim=1)
        else:
            z_q = torch.matmul(
                min_encodings.reshape(embedding.shape[0], min_encodings.shape[0] // embedding.shape[0], -1)
                , embedding)
        return z_q

    def forward(self, x_enc, z):
        """

        @param z:
        @param x_enc:
        @return:
        """
        loss = torch.zeros(1).to(z.device)
        embedding = self.create_conditioned_embedding(x_enc)
        min_encodings, min_encoding_indices, logits = self.vqvae_min_encodings(z, embedding)
        # get quantized latent vectors
        z_q = self.extract_zq(min_encodings, embedding)

        # compute loss for embedding
        loss += self.beta * torch.mean(torch.mean((z_q.detach() - z) ** 2, -1)) \
                + torch.mean(torch.mean((z_q - z.detach()) ** 2, -1))
        # enter here only if static table is optimized without dynamic table
        z_q = z + (z_q - z).detach()
        # calculate a different perplexity
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity, min_encodings, min_encoding_indices.reshape(*z.shape[:-1]), embedding

    def vqvae_min_encodings(self, z, embedding):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = ((embedding.unsqueeze(2) - z.unsqueeze(1)) ** 2).sum(-1).permute(0, 2, 1).contiguous().view(-1, self.n_e)

        # find the closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        return min_encodings, min_encoding_indices, -d


class VariationalQuantizationLora(VariationalQuantizationLayer):
    def __init__(self, d_model=512, d_proj=512, dff=2048, dropout=0, k_way=64, d_k=32, d_v=32, n_heads=1, beta=0.5,
                 force_hard=False, detach_memory=False, lora_r=1, lora_alpha=1,
                 **kwargs):
        super().__init__(d_model, d_proj, dff, dropout, k_way, d_k, d_v, n_heads, beta,
                         force_hard, detach_memory, **kwargs)
        # overwrite params
        self.sha = MultiHeadAttention(n_heads, d_model, d_k, d_v)

        self.norm = nn.LayerNorm(d_model)
        # initialize also PixelCNN to use as prior

        self.embedding_positional = PositionalEncoding(self.d_model, dropout)
        self.embedding_sha = MySelfAttention(n_heads, d_model, d_k, d_v)

        # create embedding consisting pursuing the LoRa paper: W = W0 + AB; W0 (k_way, d_model), A (k
        self.lora_alpha = lora_alpha
        self.r = lora_r
        # init lora A with uniform distribution
        self.embedding = nn.Embedding(self.n_e, self.d_model).requires_grad_(True)
        self.lora_A = nn.Parameter(self.embedding.weight.new_ones((self.r, k_way)).data.uniform_(-1.0,
                                                                                                  1.0))
        self.lora_B = nn.Parameter(self.embedding.weight.new_zeros((self.r, d_model)))
        self.scaling = self.lora_alpha  # fare in modo che lo scaling sia o 0.25 al max o 1
        self.proj_embedding = nn.Linear(self.d_proj, self.d_proj)

    def create_conditioned_embedding(self, x_enc):
        """
        @param x_enc:
        @return:
        """
        mask = torch.ones(x_enc.shape[0], 1, x_enc.shape[1]).to(x_enc.device)
        # repeat embedding for each element in batch dimension
        lora_b_cond = self.lora_B.unsqueeze(0).repeat(x_enc.shape[0], 1, 1)
        # apply positional encoding
        lora_b_cond = self.embedding_positional(lora_b_cond)
        if self.detach_memory:
            x_enc = x_enc.detach()
        # apply sha
        lora_b_cond_cha, _ = self.sha(lora_b_cond, x_enc, x_enc, mask)
        lora_b_cond = lora_b_cond + self.dropout(lora_b_cond_cha)
        lora_b_cond = self.norm(lora_b_cond)

        embedding_mask = torch.ones(lora_b_cond.shape[0], 1, lora_b_cond.shape[1]).to(lora_b_cond.device)
        lora_b_cond, _ = self.embedding_sha(lora_b_cond, lora_b_cond, lora_b_cond, embedding_mask)
        dynamic_table = einsum(self.lora_A, lora_b_cond, "r k, B r d -> B k d")
        # lora_A (r, k_way), embedding (b, r, d_model)
        dynamic_table = dynamic_table / torch.norm(dynamic_table, dim=-1, keepdim=True)
        embedding = self.embedding.weight / torch.norm(self.embedding.weight, dim=-1, keepdim=True)
        embedding = embedding.unsqueeze(0) + self.scaling * dynamic_table
        embedding = self.proj(embedding)
        embedding = self.proj_embedding(embedding)
        return embedding