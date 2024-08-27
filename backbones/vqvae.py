# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from argparse import ArgumentParser

from backbones.stt import STTModel, LinearEmbedding, PositionalEncoding, DecoderLayer
from torch.functional import F
from copy import deepcopy
import importlib

from utils.utilities import parse_bool

VQ_MODULES = {
    x: getattr(importlib.import_module('backbones.quantization_layers'), x) for x in
    importlib.import_module('backbones.quantization_layers').__dir__() if 'Variational' in x

}

PIXEL_MODULES = {
    x: getattr(importlib.import_module('backbones.diffusion_transformer'), x) for x in
    importlib.import_module('backbones.diffusion_transformer').__dir__() if 'Pixel' in x
}


def get_parser(parser) -> ArgumentParser:
    """
    Defines the arguments of the model.
    """

    parser.add_argument('--enc_in_dim', default=2, type=int, help='# of features of the encoder input')
    parser.add_argument('--dec_in_dim', default=3, type=int, help='# of features of the decoder input')
    parser.add_argument('--dec_out_dim', default=2, type=int, help='# of features of the decoder output')
    parser.add_argument('--d_model', default=32, type=int, help='# of features of the embeddings')  #
    parser.add_argument('--d_ff', default=128, type=int, help='# of features for the position-wise FF')
    parser.add_argument('--n_layers', default=1, type=int, help='# of encoder/decoder layers')  #
    parser.add_argument('--n_heads', default=8, type=int, help='# of attention heads')
    parser.add_argument('--dropout', default=0.01, type=float, help='Dropout probability')
    parser.add_argument('--flip_pos_encodings', type=parse_bool, default=False, help='Flip positional encodings')
    parser.add_argument('--k_way', default=64, type=int, help='K-way vq-vae embedding size')  #
    parser.add_argument('--beta', default=0.25, type=float, help='Beta for vq-vae loss')
    parser.add_argument('--do_memory_attn', type=parse_bool, default=True, help='Do memory attention in decoder')
    parser.add_argument('--dropout_memory', default=0.01, type=float, help='Dropout probability for memory attention')
    parser.add_argument('--detach_memory', type=parse_bool, default=False,
                        help='Detach memory in cross attention layer')
    parser.add_argument('--pixelcnn_n_layers', default=1, type=int, help='PixelCNN layers')
    parser.add_argument('--dropout_decoder', default=0.01, type=float, help='Dropout probability')
    parser.add_argument('--vq_module', default='VariationalQuantizationLoraFuture2',
                        choices=list(VQ_MODULES.keys()), help='VQ type')
    parser.add_argument('--lora_r', default=1, type=int, help='Rank of Lora matrix')  #
    parser.add_argument('--lora_alpha', default=1, type=float, help='Weight for sum of deltaW in lora vq layer')  #
    parser.add_argument('--project_codebook_dim', type=int, default=0, help='Project codebook')  #
    parser.add_argument('--disable_decoder_memory_attn', type=parse_bool, default=True)

    # discrete token predictor params
    parser.add_argument('--pixelcnn', type=str, help='PixelCNN model to use',
                        choices=list(PIXEL_MODULES.keys()), default='PixelDiffusionTransformer')
    parser.add_argument('--pixelcnn_nc', default=None, type=int, help='PixelCNN channels')  #
    parser.add_argument('--pixelcnn_dropout', default=0, type=float, help='PixelCNN dropout')  #
    parser.add_argument('--pixelcnn_spatial_masking', type=parse_bool, default=False)
    parser.add_argument('--diff_auxiliary_loss_weight', type=float, default=1e-3)  #
    parser.add_argument('--diff_adaptive_auxiliary_loss', type=parse_bool, default=True)  #
    parser.add_argument('--diff_diffusion_step', type=int, default=100)  #
    parser.add_argument('--diff_schedule', type=str, default='linear', choices=['linear', 'cosine'])  #
    parser.add_argument('--diff_sample_type', type=str, default='regular')
    parser.add_argument('--sample_power', type=int, default=1)
    parser.add_argument('--pixelcnn_categorical_temporal_dim', type=parse_bool, default=False)
    return parser


class VQVAE(STTModel):
    NAME = 'vqvae'

    def __init__(self, args):

        super().__init__(args, init_weights=False)
        self.variational_quantization = None
        self.pixelcnn = None
        assert args.d_model % args.n_heads == 0

        self.model_type = 'Spatio-Temporal Global Variational Transformer'
        self.encoder_response_embedding = LinearEmbedding(self.decoder_input_dim - 1, self.d_model)
        self.encoder_response_positional = PositionalEncoding(self.d_model, self.dropout)
        self.encoder_response_layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads,
                                                                   self.d_ff, self.dropout) for _ in
                                                      range(self.n_layers)])
        self.beta = args.beta
        self.pixelcnn_type = args.pixelcnn
        self.k_way = args.k_way
        self.pixelcnn_nc = args.pixelcnn_nc
        self.pixelcnn_dropout = args.pixelcnn_dropout
        self.pixelcnn_train_with_original_src = args.pixelcnn_train_with_original_src
        self.do_memory_attn = args.do_memory_attn
        self.dropout_memory = args.dropout_memory
        self.src_tokenizer = nn.Identity()
        self.detach_memory = args.detach_memory
        self.pixelcnn_n_layers = args.pixelcnn_n_layers
        self.dropout_decoder = args.dropout_decoder
        self.vq_module = args.vq_module
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.project_codebook_dim = args.project_codebook_dim
        self.codebook_dim = self.d_model
        # self.encoder_tgt_masking_p = args.encoder_tgt_masking_p
        self.pixelcnn_spatial_masking = args.pixelcnn_spatial_masking
        self.pred_len = args.pred_len
        self.diff_auxiliary_loss_weight = args.diff_auxiliary_loss_weight
        self.diff_adaptive_auxiliary_loss = args.diff_adaptive_auxiliary_loss
        self.diff_diffusion_step = args.diff_diffusion_step
        self.diff_schedule = args.diff_schedule
        self.diff_sample_type = args.diff_sample_type
        self.sample_power = args.sample_power
        self.pixelcnn_categorical_temporal_dim = args.pixelcnn_categorical_temporal_dim
        self.disable_decoder_memory_attn = args.disable_decoder_memory_attn

        self.skipping_init_params = ['variational_quantization.lora_B',
                                     'variational_quantization.lora_A',
                                     'masked_token']

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model,
                         self.d_k, self.d_v, self.n_heads,
                         self.d_ff, self.dropout_decoder,
                         self.do_memory_attn,
                         self.dropout_memory, self.detach_memory,
                         disable_memory_attn=self.disable_decoder_memory_attn,
                         ) for _ in
            range(self.n_layers)

        ])

        if self.project_codebook_dim:
            self.proj_encoder_src = nn.Linear(self.d_model, self.project_codebook_dim)
            self.proj_encoder_tgt = nn.Linear(self.d_model, self.project_codebook_dim)
            self.unproj_encoder_tgt = nn.Linear(self.project_codebook_dim, self.d_model)
            self.codebook_dim = self.project_codebook_dim
        else:
            self.proj_encoder_src = nn.Identity()
            self.proj_encoder_tgt = nn.Identity()
            self.unproj_encoder_tgt = nn.Identity()

        self.init_variational_quantization(pred_len=args.pred_len)

        self.init_pixelcnn(pixelcnn_type=args.pixelcnn)
        self.init_weights()

    def init_variational_quantization(self, **kwargs):
        self.variational_quantization = VQ_MODULES[self.vq_module](
            self.d_model,
            self.codebook_dim,
            d_k=self.d_k,
            d_v=self.d_v,
            n_heads=self.n_heads,
            k_way=self.k_way,
            beta=self.beta,
            dropout=self.dropout,
            detach_memory=False,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            n_layers=self.n_layers,
            **kwargs
        )

    #
    def init_weights(self):
        # avoid initializing the embedding layer
        for p in self.named_parameters():
            if 'variational_quantization.embedding.weight' in p[0]:
                self.variational_quantization.embedding.weight.data.uniform_(-1.0,
                                                                             1.0)
            elif 'masked_token' in p[0]:
                torch.nn.init.normal_(self.masked_token, std=.02)
            elif p[0] in self.skipping_init_params:
                pass
            else:
                if p[1].dim() > 1:
                    nn.init.xavier_uniform_(p[1])

    def init_pixelcnn(self, pixelcnn_type, init_weights=True, **kwargs):
        self.pixelcnn = PIXEL_MODULES[pixelcnn_type](
            encoder_layers=deepcopy(self.encoder_layers),
            nc_out=self.k_way,
            nc=self.d_model if self.pixelcnn_nc is None else self.pixelcnn_nc,
            d_ff=self.d_ff,
            dropout=self.pixelcnn_dropout,
            src_nc=self.d_model if not self.pixelcnn_train_with_original_src else
            self.encoder_input_dim,
            nc_embedding=self.k_way,
            categorical_embedding=True,
            n_layers=self.pixelcnn_n_layers,
            init_weights=init_weights,
            spatial=self.pixelcnn_spatial_masking,
            categorical_embedding_temporal_dim=None,
            auxiliary_loss_weight=self.diff_auxiliary_loss_weight,
            adaptive_auxiliary_loss=self.diff_adaptive_auxiliary_loss,
            alpha_init_type=self.diff_schedule,
            diffusion_step=self.diff_diffusion_step,
            sample_type=self.diff_sample_type,
            content_seq_len=self.pred_len,
            sample_power=self.sample_power,
        )

    def encode_response(self, tgt, memory, tgt_mask, memory_mask, space_mask):

        attn_matrices = []
        output = tgt

        for layer in self.encoder_response_layers:
            output, tp_attn, sp_attn = layer(output, memory, tgt_mask, memory_mask, space_mask)
        attn_matrices.extend([tp_attn, sp_attn])

        return output, attn_matrices

    def decode(self, tgt, memory, tgt_mask, memory_mask, space_mask, z=None):

        # the influence of z on the decoder could be implemented as:
        # 1. add a cross attention layer between the decoder and the z

        output = tgt
        attn_matrices = []

        for layer in self.decoder_layers:
            output, tp_attn, sp_attn = layer(output, memory, tgt_mask, memory_mask, space_mask, z)
        attn_matrices.extend([tp_attn, sp_attn])

        return output, attn_matrices

    def decodify(self, tgt, encoding, z_q, tgt_mask, src_mask, tgt_space_mask):
        decoder_input = self.decoder_positional(z_q)
        decoding, dec_attn_matrices = super().decode(decoder_input, encoding, tgt_mask, src_mask, tgt_space_mask)
        return decoding, dec_attn_matrices

    def variational_through_pixelcnn(self, encoding, encoding_tgt, prev_indices, previous_z, encoder_input,
                                     src_mask, space_mask, tgt_space_mask, inpainting_tensor=None, **kwargs):
        embedding = self.variational_quantization.create_conditioned_embedding(encoding)
        # adding a zero vector to previous indices to enable the generation of the last token
        model_input = torch.ones(encoding.shape[0], 1, 1).to(encoding.device) * self.k_way
        if prev_indices is not None:
            model_input = torch.cat([model_input, prev_indices], dim=1)
        # create model input: indices or embeddings selected by indices
        inputs = {
            'x': encoder_input,
            'y': model_input,
            'src_mask': src_mask,
            'space_mask': space_mask,
            'tgt_space_mask': tgt_space_mask,
            'inpainting_tensor': inpainting_tensor
        }
        min_encoding_indices = self.pixelcnn.generate(**inputs)
        if min_encoding_indices.dim() != 2:
            min_encoding_indices = min_encoding_indices[:, None, None]
            # remove last token from previous indices
            if prev_indices is not None:
                min_encoding_indices = torch.cat([prev_indices, min_encoding_indices], dim=1)
                min_encoding_indices = min_encoding_indices[:, :encoding_tgt.shape[1], :].contiguous()
            min_encoding_indices = min_encoding_indices.squeeze(-1)
        min_encodings = F.one_hot(min_encoding_indices.long(), self.k_way).float()
        # get quantized latent vectors
        if embedding.dim() == 4:
            # enter inside if the embedding has a time dimension (the third dimension)
            # the second dimension is the number of time steps
            z_q = []
            for i in range(min_encodings.shape[1]):
                z_q.append(torch.matmul(min_encodings[:, [i], :], embedding[:, :, i, :]))
            z_q = torch.cat(z_q, dim=1)
        else:
            z_q = torch.matmul(min_encodings, embedding)
        min_encoding_indices = min_encoding_indices.reshape(encoding_tgt.shape[0], -1, 1)
        return z_q, None, None, None, min_encoding_indices, embedding

    def forward(self, src, tgt, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_enc=None, distill=False, **kwargs):
        """
        :param src: encoder input tensor (batch, seq_len, d_model)
        :param tgt: decoder input tensor (batch, seq_len, d_model)
        :param src_mask: temporal attention mask for source (batch, 1, seq_len)
        :param tgt_mask: subsequent temporal attention mask for target (batch, seq_len, seq_len)
        :param space_mask: spatial attention mask for source/target (seq_len, 1, batch)
        :param distill: if True, returns also encoding and decoding outputs for further knowledge distillation
        """
        masked = None
        # enc input: (batch, seq_len, d_model). I'll reuse encoding for cross-attention
        encoder_input = self.encoder_positional(self.encoder_embedding(src), flip=self.flip_pos_encodings)

        encoding, enc_attn_matrices = self.encode(encoder_input, src_mask, space_mask)
        if self.training or kwargs.get('extract_training_samples', False):
            # enc target. The encoding of target is not masked (target mask is refactored inside encode_response)
            encoder_tgt = self.encoder_response_positional(self.encoder_response_embedding(tgt_enc))
            # get tgt temporal mask
            tgt_encode_mask = torch.ones_like(tgt_mask)
            encoding_tgt, enc_tgt_attn_matrices = self.encode_response(encoder_tgt, encoding,
                                                                       tgt_encode_mask,
                                                                       src_mask,
                                                                       tgt_space_mask)

            # proj_encoding_src = self.proj_encoder_src(encoding)
            encoding_tgt = self.proj_encoder_tgt(encoding_tgt)
            z_q, embedding_loss, perplexity, min_encodings, min_encoding_indices, vq_embedding = self.variational_quantization(
                x_enc=encoding, z=encoding_tgt)
        else:
            # sample from the pixelcnn
            prev_indices = kwargs.get('prev_indices', None)
            prev_z = kwargs.get('prev_z', None)
            z_q, embedding_loss, perplexity, min_encodings, min_encoding_indices, vq_embedding = self.variational_through_pixelcnn(
                encoding, tgt, prev_indices, prev_z,
                encoder_input if not self.pixelcnn_train_with_original_src else src,
                src_mask, space_mask, tgt_space_mask)
        z_q = self.unproj_encoder_tgt(z_q)
        # add z to decoder input
        # dec
        tgt_mask = torch.ones_like(tgt_mask)
        decoding, dec_attn_matrices = self.decodify(tgt, encoding, z_q, tgt_mask, src_mask, tgt_space_mask)

        # project decoder output to expected output dimension
        out = self.out_projection(decoding)
        output_dict = {'pred': out, 'z': z_q, 'embedding_loss': embedding_loss, 'encoding': encoding,
                       'indices': min_encoding_indices, 'perplexity': perplexity, 'vq_embedding': vq_embedding,
                       'encoder_input': encoder_input, 'masked_token': masked}

        return output_dict

    def get_encoder_tgt_mask(self, tgt_mask, tgt_space_mask):
        tgt_space_masked = torch.clone(tgt_space_mask)
        tgt_mask_masked = torch.ones_like(tgt_mask)
        return tgt_mask_masked, tgt_space_masked

    def get_last_layer(self):
        return self.out_projection.weight
