# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

from matplotlib import pyplot as plt

from models.utils.base_model import BaseModel
import torch
from utils.loss import PredictionLoss
from utils.utilities import (rotate_and_shift_batch, attention_mask, spatial_attention_mask,
                             spatial_attention_mask_adj, normalize_data, parse_bool)


def get_parser(parser) -> ArgumentParser:
    """
    Defines the arguments of the model.
    """
    parser.add_argument('--avoid_last_obs', type=parse_bool, default=False,
                        help='Avoid last obs (0,0) after normalization')
    parser.add_argument('--random_rotation', default=True, type=parse_bool, help='Random rotation')
    parser.add_argument('--spatial_att_adj', type=parse_bool, default=False, help='Use spatial attention mask with adj')
    parser.add_argument('--space_threshold', default=5, type=float, help='Space threshold for spatial attention mask')
    parser.add_argument('--apply_normalization', type=parse_bool, default=False, help='Apply normalization')
    parser.add_argument('--nabs', type=parse_bool, default=True, help='Use nabs')
    return parser


class ScratchTrajectory(BaseModel):
    NAME = 'scratchtrajectory'

    def __init__(self, backbone, loss, args, transform):
        super(ScratchTrajectory, self).__init__(backbone, loss, args, transform)
        self.best_ade = None
        self.obs_len = args.obs_len  # args.obs_len + 1 if self.net.NAME == 'gvt' else
        self.validation_loss = PredictionLoss()

    def preprocess(self, src, tgt, seq_start_end, rotate=True, spatial_att_adj=False, degrees=None):

        _, src_len, _ = src.shape
        means = []
        if self.args.apply_normalization:
            src_normalized, tgt_normalized, means = normalize_data(src, tgt, seq_start_end)
        else:
            src_normalized, tgt_normalized = src, tgt
        # rotate and shift
        trj_norm, shift_value, theta, velocity, acceleration = rotate_and_shift_batch(
            torch.cat((src_normalized, tgt_normalized), dim=1),
            rotate=rotate,
            obs_len=self.obs_len,
            nabs=self.args.nabs,
            dataset_name=self.args.dataset,
            degrees=degrees,
            rotation_type=self.args.rotation_type
        )
        src_norm, tgt_norm = trj_norm[:, :src_len], trj_norm[:, src_len:]
        # after rotation and shifting, the last obs is always (0,0), we maybe want to get rid of that
        # append to src the velocity and acceleration
        encoder_input = src_norm[:, :-1] if self.args.avoid_last_obs else src_norm
        batch, src_len, _ = encoder_input.shape

        # add a binary feature to know if the time-step is the start token (1) or not (0)
        _, tgt_len, _ = tgt_norm.shape
        tgt_tmp = torch.cat((tgt_norm, torch.zeros((batch, tgt_len, 1)).to(self.device)), -1)

        # add start token to tgt
        start_token = torch.Tensor([0., 0., 1.]).unsqueeze(0).unsqueeze(1).repeat(batch, 1, 1).to(self.device)
        decoder_input = torch.cat((start_token, tgt_tmp[:, :-1]), 1)

        # attention masks ('0' for masking)
        src_mask = attention_mask(batch, src_len).to(self.device)
        if spatial_att_adj:
            space_mask = spatial_attention_mask_adj(src, seq_start_end, self.args.space_threshold).to(self.device)
            tgt_space_mask = spatial_attention_mask_adj(tgt, seq_start_end, self.args.space_threshold).to(self.device)
        else:
            space_mask = spatial_attention_mask(seq_start_end).to(self.device).unsqueeze(0).repeat(src_len, 1, 1)
            tgt_space_mask = spatial_attention_mask(seq_start_end).to(self.device).unsqueeze(0).repeat(tgt_len, 1, 1)
        space_mask = space_mask[-src_len:, :, :]
        tgt_mask = attention_mask(batch, tgt_len, subsequent=True).to(self.device)
        # if self.args.add_velocity_and_acceleration:
        #     ...
        return encoder_input, decoder_input, src_mask, tgt_mask, space_mask, tgt_space_mask, tgt_norm, shift_value, theta, means
