# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch


def seq_collate(data):
    """
    Custom collate function.

    :param data: list of tensors containing the data of the sequences/scenes sampled by the PyTorch batch sampler
    :return: tuple: tuple containing collated data
    """

    (src_seq_list, tgt_seq_list, src_seq_rel_list, tgt_seq_rel_list, src_seq_norm_list, tgt_seq_norm_list,
     means_list, stds_list) = zip(*data)

    _len = [len(seq) for seq in src_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    src = torch.cat(src_seq_list, dim=0).transpose(1, 2)
    tgt = torch.cat(tgt_seq_list, dim=0).transpose(1, 2)
    seq_start_end = torch.LongTensor(seq_start_end)

    return tuple([src, tgt, seq_start_end])


def seq_collate_eval(data):
    """
    Custom collate function.

    :param data: list of tensors containing the data of the sequences/scenes sampled by the PyTorch batch sampler
    :return: tuple: tuple containing collated data
    :param return_normalized:
    """

    (src_seq_list, tgt_seq_list, src_seq_rel_list, tgt_seq_rel_list, src_seq_norm_list, tgt_seq_norm_list,
     means_list, stds_list) = zip(*data)

    _len = [len(seq) for seq in src_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    src = torch.cat(src_seq_list, dim=0).transpose(1, 2)
    tgt = torch.cat(tgt_seq_list, dim=0).transpose(1, 2)
    seq_start_end = torch.LongTensor(seq_start_end)

    return tuple([src, tgt, seq_start_end, src, tgt, torch.tensor(0)])