# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from copy import deepcopy

import torch
import random
import logging
import numpy as np
from torch import nn
from torch.functional import F


def set_logger(run_name=None, log_dir=None, console=False, overwrite=False):
    """
    Set the logger.

    :param run_name: name of the current run/experiment
    :param log_dir: directory where writing eventual text log
    :param console: if True write to console (e.g. in debugging), otherwise write to file
    :param overwrite: if True, overwrite the log file when already existing
    """
    assert (
                   run_name is not None and log_dir is not None) or console is True, 'Either run_name or console must be provided'
    format = '[%(levelname)s | %(asctime)s]: %(message)s'  # logger string format
    dateformat = '%Y/%m/%d %H:%M'
    handlers = []
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_dir is not None:
        file_name = log_dir.absolute() / '{}.log'.format(run_name)
        # where to log
        file_mode = 'w' if overwrite else 'a'
        handlers.append(logging.FileHandler(file_name, mode=file_mode))
    if console:
        handlers.append(logging.StreamHandler())

    # set formats
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt=format, datefmt=dateformat))
        logger.addHandler(handler)


def rotate_and_shift_batch(data, obs_len=8, nabs=True, rotate=False,
                           dataset_name='sdd', degrees=None, rotation_type='normal'):
    """
    Random ration and zero shifting. (same strategy as: https://github.com/zhangpur/SR-LSTM)

    :param data: trajectories expressed in absolute coordinates (batch, seq_len, xy)
    :param obs_len: number of observation steps
    :param rotate: if True, apply a random rotation to the whole batch
    """
    assert rotation_type in ['normal', 'red'], 'Rotation type must be either normal or red'
    rotated = data.clone()
    th = 0

    # rotate all the trajectories in the batch with a random theta
    if rotate:
        rad = np.pi
        if degrees is not None:
            rad = np.deg2rad(degrees)
        if rotation_type == 'normal':
            th = torch.tensor(random.random() * rad, device=data.device)
        else:
            # apply a random rotation for each element in the batch ensuring that th is very small
            assert degrees <= 5, 'red rotation should be used with small degrees'
            th = torch.rand((data.shape[0], 1), device=data.device) * rad
        rotated[:, :, 0] = data[:, :, 0] * torch.cos(th) - data[:, :, 1] * torch.sin(th)
        rotated[:, :, 1] = data[:, :, 0] * torch.sin(th) + data[:, :, 1] * torch.cos(th)
    # velocity, acceleration = calc_vel_acc(rotated, dataset_name=dataset_name)
    velocity, acceleration = torch.ones_like(rotated), torch.ones_like(rotated)
    # get "shift value" (i.e. the absolute position in the last observation time-step)
    # the relative positions will be the displacements wrt to this value
    s = rotated[:, obs_len - 1:obs_len] if nabs else rotated[:, 0:1]

    return rotated - s, s, th, velocity, acceleration


def make_continuous_copy(alpha):
    alpha = (alpha + torch.pi) % (2.0 * torch.pi) - torch.pi
    continuous_x = torch.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (torch.sign(alpha[i]) == torch.sign(alpha[i - 1])) and torch.abs(alpha[i]) > torch.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - torch.sign(
                (alpha[i] - alpha[i - 1])) * 2 * torch.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


def unrotate_and_unshift_batch(data, s, th, nabs=True, unrotate=False):
    """
    Reverse random ration and zero shifting.

    :param data: rotated and shifted trajectories (batch, seq_len, xy)
    :param s: shift value
    :param th: theta used to random rotate the original trajectories
    :param unrotate: if True, reverse the random rotation using the theta provided
    :return orig: original trajectories
    """

    # always unshift
    data = data[:, :, :2] + s if nabs else data[:, :, :2].cumsum(dim=1) + s
    orig = data.clone()

    # maybe reverse rotation
    if unrotate:
        orig[:, :, 0] = data[:, :, 0] * torch.cos(-th) - data[:, :, 1] * torch.sin(-th)
        orig[:, :, 1] = data[:, :, 0] * torch.sin(-th) + data[:, :, 1] * torch.cos(-th)

    return orig


def rotate_and_normalize_batch(data, rotate=False):
    """
    Random ration and relative coordinates.

    :param data: trajectories expressed in absolute coordinates (batch, seq_len, xy)
    :param obs_len: number of observation steps
    :param rotate: if True, apply a random rotation to the whole batch
    """

    rotated = data.clone()
    th = 0

    # rotate all the trajectories in the batch with a random theta
    if rotate:
        th = random.random() * np.pi
        rotated[:, :, 0] = data[:, :, 0] * np.cos(th) - data[:, :, 1] * np.sin(th)
        rotated[:, :, 1] = data[:, :, 0] * np.sin(th) + data[:, :, 1] * np.cos(th)

    # find first not-nan element for each trj
    s_indexes = torch.argmax((~torch.isnan(rotated[:, :, 0])).double(), dim=1)
    s = torch.gather(rotated, 1, s_indexes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2))

    # relative coordinates
    B, T, C = rotated.shape
    rel_data = rotated[:, 1:] - rotated[:, :-1]
    rel_data[rel_data != rel_data] = 0
    rel_data = torch.cat((torch.zeros(B, 1, C).to(rotated.device), rel_data), dim=1)

    return rel_data, s, th


def unrotate_and_unnormalize_batch(data, s, th, unrotate=False):
    """
    Reverse random ration and zero shifting.

    :param data: rotated and shifted trajectories (batch, seq_len, xy)
    :param s: shift value
    :param th: theta used to random rotate the original trajectories
    :param unrotate: if True, reverse the random rotation using the theta provided
    :return orig: original trajectories
    """

    # always unshift
    data = data.cumsum(dim=1) + s
    orig = data.clone()

    # maybe reverse rotation
    if unrotate:
        orig[:, :, 0] = data[:, :, 0] * np.cos(-th) - data[:, :, 1] * np.sin(-th)
        orig[:, :, 1] = data[:, :, 0] * np.sin(-th) + data[:, :, 1] * np.cos(-th)

    return orig


def average_displacement_error(pred_trj, pred_trj_gt, mode='sum'):
    """
    Calc ADE metric.

    :param pred_trj: predicted trajectories (batch, seq_len, 2)
    :param pred_trj_gt: Ground-truth trajectories (batch, seq_len, 2)
    :param mode: 'raw' (one ade per elem in batch) or 'sum' (sum of all ades inside batch)
    :return: ade (or ades) for current batch
    """

    loss = (pred_trj_gt - pred_trj) ** 2

    if mode == 'sum':
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
        return torch.sum(loss)
    elif mode == 'raw':
        loss = torch.sqrt(loss.sum(dim=2))
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    """
    Calc FDE metric.

    :param pred_pos: predicted final position (batch, 2)
    :param pred_pos_gt: ground-truth final position (batch, 2)
    :param mode: 'raw' (one fde per elem in batch) or 'sum' (sum of all fdes inside batch)
    :return fde (or fde) for current batch
    """

    loss = (pred_pos - pred_pos_gt) ** 2
    loss = torch.sqrt(torch.sum(loss, dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def attention_mask(batch, n_keys, n_queries=None, subsequent=False, dropattention=0.):
    """
    Computes attention mask for a given batch of sequences.

    :param batch: batch size
    :param length: length of the sequences
    :param subsequent: if True, masks subsequently future positions (e.g. decoder mask for tgt)
    """

    n_queries = n_keys if n_queries is None else n_queries

    if dropattention == 0.:
        mask = torch.tril(torch.ones((batch, n_queries, n_keys)), diagonal=0) if subsequent else torch.ones(batch, 1,
                                                                                                            n_keys)
    else:
        drop_mask = (torch.ones(batch, n_queries, n_keys).uniform_() >= dropattention)
        mask = torch.logical_and(torch.tril(torch.ones(batch, n_queries, n_keys)),
                                 drop_mask) if subsequent else drop_mask

        # to not break attention softmax, make sure each row in QxK mask has at least one True
        while not mask.any(-1).all():
            drop_mask = (torch.ones(batch, n_queries, n_keys).uniform_() >= dropattention)
            mask = torch.logical_and(torch.tril(torch.ones(batch, n_queries, n_keys)),
                                     drop_mask) if subsequent else drop_mask
    return mask


def attention_mask_seq(batch, seq_len, n_obs):
    """
    Computes attention mask that has first n_obs position set to 1 and the rest is masked.
    the temp_mask in output has shape (batch, 1, seq_len)
    """
    src_mask = torch.ones(n_obs, n_obs)
    y_causal_mask = torch.tril(torch.ones(seq_len - n_obs, seq_len - n_obs), diagonal=0)
    # padding mask with 1s to arrive at the correct shape
    causal_mask = F.pad(src_mask, (0, 0, seq_len - n_obs, 0), value=1)
    y_causal_mask = F.pad(y_causal_mask, (0, 0, n_obs, 0), value=0)
    causal_mask = torch.cat([causal_mask, y_causal_mask], dim=1).repeat(batch, 1, 1)
    return causal_mask


def spatial_attention_mask(seq_start_end):
    """"
    Computes spatial attention mask for a given batch of sequences.

    :param seq_start_end: tensor containing temporal sequences start and end indices (num_seqs, 2)
    """
    n_seqs = seq_start_end[-1][-1]
    # mask = torch.zeros((n_seqs, n_seqs))
    # for start, end in seq_start_end:
    #     mask[start:end, start:end] = 1

    start, end = seq_start_end[:, 0], seq_start_end[:, 1]

    indices = torch.arange(n_seqs).to(start.device)
    mask = ((indices[:, None] >= start[None, :]) & (indices[:, None] < end[None, :])).float() @ \
           ((indices >= start[:, None]) & (indices < end[:, None])).float()

    return mask


def spatial_attention_mask_adj(trj, seq_start_end, threshold=5.0):
    """"
    Computes spatial attention mask for a given batch of sequences.

    :param seq_start_end: tensor containing temporal sequences start and end indices (num_seqs, 2)
    """
    _, T, _ = trj.shape
    n_seqs = seq_start_end[-1][-1]
    mask = torch.zeros((T, n_seqs, n_seqs))

    for start, end in seq_start_end:
        scene_dists = torch.cdist(trj[start:end].transpose(0, 1), trj[start:end].transpose(0, 1), p=2)
        scene_dists[scene_dists != scene_dists] = np.inf
        mask[:, start:end, start:end] = scene_dists <= threshold

    return mask


def normalize_data(src, tgt, seq_start_end):
    means = []
    new_src = []
    new_tgt = []
    for i in seq_start_end:
        means.append((src[i[0]:i[1]] / 50).mean((0, 1)))
        new_src.append(src[i[0]:i[1]] / 50 - means[-1])
        new_tgt.append(tgt[i[0]:i[1]] / 50 - means[-1])
    src = torch.cat(new_src, 0)
    tgt = torch.cat(new_tgt, 0)
    return src, tgt, means


def denormalize_data(tgt, means, seq_start_end):
    new_tgt = []
    for n, i in enumerate(seq_start_end):
        new_tgt.append(tgt[i[0]:i[1]] + means[n].unsqueeze(0))
    tgt = torch.cat(new_tgt, 0) * 50
    return tgt


def load_checkpoint(args):
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        for arg in args.args_to_update + ['checkpoint', 'checkpoint_path', 'wandb', 'only_eval']:
            checkpoint['args'].pop(arg, None)
        args.__dict__.update(checkpoint['args'])
        original_model_state_dict = deepcopy(checkpoint['model_state_dict'])
        for mod in args.checkpoint_modules_to_exclude:
            for k in original_model_state_dict.keys():
                if mod in k:
                    print(f'Excluding {k} from checkpoint')
                    checkpoint['model_state_dict'].pop(k, None)
        args.checkpoint = checkpoint
        if getattr(args, 'resume_training', False):
            args.optimizer_state_dict = checkpoint['optimizer_state_dict']
        else:
            args.optimizer_state_dict = None
    else:
        args.checkpoint = None

    return args


def init_weights(params):
    for p in params:
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def parse_bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def parse_dict(s, params):
    if s is None or s == '':
        return {}
    else:
        my_dict = s.split(',')
        my_dict = {x.split(':')[0]: x.split(':')[1] for x in my_dict}
        for k, v in my_dict.items():
            assert k in params, f'Parameter {k} not in params'
        return my_dict


def get_denormalize_fun(loader):
    try:
        denormalize = loader.dataset.denormalize
    except AttributeError:
        denormalize = lambda x, *args, **kwargs: x
        print('No denormalize function found in dataset. Use pred absolute values.')
    return denormalize


def get_module(cls):
    """
    Returns the underlying module of a torch.nn.DataParallel instance.
    """
    try:
        return cls.net.module
    except AttributeError:
        return cls.net


def print_arguments(args: dict):
    print("*" * 91)
    print(" " * 39 + "Script Arguments" + " " * 38)
    print("*" * 91)
    for arg, value in args.items():
        try:
            print(f"* {arg:<35}: {value:<51} *")
        except:
            print(f"* {arg:<35}: {str(value):<51} *")
    print("*" * 91)
