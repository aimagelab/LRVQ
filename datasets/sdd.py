# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
import pickle

from functools import lru_cache
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.utils.base_dataset import BaseDataset
import torchvision.transforms as transforms

from datasets.utils.collate_funcs import seq_collate, seq_collate_eval
from utils.conf import base_path_dataset
from utils.loss import PredictionLoss


def get_parser(parser):
    """
    """
    parser.add_argument('--obs_len', type=int, default=8, help='Length of the observed trajectory')
    parser.add_argument('--pred_len', type=int, default=12, help='Length of the predicted trajectory')
    parser.add_argument('--skip', type=int, default=1, help='Skip every n-th frame')
    parser.add_argument('--min_ped', type=int, default=1, help='Minimum number of pedestrians in a scene')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for the dataloader')

    return parser


class StanfordDroneDataset(Dataset):
    """ Dataloder for the Stanford Drone Dataset """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1):

        """
        Args:
        - data_dir: directory containing dataset files in the format '<frame_id> <ped_id> <x> <y>'
        - obs_len: # of observation time-steps
        - pred_len: # of prediction time-steps
        - skip: # of time-steps to skip while making the dataset
        - threshold: minimum error to be considered for non linear traj when using a linear predictor
        - min_ped: minimum # of pedestrians that should be in a sequence/scene
        """

        super(StanfordDroneDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_ped = min_ped
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        original_data = np.load(data_dir, allow_pickle=True)
        self.original_data = original_data
        elab_data_dir = data_dir.replace('.pkl', '_elab.pkl')
        all_data = np.load(data_dir, allow_pickle=True)
        if not os.path.isfile(elab_data_dir):
            # if True:
            print('Elaborating data...')
            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            seq_list_norm = []
            means_list = []
            stds_list = []
            scenes = []
            group = all_data.groupby("sceneId")
            if os.getenv('DEBUG', 'false').lower() == 'true':
                print('DEBUG MODE')
                # group = list(group)[:3]
                group = [list(group)[0], list(group)[0]]
            for scene, data in group:
                # extract valid sequences from the scenario
                num_peds, seqs, seqs_rel, seqs_norm, means, stds = self.__get_data_from_scene(scene, data)

                if not num_peds:
                    continue  # no valid sequences

                num_peds_in_seq.extend(num_peds)
                seq_list.extend(seqs)
                seq_list_rel.extend(seqs_rel)
                seq_list_norm.append(seqs_norm)
                means_list.append(means)
                stds_list.append(stds)
                scenes.append(scene)
            print('Saving elaborated data...')
            with open(elab_data_dir, 'wb') as f:
                pickle.dump([num_peds_in_seq, seq_list, seq_list_rel, seq_list_norm, means_list, stds_list],
                            f)
        else:
            print('Loading elaborated data...')

            num_peds_in_seq, seq_list, seq_list_rel, seq_list_norm, means_list, stds_list = np.load(elab_data_dir,
                                                                                                    allow_pickle=True)

        self.num_seq = len(seq_list)

        seqs_np = np.concatenate(seq_list, axis=0)
        seqs_rel_np = np.concatenate(seq_list_rel, axis=0)
        seqs_norm_np = np.concatenate(seq_list_norm, axis=0)
        means_np = np.concatenate(means_list, axis=0)
        stds_np = np.concatenate(stds_list, axis=0)
        seq_name = []
        for num, gr in enumerate(all_data.groupby("sceneId")):
            seq_name.extend([gr[0]] * len(means_list[num]))
        # indices for splitting data tensors into groups of peds (-> a group contains all the peds in the same seq)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_name = seq_name
        # Convert numpy -> Torch Tensor
        self.src = torch.from_numpy(seqs_np[:, :, :self.obs_len]).type(torch.float)
        self.tgt = torch.from_numpy(seqs_np[:, :, self.obs_len:]).type(torch.float)
        self.src_rel = torch.from_numpy(seqs_rel_np[:, :, :self.obs_len]).type(torch.float)
        self.tgt_rel = torch.from_numpy(seqs_rel_np[:, :, self.obs_len:]).type(torch.float)
        self.src_norm = torch.from_numpy(seqs_norm_np[:, :, :self.obs_len]).type(torch.float)
        self.tgt_norm = torch.from_numpy(seqs_norm_np[:, :, self.obs_len:]).type(torch.float)
        self.means = torch.from_numpy(means_np).type(torch.float)
        self.stds = torch.from_numpy(stds_np).type(torch.float)
        self.traj_scale = 50

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        """
        the output of the first 6 element have dimension (num_peds, 2, seq_len). the last 2 elements have dimension (num_peds, 2)
        """
        start, end = self.seq_start_end[index]
        out = [self.src[start:end, :], self.tgt[start:end, :],
               self.src_rel[start:end, :], self.tgt_rel[start:end, :],
               self.src_norm[start:end], self.tgt_norm[start:end, :],
               self.means[start:end, :], self.stds[start:end, :]]
        return out

    def __get_data_from_scene(self, scene, data):
        num_peds_in_seq = []
        seq_in_curr_file = []
        seq_rel_in_curr_file = []

        scene_name = scene

        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        data['frame'] -= data['frame'].min()

        data['node_id'] = data['trackId'].astype(float)

        # apply data scale as same as PECnet
        data['x'] = data['x'] / 50
        data['y'] = data['y'] / 50

        # Mean Position
        data['x'] = data['x'] - data['x'].mean()
        data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()
        # assign a unique value every 20 frames
        if len(data) > 0:
            for idx in range(data['frame'].min(), data['frame'].max()):
                curr_seq_data = data[(data['frame'] >= idx) & (data['frame'] < idx + self.seq_len)]
                peds_in_curr_seq = curr_seq_data.node_id.unique()
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))

                num_peds_considered = 0  # peds in this sequence with trjs longer or equal to self.seq_len
                _non_linear_ped = []
                # if the current sequence does not contain the desired number of peds skip it
                n = 0
                for _, ped_id in enumerate(peds_in_curr_seq):

                    curr_ped_seq = curr_seq_data[curr_seq_data['node_id'] == ped_id]

                    if len(curr_ped_seq) != self.seq_len:
                        continue
                    assert np.all(np.diff(curr_ped_seq['frame']) == 1)

                    curr_ped_seq = np.transpose(curr_ped_seq[['x', 'y']].values)
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    # put valid sequences in the support vectors
                    curr_seq[num_peds_considered, :, :] = curr_ped_seq
                    curr_seq_rel[num_peds_considered, :, :] = rel_curr_ped_seq

                    num_peds_considered += 1

                # accumulate extracted vectors only if the sequence contains the desired minimum number of peds
                if num_peds_considered >= self.min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_in_curr_file.append(curr_seq[:num_peds_considered])
                    seq_rel_in_curr_file.append(curr_seq_rel[:num_peds_considered])

        if not seq_in_curr_file:
            return [], [], [], np.array([]), np.array([]), np.array([])

        # obtain normalized (standardized) sequences
        seq_rel_in_curr_file_np = np.concatenate(seq_rel_in_curr_file, axis=0)
        means = seq_rel_in_curr_file_np[:, :, 1:].mean((0, 2))
        stds = seq_rel_in_curr_file_np[:, :, 1:].std((0, 2))
        seq_norm_in_curr_file = np.zeros(seq_rel_in_curr_file_np.shape)
        seq_norm_in_curr_file[:, :, 1:] = ((seq_rel_in_curr_file_np[:, :, 1:].swapaxes(1, 2) - means)
                                           / stds).swapaxes(1, 2)

        total_n_seqs = seq_rel_in_curr_file_np.shape[0]
        # every element in seq_in_curr_file has dimension (num_peds, 2, seq_len)
        return num_peds_in_seq, seq_in_curr_file, seq_rel_in_curr_file, seq_norm_in_curr_file, \
            np.expand_dims(means, axis=0).repeat(total_n_seqs, axis=0), \
            np.expand_dims(stds, axis=0).repeat(total_n_seqs, axis=0)


def data_loader(args, path, split, skip=1, shuffle=True, transform=None):
    """
    Returns dataset and relative PyTorch dataloader for the given split.

    :param args: model arguments
    :param path: dataset main directory
    :param split: dataset split
    :return: dataset: all data for given split
    :return loader: PyTorch dataloader for given split
    """

    assert split in ['train', 'validation', 'test']

    npy_files_path = pathlib.Path(path)
    split_path = '{}/{}_trajnet.pkl'.format(npy_files_path, split)

    dataset = StanfordDroneDataset(data_dir=split_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=skip,
                                   min_ped=args.min_ped)
    sampler = None
    if split == 'train':
        collate_fun = seq_collate
    else:
        collate_fun = seq_collate_eval
    loader = DataLoader(dataset,
                        batch_size=args.batch_size if split == 'train' else args.eval_batch_size,
                        num_workers=args.n_workers,
                        shuffle=shuffle,
                        collate_fn=collate_fun,
                        sampler=sampler
                        )

    return dataset, loader, sampler


class SddDatase(BaseDataset):
    NAME = 'sdd'
    TRANSFORM = None

    @lru_cache()
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, None]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        path = os.path.join(base_path_dataset(), 'sdd')

        _, train_loader, train_sampler = data_loader(self.args, path, 'train', skip=self.args.skip, transform=transform)

        _, val_loader, _ = data_loader(self.args, path, 'test', skip=self.args.skip, transform=test_transform,
                                       shuffle=False)

        _, test_loader, _ = data_loader(self.args, path, 'test', skip=self.args.skip, transform=test_transform,
                                        shuffle=False)

        return train_loader, val_loader, test_loader, train_sampler

    def get_normalization_transform(self):
        return None

    def get_loss(self):
        return PredictionLoss()
