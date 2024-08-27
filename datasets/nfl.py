# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from functools import lru_cache
from typing import Tuple, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

from torchvision.transforms import transforms

from datasets import BaseDataset
from datasets.utils.collate_funcs import seq_collate, seq_collate_eval
from utils.conf import base_path_dataset
from utils.loss import PredictionLoss


def get_parser(parser):
    """
    """
    parser.add_argument('--obs_len', type=int, default=8, help='Length of the observed trajectory')
    parser.add_argument('--pred_len', type=int, default=16, help='Length of the predicted trajectory')
    parser.add_argument('--skip', type=int, default=1, help='Skip every n-th frame')
    parser.add_argument('--min_ped', type=int, default=1, help='Minimum number of pedestrians in a scene')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for the dataloader')
    parser.add_argument('--dataset_name', type=str, default='20240210142031', help='Name of the dataset')
    return parser


class NFLDs(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_root, obs_len=5, pred_len=10, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(NFLDs, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        self.trajs = np.load(data_root)  # (N,T,2,23)
        # from yards to meters
        self.trajs *= 0.9144
        self.trajs = self.trajs.transpose(0, 1, 3, 2)  # (N,T,23,2)
        if os.getenv('DEBUG', 'false').lower() == 'true':
            self.trajs = self.trajs[:10]
        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)
        self.traj_means = [14, 7.5]
        self.traj_scale = 25
        # create cum_start_idx. the traj_abs are in the shape of (N, 30, 11, 2), where 30 is the seq_len, 11 is the number of players, 2 is the x and y coordinates
        # squeezing the traj_abs to (N * 11, 30, 2) and create cum_start_idx [0, 11, 22, 33, ...]
        total_num = self.traj_abs.shape[0]
        agent_num = self.traj_abs.shape[2]
        my_traj = self.traj_abs.permute(0, 2, 1, 3).reshape(-1, self.seq_len, 2)
        cum_start_idx = np.cumsum([0] + [agent_num for _ in range(total_num)])
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        self.src = my_traj[:, :self.obs_len].permute(0, 2, 1).type(torch.float) / self.traj_scale
        self.tgt = my_traj[:, self.obs_len:].permute(0, 2, 1).type(torch.float) / self.traj_scale
        self.src_rel = torch.Tensor()
        self.tgt_rel = torch.Tensor()
        self.src_norm = torch.Tensor()
        self.tgt_norm = torch.Tensor()
        self.means = torch.Tensor()
        self.stds = torch.Tensor()

        self.actor_num = agent_num
        self.num_seq = total_num
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        """
        the output of the first 6 element have dimension (num_peds, 2, seq_len). the last 2 elements have dimension (num_peds, 2)
        """
        start, end = self.seq_start_end[index]
        out = [self.src[start:end, :], self.tgt[start:end, :],
               self.src_rel, self.tgt_rel,
               self.src_norm, self.tgt_norm,
               self.means, self.stds]
        return out

    def test_transformation(self):
        scene_idx = np.random.randint(0, self.num_seq)
        scene_from_traj = self.traj_abs[scene_idx]
        idx_from_seq_start_end = self.seq_start_end[scene_idx]
        scene_src_from_src = self.src[idx_from_seq_start_end[0]:idx_from_seq_start_end[1]]
        scene_tgt_from_tgt = self.tgt[idx_from_seq_start_end[0]:idx_from_seq_start_end[1]]
        scene_from_elab = torch.cat([scene_src_from_src, scene_tgt_from_tgt], dim=2)
        scene_from_elab = scene_from_elab.permute(2, 0, 1) * self.traj_scale
        assert torch.allclose(scene_from_traj, scene_from_elab)

    def create_animation(self, scene_idx, save_path, actors=-1):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('scene {}'.format(scene_idx))
        idx_start, idx_end = self.seq_start_end[scene_idx]
        scene_src = self.src[idx_start:idx_end]
        scene_tgt = self.tgt[idx_start:idx_end]
        assert scene_src.shape[0] == scene_tgt.shape[0]
        scene = torch.cat([scene_src, scene_tgt], dim=2)
        for i in range(scene[:actors].shape[0]):
            ax.plot(scene[i, 0], scene[i, 1], linewidth=.7, color='green', marker='o')

        def update(frame):
            ax.clear()
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('scene {}'.format(scene_idx))
            for i in range(scene[:actors].shape[0]):
                ax.plot(scene[i, 0, :frame], scene[i, 1, :frame], linewidth=.7, color='green', marker='o')

        ani = FuncAnimation(fig, update, frames=range(self.seq_len), interval=100)
        ani.save(save_path, writer='ffmpeg', fps=10)
        plt.show()


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
    split_path = '{}/{}_nfl_{}.npy'.format(npy_files_path, args.dataset_name, split)

    dataset = NFLDs(data_root=split_path, obs_len=args.obs_len, pred_len=args.pred_len, training=split == 'train')
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


class NFLDataset(BaseDataset):
    NAME = 'nfl'
    TRANSFORM = None

    @lru_cache()
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, None]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        path = os.path.join(base_path_dataset(), 'nfl')

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


# # https://github.com/a-vhadgar/Big-Data-Bowl/blob/master/README.md
