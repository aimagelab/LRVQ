# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.kmeans import get_kmeans_torch


class Subsample:
    def __init__(self, n_samples, method='identity', params=None):
        self.method = method
        self.params = params if params is not None else {}
        self.n_samples = n_samples

    def __call__(self, data: torch.Tensor):
        """
        data: torch.Tensor
            The data to subsample with dimension (N_samples, batch_size, n_features)
        """
        if self.method == 'random':
            return self.random(data)
        elif self.method == 'kmeans':
            return self.kmeans(data)
        elif self.method == 'identity':
            return data
        else:
            raise ValueError(f'Unknown subsampling method: {self.method}')

    def random(self, data: torch.Tensor):
        # apply random subsampling using multinomial torch
        subsampled = []
        m_sample = data.shape[0]
        for i in range(data.shape[1]):
            one_sample = data[:, i, :].reshape(m_sample, -1)
            idx = self._random(one_sample[:, 0])
            subsampled.append(data[idx, i, :])
        return torch.stack(subsampled, dim=1)

    def _random(self, data: torch.Tensor):
        return torch.multinomial(torch.ones_like(data), self.n_samples, replacement=False)

    def kmeans(self, data: torch.Tensor):
        # apply kmeans subsampling
        subsampled = []
        m_sample = data.shape[0]
        for i in range(data.shape[1]):
            one_sample = data[:, i, :].reshape(m_sample, -1)
            idx = self._kmeans(one_sample)
            if self.params.get('return_centers', True):
                subsampled.append(idx.reshape(self.n_samples, data.shape[-2], data.shape[-1]))
            else:
                subsampled.append(subsampled[idx])
        return torch.stack(subsampled, dim=1)

    def _kmeans(self, data: torch.Tensor):
        return get_kmeans_torch(data, self.n_samples, **self.params)
